from pathlib import Path

import simplejson as json
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from loguru import logger

from justatom.modeling.div import loss_per_head_sum
from justatom.modeling.mask import IHead, ILanguageModel
from justatom.processing.mask import IProcessor
from justatom.running.mask import IModelRunner


class EncoderRunner(IModelRunner, torch.nn.Module):
    def __init__(
        self,
        model: ILanguageModel,
        prediction_heads: list[IHead],
        device="cpu",
        processor: IProcessor | None = None,
    ):
        super(EncoderRunner, self).__init__()  # noqa: UP008
        self.model = model
        self.prediction_heads = prediction_heads or []
        self.device = device
        self.dropout = torch.nn.Dropout(0.1)
        self.loss_aggregation_fn = loss_per_head_sum
        self.config = dict()
        self.processor = processor
        self.to(device)

    def to(self, device):
        logger.info(f"Moving to device {str(device)}")
        if self.model is None:
            raise RuntimeError("EncoderRunner model has not been initialised")
        self.model.to(device)
        for i, mod in enumerate(self.prediction_heads):
            if mod.device != device:
                logger.info(f"Moving head[{str(i)}].to[device={device}]")
                mod.to(device)
            if mod.loss.device != device:
                logger.info(f"Moving head[{str(i)}].LOSS.to[device={device}]")
        super(EncoderRunner, self).to(device)  # noqa: UP008

    @classmethod
    def load(cls, data_dir: Path | str, config=None, **props):
        # model_config.json supposed to be present in directory
        _model_config = Path(data_dir) / "runner_config.json"
        assert _model_config.exists(), logger.error(f"The model file is not found for klass=[{cls.__name__}]")
        model = ILanguageModel.load(Path(data_dir))
        if config is None:
            _runner_config = Path(data_dir) / "runner_config.json"
            with open(_runner_config) as f:
                config = json.load(f)
        heads = []
        heads_dir = Path(data_dir) / "heads"
        if heads_dir.is_dir():
            n_dirs = len(list(heads_dir.iterdir()))
            for idx in range(n_dirs):
                head_path = heads_dir / f"head_{idx}"
                hi = IHead.load(head_path)
                heads.append(hi)
        return cls(model=model, prediction_heads=heads)

    def save(self, save_dir: str):
        """
        Dumps the config to the .json file

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        super().save(save_dir)
        if self.processor is not None:
            self.processor.save(save_dir)
        if len(self.prediction_heads) > 0:
            # TODO: Добавить создание под-директории <heads> для весов голов. + конфиг
            heads_dir = Path(save_dir) / "heads"
            for idx, head in enumerate(self.prediction_heads):
                head.save(str(heads_dir / f"head_{idx}"))

    def logits_to_loss_per_head(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Collect losses from each prediction head.

        :param logits: Logits, can vary in shape and type, depending on task.
        :return: The per sample per prediciton head loss whose first two dimensions
                 have length n_pred_heads, batch_size.
        """
        all_losses = []
        for head, logits_for_one_head, labels_for_one_head in zip(self.prediction_heads, logits, labels, strict=False):
            # check if PredictionHead connected to Processor
            assert hasattr(head, "label_tensor_name"), (
                f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model"
                " with the processor through either 'model.connect_heads_with_processor(processor.tasks)'"
                " or by passing the processor to the Adaptive Model?"
            )
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head, labels=labels_for_one_head))
        return all_losses

    def logits_to_loss(self, logits: torch.Tensor, global_step: int | None = None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: Logits, can vary in shape and type, depending on task.
        :param global_step: Number of current training step.
        :param kwargs: Placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :return: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def prepare_labels(self, labels):
        """
        Label conversion to original label space, per prediction head.

        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :return: labels in the right format
        """
        all_labels = []
        # for head, label_map_one_head in zip(self.prediction_heads):
        #     labels = head.prepare_labels(label_map=label_map_one_head, **kwargs)
        #     all_labels.append(labels)
        for head in self.prediction_heads:
            labels = head.prepare_labels(labels)
            all_labels.append(labels)
        return all_labels

    def maybe_norm(self, xs: torch.Tensor, norm: bool) -> torch.Tensor:
        if norm:
            return F.normalize(xs, p=2, dim=len(xs.shape) - 1)
        return xs

    def forward(self, batch, norm: bool = True, **props):
        # Run forward pass of (multiple) prediction heads using the output from above
        if self.model is None:
            raise RuntimeError("EncoderRunner model has not been initialised")
        Q = self.model(**batch, **props)
        all_logits = []
        for Qi in Q:
            if len(self.prediction_heads) > 0:
                for head in self.prediction_heads:
                    all_logits.append(self.maybe_norm(head(self.dropout(Qi)), norm=norm))
            else:  # If no head is initialized => simple forward pass of a model
                all_logits.append(self.maybe_norm(self.dropout(Qi), norm=norm))

        return all_logits


class BiEncoderRunner(IModelRunner, nn.Module):
    """
    Base Class for implementing M=2 `LanguageModel`. One model encodes queries, the other passages.
    There can be prediction heads on top that minimize similarity loss between positive query-passage pairs and maximise between negative pairs.
    """

    def __init__(
        self,
        query_model: ILanguageModel,
        passage_model: ILanguageModel,
        prediction_heads: list[IHead] | None = None,
        embeds_dropout_prob: float = 0.1,
        device: str = "cpu",
    ):
        self.query_model = query_model.to(device)
        self.query_model_out_dims = query_model.output_dims
        self.query_dropout = nn.Dropout(embeds_dropout_prob)

        self.passage_model = passage_model.to(device)
        self.passage_model_out_dims = passage_model.output_dims
        self.passage_dropout = nn.Dropout(embeds_dropout_prob)
        self.prediction_heads = [ph.to(device) for ph in prediction_heads] if prediction_heads is not None else []

        self.loss_aggregation_fn = loss_per_head_sum

    def save(
        self,
        save_dir: Path,
        lm1_name: str = "query_model",
        lm2_name: str = "passage_model",
    ):
        """
        Saves the 2 language model weights and respective config_files in directories
        - query_model
        - passage_model
        within save_dir.

        :param save_dir: Path to save the M2LMRunner to.
        """
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(Path.joinpath(save_dir, Path(lm1_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm1_name)))
        if not os.path.exists(Path.joinpath(save_dir, Path(lm2_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm2_name)))
        self.query_model.save(Path.joinpath(save_dir, Path(lm1_name)))
        self.passage_model.save(Path.joinpath(save_dir, Path(lm2_name)))
        for i, ph in enumerate(self.prediction_heads):
            logger.info("prediction_head saving")
            ph.save(save_dir, i)
        # TODO: Save runner config in the directory specifying the `__class__.__name__` to load afterwards.

    def forward(self, batch, **kwargs):
        pass

    def forward_lm(
        self,
        query_input_ids: torch.Tensor | None = None,
        query_segment_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        passage_input_ids: torch.Tensor | None = None,
        passage_segment_ids: torch.Tensor | None = None,
        passage_attention_mask: torch.Tensor | None = None,
    ):
        """
        Forward pass for the 2 `LanguageModel` models.

        :param kwargs: Holds all arguments that need to be passed to the language models.
        :return: 2 tensors of pooled_output from the 2 language models.
        """
        pooled_output = [None, None]

        if query_input_ids is not None and query_segment_ids is not None and query_attention_mask is not None:
            pooled_output1, _ = self.query_model(
                input_ids=query_input_ids,
                segment_ids=query_segment_ids,
                attention_mask=query_attention_mask,
            )
            pooled_output[0] = pooled_output1

        if passage_input_ids is not None and passage_segment_ids is not None and passage_attention_mask is not None:
            max_seq_len = passage_input_ids.shape[-1]
            passage_input_ids = passage_input_ids.view(-1, max_seq_len)
            passage_attention_mask = passage_attention_mask.view(-1, max_seq_len)
            passage_segment_ids = passage_segment_ids.view(-1, max_seq_len)

            pooled_output2, _ = self.passage_model(
                input_ids=passage_input_ids,
                segment_ids=passage_segment_ids,
                attention_mask=passage_attention_mask,
            )
            pooled_output[1] = pooled_output2

        return tuple(pooled_output)


class GammaHybridRunner(EncoderRunner):
    """
    Implementation for `GammaHybrid` forward pass.
    """

    def __init__(
        self,
        model: ILanguageModel,
        prediction_heads: list[IHead],
        device="cpu",
        processor: IProcessor | None = None,
        include_semantic_gamma: bool = True,
        include_keywords_gamma: bool = True,
        gamma_joint: bool = False,
        semantic_gamma: float = 0.5,
        keywords_gamma: float = 1.5,
        activation_fn: str = "sigmoid",
    ):
        super(GammaHybridRunner, self).__init__(
            model=model,
            prediction_heads=prediction_heads,
            device=device,
            processor=processor,
        )  # noqa: UP008

        if not include_semantic_gamma and not include_keywords_gamma:
            raise ValueError("Both include_semantic_gamma and include_keywords_gamma are False. Nothing to calibrate.")
        if gamma_joint and (not include_semantic_gamma or not include_keywords_gamma):
            raise ValueError("gamma_joint=True requires both include_semantic_gamma and include_keywords_gamma to be True.")

        self.gamma_joint = gamma_joint
        self.include_semantic_gamma = include_semantic_gamma
        self.include_keywords_gamma = include_keywords_gamma
        self.activation_fn = activation_fn
        self.activation = self._resolve_activation(activation_fn)
        if self.model is None:
            raise RuntimeError("GammaHybridRunner model has not been initialised")
        alpha_hidden_dim = max(32, min(256, int(self.model.output_dims) // 2))
        self.alpha_head_hidden_dim = alpha_hidden_dim
        self.alpha_head = torch.nn.Sequential(
            torch.nn.Linear(int(self.model.output_dims), alpha_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(alpha_hidden_dim, 1),
        )
        self.alpha_head.to(device)

        self.gamma1 = torch.nn.Parameter(
            torch.tensor([semantic_gamma], dtype=torch.float32, device=device),
            requires_grad=include_semantic_gamma,
        )
        self.gamma2 = torch.nn.Parameter(
            torch.tensor([keywords_gamma], dtype=torch.float32, device=device),
            requires_grad=include_keywords_gamma,
        )

    @staticmethod
    def _resolve_activation(name: str):
        normalized = str(name).strip().lower()
        mapping = {
            "sigmoid": torch.nn.Sigmoid(),
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "identity": torch.nn.Identity(),
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported gamma activation_fn={name}. Use one of {','.join(mapping.keys())}")
        return mapping[normalized]

    def gamma_parameters(self) -> list[torch.nn.Parameter]:
        if self.gamma_joint:
            return []
        params: list[torch.nn.Parameter] = []
        if self.include_semantic_gamma:
            params.append(self.gamma1)
        if self.include_keywords_gamma:
            params.append(self.gamma2)
        return params

    def alpha_parameters(self) -> list[torch.nn.Parameter]:
        if not self.gamma_joint:
            return []
        return list(self.alpha_head.parameters())

    def mixing_parameters(self) -> list[torch.nn.Parameter]:
        if self.gamma_joint:
            return self.alpha_parameters()
        return self.gamma_parameters()

    def gamma_weights(self) -> tuple[float, float]:
        semantic_weight = float(self.activation(self.gamma1).detach().item()) if self.include_semantic_gamma else 1.0
        keywords_weight = float(self.activation(self.gamma2).detach().item()) if self.include_keywords_gamma else 1.0
        return semantic_weight, keywords_weight

    def gamma_payload(self) -> dict:
        semantic_weight, keywords_weight = self.gamma_weights()
        return {
            "mode": "query-alpha" if self.gamma_joint else "scalar-gamma",
            "gamma_joint": self.gamma_joint,
            "activation_fn": self.activation_fn,
            "alpha_head": {
                "hidden_dim": self.alpha_head_hidden_dim,
            },
            "semantic_gamma": {
                "enabled": self.include_semantic_gamma,
                "raw": float(self.gamma1.item()),
                "effective": semantic_weight,
            },
            "keywords_gamma": {
                "enabled": self.include_keywords_gamma,
                "raw": float(self.gamma2.item()),
                "effective": keywords_weight,
            },
        }

    def alpha_weights(self, query_vectors: torch.Tensor) -> torch.Tensor:
        if not self.gamma_joint:
            raise RuntimeError("alpha_weights() is only available when gamma_joint=True")
        return torch.sigmoid(self.alpha_head(query_vectors))

    def mix_scores(
        self,
        semantic_scores: torch.Tensor,
        lexical_scores: torch.Tensor,
        query_vectors: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        details: dict[str, float] = {}
        if self.gamma_joint:
            if query_vectors is None:
                raise ValueError("query_vectors must be provided when gamma_joint=True")
            alpha = self.alpha_weights(query_vectors)
            mixed_scores = alpha * semantic_scores + (1.0 - alpha) * lexical_scores
            details = {
                "AlphaMean": float(alpha.detach().mean().item()),
                "AlphaStd": float(alpha.detach().std(unbiased=False).item()),
                "AlphaMin": float(alpha.detach().min().item()),
                "AlphaMax": float(alpha.detach().max().item()),
            }
            return (mixed_scores, details) if return_details else mixed_scores

        semantic_weight = self.activation(self.gamma1) if self.include_semantic_gamma else 1.0
        keywords_weight = self.activation(self.gamma2) if self.include_keywords_gamma else 1.0
        mixed_scores = semantic_weight * semantic_scores + keywords_weight * lexical_scores
        return (mixed_scores, details) if return_details else mixed_scores

    def save(self, save_dir: str):
        self.config = {
            **(self.config or {}),
            "gamma_hybrid": self.gamma_payload(),
        }
        super().save(save_dir)
        if self.gamma_joint:
            torch.save(self.alpha_head.state_dict(), Path(save_dir) / "alpha_gate.pt")

    @classmethod
    def load(cls, data_dir: Path | str, config=None, **props):
        _model_config = Path(data_dir) / "runner_config.json"
        assert _model_config.exists(), logger.error(f"The model file is not found for klass=[{cls.__name__}]")

        model = ILanguageModel.load(Path(data_dir))

        if config is None:
            _runner_config = Path(data_dir) / "runner_config.json"
            with open(_runner_config) as f:
                config = json.load(f)

        heads = []
        heads_dir = Path(data_dir) / "heads"
        if heads_dir.is_dir():
            n_dirs = len(list(heads_dir.iterdir()))
            for idx in range(n_dirs):
                head_path = heads_dir / f"head_{idx}"
                hi = IHead.load(head_path)
                heads.append(hi)

        gamma_hybrid = config.get("gamma_hybrid", {}) if isinstance(config, dict) else {}
        semantic_cfg = gamma_hybrid.get("semantic_gamma", {})
        keywords_cfg = gamma_hybrid.get("keywords_gamma", {})
        runner = cls(
            model=model,
            prediction_heads=heads,
            include_semantic_gamma=semantic_cfg.get("enabled", True),
            include_keywords_gamma=keywords_cfg.get("enabled", True),
            gamma_joint=gamma_hybrid.get("gamma_joint", False),
            semantic_gamma=semantic_cfg.get("raw", 0.5),
            keywords_gamma=keywords_cfg.get("raw", 1.5),
            activation_fn=gamma_hybrid.get("activation_fn", "sigmoid"),
        )
        alpha_gate_path = Path(data_dir) / "alpha_gate.pt"
        if runner.gamma_joint and alpha_gate_path.exists():
            runner.alpha_head.load_state_dict(torch.load(alpha_gate_path, map_location="cpu"))
        return runner
