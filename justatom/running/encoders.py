from pathlib import Path
import math

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

    @staticmethod
    def _activation_module(name: str) -> nn.Module:
        normalized = str(name).strip().lower()
        if normalized == "gelu":
            return nn.GELU()
        if normalized == "relu":
            return nn.ReLU()
        if normalized in {"silu", "swish"}:
            return nn.SiLU()
        if normalized == "tanh":
            return nn.Tanh()
        raise ValueError("alpha_head_activation must be one of: gelu, relu, silu, tanh")

    @classmethod
    def _build_scalar_head(
        cls,
        *,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        dropout: float,
        activation: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for _ in range(max(int(hidden_layers), 0)):
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(cls._activation_module(activation))
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _last_linear(module: nn.Sequential) -> nn.Linear:
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                return layer
        raise RuntimeError("scalar head has no Linear layer")

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
        alpha_residual: bool = False,
        alpha_prior: float = 0.5,
        alpha_residual_scale: float = 0.25,
        alpha_head_hidden_dim: int | None = None,
        alpha_head_layers: int = 1,
        alpha_head_dropout: float = 0.0,
        alpha_head_activation: str = "gelu",
        alpha_include_doc: bool | None = None,
        query_diagonal_gate: bool = False,
        query_diagonal_gate_scale: float = 0.25,
        query_diagonal_gate_mode: str = "raw",
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
        if query_diagonal_gate and not gamma_joint:
            raise ValueError("query_diagonal_gate=True requires gamma_joint=True.")

        self.gamma_joint = gamma_joint
        self.include_semantic_gamma = include_semantic_gamma
        self.include_keywords_gamma = include_keywords_gamma
        self.activation_fn = activation_fn
        self.activation = self._resolve_activation(activation_fn)
        self.alpha_residual = bool(alpha_residual)
        self.alpha_prior = float(alpha_prior)
        self.alpha_residual_scale = float(alpha_residual_scale)
        if not 0.0 < self.alpha_prior < 1.0:
            raise ValueError(f"alpha_prior must be strictly between 0 and 1, got {self.alpha_prior}")
        if self.alpha_residual_scale < 0.0:
            raise ValueError(f"alpha_residual_scale must be >= 0, got {self.alpha_residual_scale}")
        self.alpha_prior_logit = math.log(self.alpha_prior / (1.0 - self.alpha_prior))
        self.query_diagonal_gate = query_diagonal_gate
        self.query_diagonal_gate_scale = float(query_diagonal_gate_scale)
        self.query_diagonal_gate_mode = self._resolve_query_diagonal_gate_mode(query_diagonal_gate_mode)
        if self.model is None:
            raise RuntimeError("GammaHybridRunner model has not been initialised")
        alpha_hidden_dim = (
            max(32, min(256, int(self.model.output_dims) // 2))
            if alpha_head_hidden_dim is None
            else int(alpha_head_hidden_dim)
        )
        if alpha_hidden_dim < 1:
            raise ValueError(f"alpha_head_hidden_dim must be >= 1, got {alpha_hidden_dim}")
        self.alpha_head_hidden_dim = alpha_hidden_dim
        self.alpha_head_layers = max(int(alpha_head_layers), 0)
        self.alpha_head_dropout = float(alpha_head_dropout)
        if not 0.0 <= self.alpha_head_dropout < 1.0:
            raise ValueError(f"alpha_head_dropout must be in [0, 1), got {self.alpha_head_dropout}")
        self.alpha_head_activation = str(alpha_head_activation).strip().lower()
        # N1 (Per-Pair Adaptive Auxiliary Gating): when enabled, alpha_head consumes
        # the joint pair feature [q ; d+ ; q*d+] (3*D) instead of just q (D). This
        # turns alpha into a per-(q,d+) confidence rather than a per-query average.
        # Backwards compatible: the head is callable with q only; pos_doc is optional.
        self.alpha_include_doc = (
            bool(int(os.environ.get("ALPHA_GATE_INCLUDE_DOC", "0")))
            if alpha_include_doc is None
            else bool(alpha_include_doc)
        )
        _alpha_in_dim = int(self.model.output_dims) * (3 if self.alpha_include_doc else 1)
        self.alpha_head = self._build_scalar_head(
            input_dim=_alpha_in_dim,
            hidden_dim=alpha_hidden_dim,
            hidden_layers=self.alpha_head_layers,
            dropout=self.alpha_head_dropout,
            activation=self.alpha_head_activation,
        )
        # Optional bias init on the final linear (plain-sigmoid mode only;
        # alpha_residual already centers on alpha_prior).
        # Prefer ALPHA_GATE_PRIOR_INIT (probability in (0,1)) which is converted
        # to the corresponding logit; ALPHA_GATE_BIAS_LOGIT_INIT sets the raw
        # bias directly.
        if not self.alpha_residual:
            _bias_logit: float | None = None
            _prior_env = os.environ.get("ALPHA_GATE_PRIOR_INIT")
            _logit_env = os.environ.get("ALPHA_GATE_BIAS_LOGIT_INIT")
            if _prior_env is not None:
                _p = float(_prior_env)
                if not 0.0 < _p < 1.0:
                    raise ValueError(f"ALPHA_GATE_PRIOR_INIT must be in (0,1), got {_p}")
                _bias_logit = math.log(_p / (1.0 - _p))
            elif _logit_env is not None:
                _bias_logit = float(_logit_env)
            if _bias_logit is not None:
                with torch.no_grad():
                    self._last_linear(self.alpha_head).bias.fill_(_bias_logit)
        self.alpha_head.to(device)
        self.query_diagonal_head_hidden_dim = alpha_hidden_dim if self.query_diagonal_gate else None
        self.query_diagonal_head = None
        if self.query_diagonal_gate:
            self.query_diagonal_head = torch.nn.Sequential(
                torch.nn.Linear(int(self.model.output_dims), alpha_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(alpha_hidden_dim, int(self.model.output_dims)),
            )
            self.query_diagonal_head.to(device)

        self.gamma1 = torch.nn.Parameter(
            torch.tensor([semantic_gamma], dtype=torch.float32, device=device),
            requires_grad=include_semantic_gamma,
        )
        self.gamma2 = torch.nn.Parameter(
            torch.tensor([keywords_gamma], dtype=torch.float32, device=device),
            requires_grad=include_keywords_gamma,
        )
        # Optional warmup override: when set to a float in (0,1) by the trainer,
        # _alpha_from_query bypasses alpha_head and returns a constant tensor,
        # blocking gradient flow into alpha_head from every call site.
        self.alpha_override: float | None = None

        # N3: query-conditional temperature tau(q) = tau_0 * exp(s * tanh(head(q))).
        # Head consumes the query vector and outputs a scalar log-deviation; the
        # multiplicative deviation is bounded to [exp(-s), exp(+s)].
        # Discarded at eval (loss-only path, dot product unchanged).
        self.tau_query_conditional = bool(int(os.environ.get("TAU_QUERY_CONDITIONAL", "0")))
        self.tau_query_log_scale = float(os.environ.get("TAU_QUERY_LOG_SCALE", "0.5"))
        self.tau_head = None
        self.tau_head_hidden_dim: int | None = None
        if self.tau_query_conditional:
            tau_hidden = max(32, min(256, int(self.model.output_dims) // 2))
            self.tau_head_hidden_dim = tau_hidden
            self.tau_head = torch.nn.Sequential(
                torch.nn.Linear(int(self.model.output_dims), tau_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(tau_hidden, 1),
            )
            with torch.no_grad():
                self.tau_head[-1].weight.zero_()
                self.tau_head[-1].bias.zero_()
            self.tau_head.to(device)

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

    @staticmethod
    def _resolve_query_diagonal_gate_mode(name: str) -> str:
        normalized = str(name).strip().lower()
        allowed = {"raw", "mean-one"}
        if normalized not in allowed:
            raise ValueError(f"Unsupported query_diagonal_gate_mode={name}. Use one of {','.join(sorted(allowed))}")
        return normalized

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

    def query_diagonal_parameters(self) -> list[torch.nn.Parameter]:
        if not self.gamma_joint or not self.query_diagonal_gate or self.query_diagonal_head is None:
            return []
        return list(self.query_diagonal_head.parameters())

    def tau_parameters(self) -> list[torch.nn.Parameter]:
        if self.tau_head is None:
            return []
        return list(self.tau_head.parameters())

    def mixing_parameters(self) -> list[torch.nn.Parameter]:
        if self.gamma_joint:
            return [
                *self.alpha_parameters(),
                *self.query_diagonal_parameters(),
                *self.tau_parameters(),
            ]
        return [*self.gamma_parameters(), *self.tau_parameters()]

    def gamma_weights(self) -> tuple[float, float]:
        semantic_weight = float(self.activation(self.gamma1).detach().item()) if self.include_semantic_gamma else 1.0
        keywords_weight = float(self.activation(self.gamma2).detach().item()) if self.include_keywords_gamma else 1.0
        return semantic_weight, keywords_weight

    def gamma_payload(self) -> dict:
        semantic_weight, keywords_weight = self.gamma_weights()
        return {
            "mode": (
                f"query-alpha{'-residual' if self.alpha_residual else ''}+diag-{self.query_diagonal_gate_mode}"
                if self.gamma_joint and self.query_diagonal_gate
                else (
                    "query-alpha-residual"
                    if self.gamma_joint and self.alpha_residual
                    else ("query-alpha" if self.gamma_joint else "scalar-gamma")
                )
            ),
            "gamma_joint": self.gamma_joint,
            "activation_fn": self.activation_fn,
            "alpha_head": {
                "hidden_dim": self.alpha_head_hidden_dim,
                "layers": self.alpha_head_layers,
                "dropout": self.alpha_head_dropout,
                "activation": self.alpha_head_activation,
                "include_doc": self.alpha_include_doc,
                "input_features": "[q;d+;q*d+]" if self.alpha_include_doc else "[q]",
                "residual": {
                    "enabled": self.alpha_residual,
                    "prior": self.alpha_prior,
                    "scale": self.alpha_residual_scale,
                    "parameterization": "sigmoid(logit(alpha_prior)+scale*tanh(delta(q)))",
                },
            },
            "query_diagonal_gate": {
                "enabled": self.query_diagonal_gate,
                "hidden_dim": self.query_diagonal_head_hidden_dim,
                "scale": self.query_diagonal_gate_scale,
                "mode": self.query_diagonal_gate_mode,
            },
            "tau_query_conditional": {
                "enabled": self.tau_query_conditional,
                "hidden_dim": self.tau_head_hidden_dim,
                "log_scale": self.tau_query_log_scale,
                "parameterization": "tau_base * exp(log_scale * tanh(head(q)))",
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

    def _alpha_from_query(
        self,
        query_vectors: torch.Tensor,
        pos_doc_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # When alpha_override is set, bypass alpha_head entirely so no gradient
        # reaches it from any call site (SimCSE-gate, gamma-mixer, mix_scores).
        if self.alpha_override is not None:
            override = float(self.alpha_override)
            alpha = torch.full(
                (query_vectors.shape[0], 1),
                override,
                dtype=query_vectors.dtype,
                device=query_vectors.device,
            )
            return alpha, {}
        # N1 per-pair input: feed [q ; d+ ; q*d+] when enabled and a positive
        # document is available. Falls back to q-only when pos_doc is None or
        # alpha_include_doc is off (keeps eval / mix_scores call sites working).
        if self.alpha_include_doc and pos_doc_vectors is not None:
            if pos_doc_vectors.shape != query_vectors.shape:
                raise ValueError(
                    f"alpha_include_doc requires matching shapes, got q={tuple(query_vectors.shape)} "
                    f"d+={tuple(pos_doc_vectors.shape)}"
                )
            alpha_input = torch.cat(
                (query_vectors, pos_doc_vectors, query_vectors * pos_doc_vectors),
                dim=-1,
            )
        else:
            alpha_input = query_vectors
        raw_alpha = self.alpha_head(alpha_input)
        details: dict[str, float] = {}
        if self.alpha_residual:
            alpha_delta = self.alpha_residual_scale * torch.tanh(raw_alpha)
            alpha = torch.sigmoid(alpha_delta + self.alpha_prior_logit)
            details = {
                "AlphaPrior": self.alpha_prior,
                "AlphaResidualScale": self.alpha_residual_scale,
                "AlphaResidualDeltaMean": float(alpha_delta.detach().mean().item()),
                "AlphaResidualDeltaStd": float(alpha_delta.detach().std(unbiased=False).item()),
            }
            return alpha, details

        alpha = torch.sigmoid(raw_alpha)
        return alpha, details

    def alpha_weights(
        self,
        query_vectors: torch.Tensor,
        pos_doc_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.gamma_joint:
            raise RuntimeError("alpha_weights() is only available when gamma_joint=True")
        alpha, _ = self._alpha_from_query(query_vectors, pos_doc_vectors)
        return alpha

    def query_diagonal_weights(self, query_vectors: torch.Tensor) -> torch.Tensor:
        if not self.query_diagonal_gate or self.query_diagonal_head is None:
            return torch.ones_like(query_vectors)

        raw_gate = 1.0 + self.query_diagonal_gate_scale * torch.tanh(self.query_diagonal_head(query_vectors))
        if self.query_diagonal_gate_mode == "raw":
            return raw_gate

        # Keep the gate query-adaptive while structurally enforcing an average weight of 1.0 per query.
        norm = raw_gate.mean(dim=1, keepdim=True).clamp_min(1e-6)
        return raw_gate / norm

    def tau_weights(
        self,
        query_vectors: torch.Tensor,
        tau_base: torch.Tensor | float,
    ) -> torch.Tensor:
        """N3: per-query temperature tau(q) = tau_base * exp(s * tanh(head(q))).

        Returns shape [B]. Head is initialised to zero so tau(q) == tau_base at
        step 0 (additive deviation only). Bounded multiplicative range:
        [exp(-s), exp(+s)] around tau_base.
        """
        if self.tau_head is None:
            raise RuntimeError("tau_weights() requires tau_query_conditional=True")
        delta = torch.tanh(self.tau_head(query_vectors)).view(-1) * self.tau_query_log_scale
        if isinstance(tau_base, torch.Tensor):
            base = tau_base.to(delta.device).to(delta.dtype)
        else:
            base = torch.tensor(float(tau_base), device=delta.device, dtype=delta.dtype)
        return base * torch.exp(delta)

    def adaptive_semantic_pair_scores(
        self,
        query_vectors: torch.Tensor,
        doc_vectors: torch.Tensor,
        negative_doc_vectors: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        gate = self.query_diagonal_weights(query_vectors)
        gated_queries = query_vectors * gate
        pair_scores = [torch.sum(gated_queries * doc_vectors, dim=1)]
        if negative_doc_vectors is not None:
            pair_scores.append(torch.sum(gated_queries * negative_doc_vectors, dim=1))

        stacked_scores = torch.stack(pair_scores, dim=1)
        details: dict[str, float | torch.Tensor] = {}
        if self.query_diagonal_gate:
            identity_penalty = torch.mean((gate - 1.0) ** 2)
            scale = max(float(self.query_diagonal_gate_scale), 1e-6)
            saturation_penalty = torch.mean(torch.abs((gate - 1.0) / scale) ** 4)
            details = {
                "DiagGateMean": float(gate.detach().mean().item()),
                "DiagGateStd": float(gate.detach().std(unbiased=False).item()),
                "DiagGateMin": float(gate.detach().min().item()),
                "DiagGateMax": float(gate.detach().max().item()),
                "DiagGateIdentityPenalty": identity_penalty,
                "DiagGateSaturationPenalty": saturation_penalty,
            }
            if self.query_diagonal_gate_mode == "mean-one":
                raw_gate = 1.0 + self.query_diagonal_gate_scale * torch.tanh(self.query_diagonal_head(query_vectors))
                norm = raw_gate.mean(dim=1, keepdim=True)
                details.update(
                    {
                        "DiagGateRawMean": float(raw_gate.detach().mean().item()),
                        "DiagGateRawStd": float(raw_gate.detach().std(unbiased=False).item()),
                        "DiagGateRawMin": float(raw_gate.detach().min().item()),
                        "DiagGateRawMax": float(raw_gate.detach().max().item()),
                        "DiagGateNormMean": float(norm.detach().mean().item()),
                    }
                )
        return (stacked_scores, details) if return_details else stacked_scores

    def mix_scores(
        self,
        semantic_scores: torch.Tensor,
        lexical_scores: torch.Tensor,
        query_vectors: torch.Tensor | None = None,
        pos_doc_vectors: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        details: dict[str, float] = {}
        if self.gamma_joint:
            if query_vectors is None:
                raise ValueError("query_vectors must be provided when gamma_joint=True")
            alpha, alpha_details = self._alpha_from_query(query_vectors, pos_doc_vectors)
            mixed_scores = alpha * semantic_scores + (1.0 - alpha) * lexical_scores
            details = {
                "AlphaMean": float(alpha.detach().mean().item()),
                "AlphaStd": float(alpha.detach().std(unbiased=False).item()),
                "AlphaMin": float(alpha.detach().min().item()),
                "AlphaMax": float(alpha.detach().max().item()),
                **alpha_details,
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
        if self.query_diagonal_gate and self.query_diagonal_head is not None:
            torch.save(self.query_diagonal_head.state_dict(), Path(save_dir) / "query_diagonal_gate.pt")

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
        alpha_cfg = gamma_hybrid.get("alpha_head", {})
        alpha_residual_cfg = alpha_cfg.get("residual", {}) if isinstance(alpha_cfg, dict) else {}
        query_diagonal_cfg = gamma_hybrid.get("query_diagonal_gate", {})
        runner = cls(
            model=model,
            prediction_heads=heads,
            include_semantic_gamma=semantic_cfg.get("enabled", True),
            include_keywords_gamma=keywords_cfg.get("enabled", True),
            gamma_joint=gamma_hybrid.get("gamma_joint", False),
            semantic_gamma=semantic_cfg.get("raw", 0.5),
            keywords_gamma=keywords_cfg.get("raw", 1.5),
            activation_fn=gamma_hybrid.get("activation_fn", "sigmoid"),
            alpha_residual=alpha_residual_cfg.get("enabled", False),
            alpha_prior=alpha_residual_cfg.get("prior", 0.5),
            alpha_residual_scale=alpha_residual_cfg.get("scale", 0.25),
            alpha_head_hidden_dim=alpha_cfg.get("hidden_dim"),
            alpha_head_layers=alpha_cfg.get("layers", 1),
            alpha_head_dropout=alpha_cfg.get("dropout", 0.0),
            alpha_head_activation=alpha_cfg.get("activation", "gelu"),
            alpha_include_doc=alpha_cfg.get("include_doc"),
            query_diagonal_gate=query_diagonal_cfg.get("enabled", False),
            query_diagonal_gate_scale=query_diagonal_cfg.get("scale", 0.25),
            query_diagonal_gate_mode=query_diagonal_cfg.get("mode", "raw"),
        )
        alpha_gate_path = Path(data_dir) / "alpha_gate.pt"
        if runner.gamma_joint and alpha_gate_path.exists():
            runner.alpha_head.load_state_dict(torch.load(alpha_gate_path, map_location="cpu"))
        query_diagonal_gate_path = Path(data_dir) / "query_diagonal_gate.pt"
        if runner.query_diagonal_gate and runner.query_diagonal_head is not None and query_diagonal_gate_path.exists():
            runner.query_diagonal_head.load_state_dict(torch.load(query_diagonal_gate_path, map_location="cpu"))
        return runner
