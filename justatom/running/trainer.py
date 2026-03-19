import math
import string
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pytorch_lightning as L
import torch
from transformers.optimization import Adafactor, AdafactorSchedule

from justatom.logging.io import CSVLogger
from justatom.running.encoders import EncoderRunner, GammaHybridRunner
from justatom.tooling import stl
from justatom.training.loss import ContrastiveLoss, FocalLoss, SoftContrastiveLoss, TripletLoss


class LightningTrainer(L.LightningModule):
    def __init__(
        self,
        runner,
        loss,
        suffix: Iterable[str],
        label_suffix: str | None = "group_ids",
        grad_acc_steps: int = 6,
    ):
        super().__init__()
        self.runner = runner
        self.loss = loss
        self.grad_acc_steps = grad_acc_steps
        self.monitor = []
        self.suffix = set(suffix)
        self.label_suffix = label_suffix

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.runner.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
        )
        lr_scheduler = AdafactorSchedule(optimizer)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        xs = {k: batch[k].to(self.device) for k in batch if not k.endswith(self.label_suffix)}
        ys = {k: batch[k].to(self.device) for k in batch if k.endswith(self.label_suffix)}
        output = self.runner(xs, average=True)

        all_losses = []
        for head in self.runner.prediction_heads:
            if isinstance(head.loss, TripletLoss):
                loss, info = head.loss(*output, ys.get("group_ids"))
                self.log("TrainingLoss", loss, logger=True)
                self.log("DistAcc", info.get("dist_acc"), logger=True)
                self.log("POS distance", info.get("dist_p"), logger=True)
                self.log("NEG distance", info.get("dist_n"), logger=True)
            elif isinstance(head.loss, ContrastiveLoss):
                loss = head.loss(*output)
                self.log("TrainingLoss", loss, logger=True)
            else:
                raise ValueError(f"Unexpected LOSS {self.loss} of UNKNOWN type for ANN tuning")
            all_losses.append(loss)
        per_sample_loss = self.runner.loss_aggregation_fn(all_losses)
        return self.adjust_loss(per_sample_loss)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        xs = {k: batch[k].to(self.device) for k in batch if not k.endswith(self.label_suffix)}
        self.runner(xs, average=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return None

    def adjust_loss(self, loss):
        mean_loss = loss.mean()
        if self.grad_acc_steps > 1:
            mean_loss = mean_loss / self.grad_acc_steps
        return mean_loss


class _BaseGammaLightningTrainer(L.LightningModule):
    save_subdir = "Gamma"

    def __init__(
        self,
        runner: GammaHybridRunner,
        freeze_encoder: bool = True,
        loss_name: str = "contrastive",
        focal_gamma: float = 2.0,
        lr_gamma: float = 1e-2,
        lr_encoder: float = 2e-5,
        weight_decay: float = 0.01,
        alpha_entropy_weight: float = 0.0,
        alpha_train_only: bool = False,
        alpha_mix_weight: float = 0.3,
        lexical_text_by_content: dict[str, str] | None = None,
        save_dir: str | Path | None = None,
        metrics_path: str | Path | None = None,
        stopsyms: str | None = None,
    ):
        super().__init__()
        self.runner = runner
        self.freeze_encoder = freeze_encoder
        self.loss_name = loss_name
        self.focal_gamma = focal_gamma
        self.lr_gamma = lr_gamma
        self.lr_encoder = lr_encoder
        self.weight_decay = weight_decay
        self.alpha_entropy_weight = alpha_entropy_weight
        self.alpha_train_only = alpha_train_only
        self.alpha_mix_weight = alpha_mix_weight
        self.lexical_text_by_content = lexical_text_by_content or {}
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self.metrics_logger = CSVLogger(self.metrics_path) if self.metrics_path is not None else None
        self.stopsyms = "«»:\"'" if stopsyms is None else stopsyms
        self.loss_fn = self._build_loss_fn()
        self.automatic_optimization = False
        self._configure_encoder()

    def _build_loss_fn(self):
        if self.loss_name == "contrastive":
            return torch.nn.functional.cross_entropy
        if self.loss_name == "soft-contrastive":
            return SoftContrastiveLoss(margin=0.5, size_average=True)
        if self.loss_name == "focal-contrastive":
            return FocalLoss(gamma=self.focal_gamma, reduction="mean")
        raise ValueError(f"Unsupported loss={self.loss_name}. Use one of contrastive,soft-contrastive,focal-contrastive")

    @staticmethod
    def _sample_negative_indices(batch_size: int, device: torch.device) -> torch.Tensor:
        if batch_size < 2:
            raise ValueError("loss=soft-contrastive requires batch_size >= 2 to construct negative pairs")

        permutation = torch.randperm(batch_size, device=device)
        identity = torch.arange(batch_size, device=device)
        collision_mask = permutation == identity
        if collision_mask.any():
            permutation = torch.roll(identity, shifts=1)
        return permutation

    def _pairwise_similarity_loss(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        positive_distance = 1.0 - positive_scores
        negative_distance = 1.0 - negative_scores
        positive_loss = 0.5 * positive_distance.pow(2)
        negative_loss = 0.5 * torch.relu(margin - negative_distance).pow(2)
        return (positive_loss + negative_loss).mean()

    def _configure_encoder(self):
        model_parameters = self.runner.model.parameters()
        if self.freeze_encoder:
            self.runner.eval()
            for tensor in model_parameters:
                tensor.requires_grad = False
        else:
            self.runner.train()
            for tensor in model_parameters:
                tensor.requires_grad = True

    def _fn_inverse_idf_recall(
        self,
        query: str,
        doc_text: list[str] | str,
        stopsyms: str | None = None,
    ) -> float:
        stopsyms = stopsyms or self.stopsyms
        stopsyms = string.punctuation if stopsyms is None else stopsyms + string.punctuation

        if isinstance(doc_text, list):
            k_words = Counter(
                stl.flatten_list(["".join([w for w in txt.lower().strip() if w not in stopsyms]).split() for txt in doc_text])
            )
        else:
            k_words = Counter(["".join([ch for ch in w.lower().strip() if ch not in stopsyms]) for w in doc_text.split()])

        q_words = "".join(w for w in query if w not in stopsyms).lower().strip().split()
        numerator = sum([1.0 / math.log(1 + k_words.get(w, 1)) for w in q_words if w in k_words])
        denominator = sum([1.0 / math.log(1 + k_words.get(w, 1)) for w in q_words])
        return numerator / denominator if denominator > 0 else 0.0

    def _build_rank_matrix(self, batch: dict[str, torch.Tensor], shape: tuple[int, int]) -> torch.Tensor:
        rank_matrix = torch.zeros(shape, device=self.device, requires_grad=False)
        with torch.no_grad():
            for i, q_tokens in enumerate(batch["input_ids"]):
                for j, d_tokens in enumerate(batch["pos_input_ids"]):
                    queries = self.runner.processor.tokenizer.decode(
                        q_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )[len(self.runner.processor.queries_prefix) :].strip()
                    content = self.runner.processor.tokenizer.decode(
                        d_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )[len(self.runner.processor.pos_queries_prefix) :].strip()
                    lexical_text = self.lexical_text_by_content.get(content, content)
                    rank_matrix[i, j] = self._fn_inverse_idf_recall(queries, lexical_text)
        return rank_matrix

    @staticmethod
    def _grad_norm(parameters) -> float:
        grads = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
        if not grads:
            return 0.0
        return torch.stack(grads).norm(2).item()

    def _gamma_metrics(self) -> dict[str, float]:
        if self.runner.gamma_joint:
            return {}
        semantic_weight, keywords_weight = self.runner.gamma_weights()
        return {"Gamma1": semantic_weight, "Gamma2": keywords_weight}

    def _gamma_grad_metrics(self) -> dict[str, float]:
        if self.runner.gamma_joint:
            return {"Grad_norm_alpha_head": self._grad_norm(self.runner.alpha_parameters())}
        return {
            "Grad_norm_gamma1": (self._grad_norm([self.runner.gamma1]) if self.runner.include_semantic_gamma else 0.0),
            "Grad_norm_gamma2": (self._grad_norm([self.runner.gamma2]) if self.runner.include_keywords_gamma else 0.0),
        }

    def _batch_retrieval_metrics(self, scores: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            batch_size = int(scores.shape[0])
            sorted_indices = torch.argsort(scores, dim=1, descending=True)
            target_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1)
            rank_positions = (sorted_indices == target_indices).nonzero(as_tuple=False)[:, 1] + 1

            rank_positions_f = rank_positions.float()
            reciprocal_ranks = 1.0 / rank_positions_f

            hit_at_1 = (rank_positions <= 1).float().mean().item()
            hit_at_3 = (rank_positions <= min(3, batch_size)).float().mean().item()
            hit_at_5 = (rank_positions <= min(5, batch_size)).float().mean().item()
            errors_at_1 = int((rank_positions > 1).sum().item())

            return {
                "BatchSize": float(batch_size),
                "ErrorsAt1": float(errors_at_1),
                "ErrorRateAt1": float(1.0 - hit_at_1),
                "HitRateAt1": float(hit_at_1),
                "HitRateAt3": float(hit_at_3),
                "HitRateAt5": float(hit_at_5),
                "MRR": float(reciprocal_ranks.mean().item()),
                "MeanRank": float(rank_positions_f.mean().item()),
                "MedianRank": float(rank_positions_f.median().item()),
            }

    def configure_optimizers(self):
        mixing_params = [p for p in self.runner.mixing_parameters() if p.requires_grad]
        param_groups = []
        if mixing_params:
            param_groups.append({"params": mixing_params, "lr": self.lr_gamma, "weight_decay": 0.0})

        if not self.freeze_encoder:
            mixing_param_ids = {id(p) for p in mixing_params}
            encoder_params = [p for p in self.runner.parameters() if p.requires_grad and id(p) not in mixing_param_ids]
            if encoder_params:
                param_groups.append(
                    {
                        "params": encoder_params,
                        "lr": self.lr_encoder,
                        "weight_decay": self.weight_decay,
                    }
                )

        return torch.optim.AdamW(param_groups)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.freeze_encoder:
            with torch.no_grad():
                q_vecs, d_vecs = self.runner(batch, norm=True)
        else:
            q_vecs, d_vecs = self.runner(batch, norm=True)
        scores = q_vecs @ d_vecs.T
        rank_matrix = self._build_rank_matrix(batch, scores.shape)
        mixed_output = self.runner.mix_scores(
            scores,
            rank_matrix,
            query_vectors=q_vecs,
            return_details=True,
        )
        assert isinstance(mixed_output, tuple)
        mixed_scores, mix_metrics = mixed_output
        return mixed_scores, mix_metrics, scores, rank_matrix, q_vecs, d_vecs

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        output, mix_metrics, semantic_scores, rank_matrix, q_vecs, d_vecs = self.forward(batch)
        labels = torch.arange(len(output), device=output.device)

        if self.loss_name == "soft-contrastive":
            negative_indices = self._sample_negative_indices(q_vecs.shape[0], q_vecs.device)
            negative_d_vecs = d_vecs[negative_indices]
            positive_labels = torch.ones(q_vecs.shape[0], device=q_vecs.device)
            negative_labels = torch.zeros(q_vecs.shape[0], device=q_vecs.device)

            semantic_loss = self.loss_fn(q_vecs, d_vecs, positive_labels) + self.loss_fn(q_vecs, negative_d_vecs, negative_labels)

            positive_scores = torch.diagonal(output)
            negative_scores = output[torch.arange(output.shape[0], device=output.device), negative_indices]
            mix_loss = self._pairwise_similarity_loss(positive_scores, negative_scores)
        else:
            mix_loss = self.loss_fn(output, labels)
            semantic_loss = self.loss_fn(semantic_scores, labels)

        main_loss = semantic_loss if self.alpha_train_only and self.runner.gamma_joint else mix_loss
        aux_metrics: dict[str, float] = {}
        entropy_bonus = None
        if self.runner.gamma_joint:
            alpha = self.runner.alpha_weights(q_vecs)
            alpha_clamped = alpha.clamp(min=1e-6, max=1.0 - 1e-6)
            alpha_entropy = -(
                alpha_clamped * torch.log(alpha_clamped) + (1.0 - alpha_clamped) * torch.log(1.0 - alpha_clamped)
            ).mean()
            entropy_bonus = self.alpha_entropy_weight * alpha_entropy
            if self.alpha_train_only:
                loss = semantic_loss + self.alpha_mix_weight * mix_loss - entropy_bonus
            else:
                loss = mix_loss - entropy_bonus
            with torch.no_grad():
                aux_metrics["alpha_aux_loss"] = float(semantic_loss.detach().item())
                aux_metrics["alpha_mix_loss"] = float(mix_loss.detach().item())
                aux_metrics["alpha_entropy"] = float(alpha_entropy.detach().item())
                aux_metrics["alpha_entropy_bonus"] = float(entropy_bonus.detach().item())
                aux_metrics["alpha_entropy_weight"] = float(self.alpha_entropy_weight)
                aux_metrics["alpha_train_only"] = float(int(self.alpha_train_only))
                aux_metrics["alpha_mix_weight"] = float(self.alpha_mix_weight)
        else:
            loss = main_loss

        optimizer.zero_grad()
        self.manual_backward(loss)

        grad_metrics = self._gamma_grad_metrics()
        grad_metrics["Grad_Norm_model"] = self._grad_norm(self.runner.model.parameters())
        batch_metrics = self._batch_retrieval_metrics(output.detach())
        optimizer.step()

        payload = {
            "Loss": float(loss.detach().item()),
            "MainLoss": float(main_loss.detach().item()),
            "FreezeEncoder": int(self.freeze_encoder),
            "GammaJoint": int(self.runner.gamma_joint),
            "ActivationFn": self.runner.activation_fn,
            **self._gamma_metrics(),
            **aux_metrics,
            **mix_metrics,
            **batch_metrics,
            **grad_metrics,
        }
        for key, value in payload.items():
            if isinstance(value, str):
                continue
            self.log(key, value, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if self.metrics_logger is not None:
            self.metrics_logger.log_metrics(payload)

        return loss.detach()

    def on_train_epoch_end(self):
        if self.save_dir is None:
            return
        save_path = self.save_dir / self.save_subdir / f"epoch{str(self.current_epoch + 1)}"
        self.runner.save(str(save_path))

    def on_train_end(self):
        if self.metrics_logger is not None:
            self.metrics_logger.close_log()


class UniGammaLightningTrainer(_BaseGammaLightningTrainer):
    save_subdir = "UniGamma"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        enabled_count = int(self.runner.include_semantic_gamma) + int(self.runner.include_keywords_gamma)
        if enabled_count != 1:
            raise ValueError("UniGammaLightningTrainer requires exactly one enabled gamma.")


class BiGammaLightningTrainer(_BaseGammaLightningTrainer):
    save_subdir = "BiGamma"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.runner.include_semantic_gamma or not self.runner.include_keywords_gamma:
            raise ValueError("BiGammaLightningTrainer requires both semantic and keywords gamma to be enabled.")


class EncoderOnlyLightningTrainer(L.LightningModule):
    save_subdir = "Encoder"

    def __init__(
        self,
        runner: EncoderRunner,
        loss_name: str = "contrastive",
        focal_gamma: float = 2.0,
        lr_encoder: float = 2e-5,
        weight_decay: float = 0.01,
        save_dir: str | Path | None = None,
        metrics_path: str | Path | None = None,
    ):
        super().__init__()
        self.runner = runner
        self.loss_name = loss_name
        self.focal_gamma = focal_gamma
        self.lr_encoder = lr_encoder
        self.weight_decay = weight_decay
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self.metrics_logger = CSVLogger(self.metrics_path) if self.metrics_path is not None else None
        self.loss_fn = self._build_loss_fn()
        self.automatic_optimization = False
        self.runner.train()
        for tensor in self.runner.model.parameters():
            tensor.requires_grad = True

    def _build_loss_fn(self):
        if self.loss_name == "contrastive":
            return torch.nn.functional.cross_entropy
        if self.loss_name == "soft-contrastive":
            return SoftContrastiveLoss(margin=0.5, size_average=True)
        if self.loss_name == "focal-contrastive":
            return FocalLoss(gamma=self.focal_gamma, reduction="mean")
        raise ValueError(f"Unsupported loss={self.loss_name}. Use one of contrastive,soft-contrastive,focal-contrastive")

    @staticmethod
    def _sample_negative_indices(batch_size: int, device: torch.device) -> torch.Tensor:
        if batch_size < 2:
            raise ValueError("loss=soft-contrastive requires batch_size >= 2 to construct negative pairs")

        permutation = torch.randperm(batch_size, device=device)
        identity = torch.arange(batch_size, device=device)
        collision_mask = permutation == identity
        if collision_mask.any():
            permutation = torch.roll(identity, shifts=1)
        return permutation

    @staticmethod
    def _grad_norm(parameters) -> float:
        grads = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
        if not grads:
            return 0.0
        return torch.stack(grads).norm(2).item()

    @staticmethod
    def _batch_retrieval_metrics(scores: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            batch_size = int(scores.shape[0])
            sorted_indices = torch.argsort(scores, dim=1, descending=True)
            target_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1)
            rank_positions = (sorted_indices == target_indices).nonzero(as_tuple=False)[:, 1] + 1

            rank_positions_f = rank_positions.float()
            reciprocal_ranks = 1.0 / rank_positions_f

            hit_at_1 = (rank_positions <= 1).float().mean().item()
            hit_at_3 = (rank_positions <= min(3, batch_size)).float().mean().item()
            hit_at_5 = (rank_positions <= min(5, batch_size)).float().mean().item()
            errors_at_1 = int((rank_positions > 1).sum().item())

            return {
                "BatchSize": float(batch_size),
                "ErrorsAt1": float(errors_at_1),
                "ErrorRateAt1": float(1.0 - hit_at_1),
                "HitRateAt1": float(hit_at_1),
                "HitRateAt3": float(hit_at_3),
                "HitRateAt5": float(hit_at_5),
                "MRR": float(reciprocal_ranks.mean().item()),
                "MeanRank": float(rank_positions_f.mean().item()),
                "MedianRank": float(rank_positions_f.median().item()),
            }

    def configure_optimizers(self):
        encoder_params = [p for p in self.runner.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            [
                {
                    "params": encoder_params,
                    "lr": self.lr_encoder,
                    "weight_decay": self.weight_decay,
                }
            ]
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q_vecs, d_vecs = self.runner(batch, norm=True)
        return q_vecs @ d_vecs.T

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        q_vecs, d_vecs = self.runner(batch, norm=True)
        output = q_vecs @ d_vecs.T

        if self.loss_name == "soft-contrastive":
            negative_indices = self._sample_negative_indices(q_vecs.shape[0], q_vecs.device)
            negative_vecs = d_vecs[negative_indices]
            positive_labels = torch.ones(q_vecs.shape[0], device=q_vecs.device)
            negative_labels = torch.zeros(q_vecs.shape[0], device=q_vecs.device)
            positive_loss = self.loss_fn(q_vecs, d_vecs, positive_labels)
            negative_loss = self.loss_fn(q_vecs, negative_vecs, negative_labels)
            loss = positive_loss + negative_loss
        else:
            labels = torch.arange(len(output), device=output.device)
            loss = self.loss_fn(output, labels)

        optimizer.zero_grad()
        self.manual_backward(loss)

        grad_norm_model = self._grad_norm(self.runner.model.parameters())
        batch_metrics = self._batch_retrieval_metrics(output.detach())
        optimizer.step()

        payload = {
            "Loss": float(loss.detach().item()),
            "FreezeEncoder": 0,
            "TrainingMode": "encoder-only",
            "Grad_Norm_model": grad_norm_model,
            **batch_metrics,
        }
        for key, value in payload.items():
            if isinstance(value, str):
                continue
            self.log(key, value, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if self.metrics_logger is not None:
            self.metrics_logger.log_metrics(payload)

        return loss.detach()

    def on_train_epoch_end(self):
        if self.save_dir is None:
            return
        save_path = self.save_dir / self.save_subdir / f"epoch{str(self.current_epoch + 1)}"
        self.runner.save(str(save_path))

    def on_train_end(self):
        if self.metrics_logger is not None:
            self.metrics_logger.close_log()
