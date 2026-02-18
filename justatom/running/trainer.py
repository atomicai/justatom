import math
import string
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pytorch_lightning as L
import torch
from transformers.optimization import Adafactor, AdafactorSchedule

from justatom.logging.io import CSVLogger
from justatom.running.encoders import GammaHybridRunner
from justatom.tooling import stl
from justatom.training.loss import ContrastiveLoss, FocalLoss, TripletLoss


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
        if self.loss_name == "focal-contrastive":
            return FocalLoss(gamma=self.focal_gamma, reduction="mean")
        raise ValueError(f"Unsupported loss={self.loss_name}. Use one of contrastive,focal-contrastive")

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
                stl.flatten_list(
                    [
                        "".join([w for w in txt.lower().strip() if w not in stopsyms]).split()
                        for txt in doc_text
                    ]
                )
            )
        else:
            k_words = Counter(
                [
                    "".join([ch for ch in w.lower().strip() if ch not in stopsyms])
                    for w in doc_text.split()
                ]
            )

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
        semantic_weight, keywords_weight = self.runner.gamma_weights()
        return {"Gamma1": semantic_weight, "Gamma2": keywords_weight}

    def _gamma_grad_metrics(self) -> dict[str, float]:
        return {
            "Grad_norm_gamma1": self._grad_norm([self.runner.gamma1]) if self.runner.include_semantic_gamma else 0.0,
            "Grad_norm_gamma2": self._grad_norm([self.runner.gamma2]) if self.runner.include_keywords_gamma else 0.0,
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
        gamma_params = [p for p in self.runner.gamma_parameters() if p.requires_grad]
        param_groups = [{"params": gamma_params, "lr": self.lr_gamma, "weight_decay": 0.0}]

        if not self.freeze_encoder:
            gamma_param_ids = {id(p) for p in gamma_params}
            encoder_params = [
                p for p in self.runner.parameters() if p.requires_grad and id(p) not in gamma_param_ids
            ]
            if encoder_params:
                param_groups.append(
                    {
                        "params": encoder_params,
                        "lr": self.lr_encoder,
                        "weight_decay": self.weight_decay,
                    }
                )

        return torch.optim.AdamW(param_groups)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                q_vecs, d_vecs = self.runner(batch, norm=True)
        else:
            q_vecs, d_vecs = self.runner(batch, norm=True)
        scores = q_vecs @ d_vecs.T
        rank_matrix = self._build_rank_matrix(batch, scores.shape)
        return self.runner.mix_scores(scores, rank_matrix)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        output = self.forward(batch)
        labels = torch.arange(len(output), device=output.device)
        loss = self.loss_fn(output, labels)

        optimizer.zero_grad()
        self.manual_backward(loss)

        grad_metrics = self._gamma_grad_metrics()
        grad_metrics["Grad_Norm_model"] = self._grad_norm(self.runner.model.parameters())
        batch_metrics = self._batch_retrieval_metrics(output.detach())
        optimizer.step()

        payload = {
            "Loss": float(loss.detach().item()),
            "FreezeEncoder": int(self.freeze_encoder),
            "ActivationFn": self.runner.activation_fn,
            **self._gamma_metrics(),
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
