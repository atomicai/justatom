import math
import string
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import pytorch_lightning as L
import torch
from transformers.optimization import Adafactor, AdafactorSchedule

from justatom.logging.io import CSVLogger
from justatom.running.encoders import EncoderRunner, GammaHybridRunner
from justatom.tooling import stl
from justatom.training.diagnostics import embedding_geometry_metrics
from justatom.training.loss import ContrastiveLoss, FocalLoss, SoftContrastiveLoss, TripletLoss


def _normalize_optimizer_name(name: str | None, *, default: str) -> str:
    normalized = (name or default).strip().lower()
    aliases = {
        "auto": "auto",
        "adamw": "adamw",
        "adam": "adamw",
        "adagrad": "adagrad",
        "adafactor": "adafactor",
    }
    if normalized not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported optimizer={name!r}. Expected one of: {allowed}")
    return aliases[normalized]


def _sample_negative_derangement(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size < 2:
        raise ValueError("loss=soft-contrastive requires batch_size >= 2 to construct negative pairs")

    # Sattolo's algorithm samples a single cycle, which guarantees no fixed points.
    permutation = torch.arange(batch_size, device=device)
    for idx in range(batch_size - 1, 0, -1):
        swap_idx = int(torch.randint(0, idx, (1,), device=device).item())
        tmp = permutation[idx].clone()
        permutation[idx] = permutation[swap_idx]
        permutation[swap_idx] = tmp
    return permutation


def _sample_safe_negative_indices(
    *,
    doc_key_ids: torch.Tensor,
    content_key_ids: torch.Tensor | None = None,
    query_key_ids: torch.Tensor | None = None,
    queries: list[str] | None = None,
    docs: list[str] | None = None,
    lexical_text_by_content: dict[str, str] | None = None,
    inverse_idf_recall_fn: Callable[[str, list[str] | str], float] | None = None,
    max_negative_inverse_idf_recall: float | None = None,
    min_negative_inverse_idf_recall: float | None = None,
    negative_sampling_mode: str = "safe-random",
    hard_negative_top_k: int | None = None,
) -> tuple[torch.Tensor, int]:
    batch_size = int(doc_key_ids.shape[0])
    device = doc_key_ids.device
    if batch_size < 2:
        raise ValueError("loss=soft-contrastive requires batch_size >= 2 to construct negative pairs")

    negative_indices = torch.empty(batch_size, dtype=torch.long, device=device)
    fallback_count = 0
    all_indices = torch.arange(batch_size, device=device)
    allowed_sampling_modes = {"safe-random", "semi-hard-idf", "hard-idf"}
    if negative_sampling_mode not in allowed_sampling_modes:
        raise ValueError(
            f"Unsupported negative_sampling_mode={negative_sampling_mode!r}. "
            f"Expected one of: {', '.join(sorted(allowed_sampling_modes))}"
        )
    if hard_negative_top_k is not None and hard_negative_top_k < 1:
        raise ValueError(f"hard_negative_top_k must be >= 1 when provided, got {hard_negative_top_k}")

    for idx in range(batch_size):
        valid_mask = all_indices != idx
        valid_mask &= doc_key_ids != doc_key_ids[idx]
        if content_key_ids is not None:
            valid_mask &= content_key_ids != content_key_ids[idx]
        if query_key_ids is not None:
            valid_mask &= query_key_ids != query_key_ids[idx]

        candidate_indices = all_indices[valid_mask]
        if candidate_indices.numel() == 0:
            fallback_mask = all_indices != idx
            fallback_mask &= doc_key_ids != doc_key_ids[idx]
            if content_key_ids is not None:
                fallback_mask &= content_key_ids != content_key_ids[idx]
            candidate_indices = all_indices[fallback_mask]

        if candidate_indices.numel() == 0:
            fallback_count += 1
            candidate_indices = all_indices[all_indices != idx]

        if (
            (
                max_negative_inverse_idf_recall is not None
                or min_negative_inverse_idf_recall is not None
                or negative_sampling_mode != "safe-random"
            )
            and queries is not None
            and docs is not None
            and inverse_idf_recall_fn is not None
        ):
            query = queries[idx]
            scored_candidates: list[tuple[int, float]] = []
            for candidate_idx in candidate_indices.detach().cpu().tolist():
                candidate_doc = docs[candidate_idx]
                candidate_sparse_text = (
                    lexical_text_by_content.get(candidate_doc, candidate_doc) if lexical_text_by_content else candidate_doc
                )
                candidate_score = inverse_idf_recall_fn(query, candidate_sparse_text)
                scored_candidates.append((candidate_idx, candidate_score))

            safe_candidates = [
                (candidate_idx, candidate_score)
                for candidate_idx, candidate_score in scored_candidates
                if max_negative_inverse_idf_recall is None or candidate_score <= max_negative_inverse_idf_recall
            ]
            preferred_candidates = [
                (candidate_idx, candidate_score)
                for candidate_idx, candidate_score in safe_candidates
                if min_negative_inverse_idf_recall is None or candidate_score >= min_negative_inverse_idf_recall
            ]

            if negative_sampling_mode == "safe-random":
                chosen_candidates = safe_candidates
            else:
                ranked_candidates = preferred_candidates if preferred_candidates else safe_candidates
                ranked_candidates = sorted(ranked_candidates, key=lambda item: item[1], reverse=True)
                if hard_negative_top_k is not None:
                    ranked_candidates = ranked_candidates[:hard_negative_top_k]
                chosen_candidates = ranked_candidates

            if chosen_candidates:
                candidate_indices = torch.tensor(
                    [candidate_idx for candidate_idx, _ in chosen_candidates],
                    dtype=torch.long,
                    device=device,
                )

        sampled_offset = int(torch.randint(0, int(candidate_indices.numel()), (1,), device=device).item())
        negative_indices[idx] = candidate_indices[sampled_offset]

    return negative_indices, fallback_count


def _inverse_idf_recall(
    query: str,
    doc_text: list[str] | str,
    *,
    stopsyms: str | None = None,
) -> float:
    stopsyms = stopsyms or "«»:\"'"
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


def _pairwise_focal_loss(loss_fn: FocalLoss, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    if positive_scores.shape != negative_scores.shape:
        raise ValueError(
            f"positive_scores and negative_scores must have the same shape. Got {positive_scores.shape} vs {negative_scores.shape}"
        )
    pair_logits = torch.stack([positive_scores, negative_scores], dim=1)
    targets = torch.zeros(pair_logits.shape[0], dtype=torch.long, device=pair_logits.device)
    return loss_fn(pair_logits, targets)


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
        loss_name: str = "soft-contrastive",
        focal_gamma: float = 2.0,
        lr_gamma: float = 1e-2,
        lr_encoder: float = 2e-5,
        weight_decay: float = 0.01,
        alpha_entropy_weight: float = 0.0,
        alpha_train_only: bool = False,
        alpha_mix_weight: float = 0.3,
        alpha_mix_weight_warmup_steps: int = 0,
        negative_sampling_mode: str = "safe-random",
        min_negative_inverse_idf_recall: float | None = None,
        hard_negative_top_k: int | None = None,
        query_diagonal_identity_weight: float = 0.0,
        query_diagonal_saturation_weight: float = 0.0,
        margin: float = 0.5,
        contrastive_temperature: float = 0.1,
        soft_contrastive_temperature: float = 1.0,
        grad_acc_steps: int = 6,
        optimizer_name: str = "auto",
        max_negative_inverse_idf_recall: float | None = None,
        contrastive_learnable_temperature: bool = True,
        contrastive_decoupled: bool = True,
        contrastive_simcse_dropout_weight: float = 0.0,
        contrastive_soft_fn_attract_weight: float = 0.0,
        contrastive_soft_fn_topk: int = 1,
        contrastive_loss_alpha_gate: bool = False,
        contrastive_loss_alpha_gate_mode: str = "augment",
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
        self.alpha_mix_weight_warmup_steps = max(int(alpha_mix_weight_warmup_steps), 0)
        self.negative_sampling_mode = negative_sampling_mode
        self.min_negative_inverse_idf_recall = min_negative_inverse_idf_recall
        self.hard_negative_top_k = hard_negative_top_k
        self.query_diagonal_identity_weight = query_diagonal_identity_weight
        self.query_diagonal_saturation_weight = query_diagonal_saturation_weight
        self.margin = margin
        self.contrastive_temperature = contrastive_temperature
        self.soft_contrastive_temperature = soft_contrastive_temperature
        self.grad_acc_steps = grad_acc_steps
        self.optimizer_name = _normalize_optimizer_name(optimizer_name, default="adamw")
        self.max_negative_inverse_idf_recall = max_negative_inverse_idf_recall
        self.contrastive_learnable_temperature = bool(contrastive_learnable_temperature)
        self.contrastive_decoupled = bool(contrastive_decoupled)
        self.contrastive_simcse_dropout_weight = float(contrastive_simcse_dropout_weight)
        self.contrastive_soft_fn_attract_weight = float(contrastive_soft_fn_attract_weight)
        self.contrastive_soft_fn_topk = max(int(contrastive_soft_fn_topk), 1)
        self.contrastive_loss_alpha_gate = bool(contrastive_loss_alpha_gate)
        mode = str(contrastive_loss_alpha_gate_mode).lower()
        if mode not in {"augment", "convex"}:
            raise ValueError("contrastive_loss_alpha_gate_mode must be one of: augment, convex")
        self.contrastive_loss_alpha_gate_mode = mode
        if self.contrastive_simcse_dropout_weight < 0.0:
            raise ValueError("contrastive_simcse_dropout_weight must be >= 0")
        if self.contrastive_soft_fn_attract_weight < 0.0:
            raise ValueError("contrastive_soft_fn_attract_weight must be >= 0")
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
            return ContrastiveLoss(
                temperature=self.contrastive_temperature,
                reduction="mean",
                learnable_temperature=self.contrastive_learnable_temperature,
                decoupled=self.contrastive_decoupled,
            )
        if self.loss_name == "soft-contrastive":
            return SoftContrastiveLoss(
                margin=self.margin,
                size_average=True,
                temperature=self.soft_contrastive_temperature,
            )
        if self.loss_name == "focal-contrastive":
            return FocalLoss(gamma=self.focal_gamma, reduction="mean")
        raise ValueError(
            f"Unsupported loss={self.loss_name}. Gamma/alpha training only supports loss in {{soft-contrastive, contrastive, focal-contrastive}}"
        )

    def _scale_soft_contrastive_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if self.soft_contrastive_temperature == 1.0:
            return scores
        return (scores / self.soft_contrastive_temperature).clamp(min=-1.0, max=1.0)

    @staticmethod
    def _model_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "pos_input_ids", "pos_attention_mask", "group_ids"}
        }

    def _sample_negative_indices(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        queries, docs = self._decode_content_from_batches(batch)
        return _sample_safe_negative_indices(
            doc_key_ids=batch["doc_key_id"],
            content_key_ids=batch.get("content_key_id"),
            query_key_ids=batch.get("query_key_id"),
            queries=queries,
            docs=docs,
            lexical_text_by_content=self.lexical_text_by_content,
            inverse_idf_recall_fn=self._fn_inverse_idf_recall,
            max_negative_inverse_idf_recall=self.max_negative_inverse_idf_recall,
            min_negative_inverse_idf_recall=self.min_negative_inverse_idf_recall,
            negative_sampling_mode=self.negative_sampling_mode,
            hard_negative_top_k=self.hard_negative_top_k,
        )

    def _pairwise_similarity_loss(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        margin: float | None = None,
    ) -> torch.Tensor:
        margin = self.margin if margin is None else margin
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
        return _inverse_idf_recall(query, doc_text, stopsyms=stopsyms or self.stopsyms)

    def _decode_content_from_batches(self, batch) -> tuple[list[str], list[str]]:
        tokenizer = self.runner.processor.tokenizer
        query_prefix = self.runner.processor.queries_prefix
        doc_prefix = self.runner.processor.pos_queries_prefix

        queries = [
            tokenizer.decode(
                q_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            .removeprefix(query_prefix)
            .strip()
            for q_tokens in batch["input_ids"]
        ]
        docs = [
            tokenizer.decode(
                d_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            .removeprefix(doc_prefix)
            .strip()
            for d_tokens in batch["pos_input_ids"]
        ]
        return queries, docs

    def _build_sparse_pair_scores(
        self,
        batch: dict[str, torch.Tensor],
        negative_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries, docs = self._decode_content_from_batches(batch)
        negative_idx_list = negative_indices.detach().cpu().tolist()

        positive_scores: list[float] = []
        negative_scores: list[float] = []
        for idx, query in enumerate(queries):
            positive_doc = docs[idx]
            negative_doc = docs[negative_idx_list[idx]]
            positive_sparse_text = self.lexical_text_by_content.get(positive_doc, positive_doc)
            negative_sparse_text = self.lexical_text_by_content.get(negative_doc, negative_doc)
            positive_scores.append(self._fn_inverse_idf_recall(query, positive_sparse_text))
            negative_scores.append(self._fn_inverse_idf_recall(query, negative_sparse_text))

        return (
            torch.tensor(positive_scores, device=self.device, dtype=torch.float32),
            torch.tensor(negative_scores, device=self.device, dtype=torch.float32),
        )

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
            metrics = {"Grad_norm_alpha_head": self._grad_norm(self.runner.alpha_parameters())}
            if self.runner.query_diagonal_gate:
                metrics["Grad_norm_query_diagonal_head"] = self._grad_norm(self.runner.query_diagonal_parameters())
            return metrics
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
        if self.optimizer_name == "adafactor":
            optimizer = Adafactor(
                param_groups,
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                lr=None,
            )
            return [optimizer], [AdafactorSchedule(optimizer)]
        if self.optimizer_name == "adagrad":
            return torch.optim.Adagrad(param_groups)

        return torch.optim.AdamW(param_groups)

    def _adjust_loss(self, loss: torch.Tensor) -> torch.Tensor:
        mean_loss = loss.mean()
        if self.grad_acc_steps > 1:
            mean_loss = mean_loss / self.grad_acc_steps
        return mean_loss

    def _effective_alpha_mix_weight(self) -> float:
        if self.alpha_mix_weight_warmup_steps <= 0:
            return float(self.alpha_mix_weight)

        warmup_progress = min(max(float(self.global_step), 0.0) / float(self.alpha_mix_weight_warmup_steps), 1.0)
        return float(self.alpha_mix_weight) * warmup_progress

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_batch = self._model_batch(batch)
        if self.freeze_encoder:
            with torch.no_grad():
                q_vecs, d_vecs = self.runner(model_batch, norm=True)
        else:
            q_vecs, d_vecs = self.runner(model_batch, norm=True)
        scores = q_vecs @ d_vecs.T
        return q_vecs, d_vecs, scores

    def _compute_contrastive_loss(
        self,
        *,
        batch: dict[str, torch.Tensor],
        q_vecs: torch.Tensor,
        d_vecs: torch.Tensor,
        metrics: dict[str, float],
    ) -> torch.Tensor:
        """Build the InfoNCE-family loss with optional SimCSE / soft-FN / alpha-gate.

        Per-row terms are aggregated as
        ``loss = main + simcse_w * simcse + soft_fn_w * soft_fn`` by default,
        or as ``(alpha(q) * main + (1 - alpha(q)) * aux).mean()`` when
        ``contrastive_loss_alpha_gate`` is enabled and ``runner.gamma_joint``
        provides a per-query alpha head, with ``aux`` being the *per-row*
        weighted sum of the SimCSE and soft-FN components.
        """
        loss_fn: ContrastiveLoss = self.loss_fn
        main_per_row = loss_fn.info_nce(q_vecs, d_vecs, reduction="none")

        simcse_per_row = None
        if self.contrastive_simcse_dropout_weight > 0.0 and not self.freeze_encoder:
            # Second forward of the *queries* with a fresh dropout mask.
            model_batch = self._model_batch(batch)
            q_alt_vecs, _ = self.runner(model_batch, norm=True)
            simcse_per_row = loss_fn.simcse_term(q_vecs, q_alt_vecs, reduction="none")

        soft_fn_per_row = None
        if self.contrastive_soft_fn_attract_weight > 0.0 and q_vecs.shape[0] > 1:
            soft_fn_per_row = loss_fn.soft_fn_term(q_vecs, d_vecs, topk=self.contrastive_soft_fn_topk, reduction="none")

        aux_per_row = None
        if simcse_per_row is not None or soft_fn_per_row is not None:
            aux_per_row = main_per_row.new_zeros(main_per_row.shape)
            if simcse_per_row is not None:
                aux_per_row = aux_per_row + self.contrastive_simcse_dropout_weight * simcse_per_row
            if soft_fn_per_row is not None:
                aux_per_row = aux_per_row + self.contrastive_soft_fn_attract_weight * soft_fn_per_row

        if self.contrastive_loss_alpha_gate and self.runner.gamma_joint and aux_per_row is not None:
            alpha_q = self.runner.alpha_weights(q_vecs).clamp(min=1e-6, max=1.0 - 1e-6)
            if alpha_q.dim() > 1:
                alpha_q = alpha_q.view(-1)
            if self.contrastive_loss_alpha_gate_mode == "convex":
                # alpha * main + (1 - alpha) * aux  (legacy; both terms per-row).
                per_row = alpha_q * main_per_row + (1.0 - alpha_q) * aux_per_row
            else:
                # "augment" (default): main + (1 - alpha) * aux.
                # alpha(q) ~ 1 -> trust main fully (semantic-confident query);
                # alpha(q) ~ 0 -> apply full SimCSE/soft-FN regularization.
                per_row = main_per_row + (1.0 - alpha_q) * aux_per_row
            loss = per_row.mean()
            metrics["ContrastiveLossAlphaGate"] = 1.0
            metrics["ContrastiveLossAlphaGateAlphaMean"] = float(alpha_q.detach().mean().item())
            metrics["ContrastiveLossAlphaGateMode"] = 1.0 if self.contrastive_loss_alpha_gate_mode == "augment" else 0.0
        else:
            loss = main_per_row.mean()
            if aux_per_row is not None:
                loss = loss + aux_per_row.mean()
            metrics["ContrastiveLossAlphaGate"] = float(int(self.contrastive_loss_alpha_gate))

        with torch.no_grad():
            metrics["ContrastiveMainLoss"] = float(main_per_row.mean().item())
            if simcse_per_row is not None:
                metrics["ContrastiveSimCSELoss"] = float(simcse_per_row.mean().item())
            if soft_fn_per_row is not None:
                metrics["ContrastiveSoftFNLoss"] = float(soft_fn_per_row.mean().item())
            metrics["ContrastiveSimCSEWeight"] = float(self.contrastive_simcse_dropout_weight)
            metrics["ContrastiveSoftFNWeight"] = float(self.contrastive_soft_fn_attract_weight)
            metrics["ContrastiveSoftFNTopK"] = float(self.contrastive_soft_fn_topk)
            metrics["ContrastiveLearnableTau"] = float(int(self.contrastive_learnable_temperature))
            metrics["ContrastiveDecoupled"] = float(int(self.contrastive_decoupled))
        return loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        q_vecs, d_vecs, semantic_scores = self.forward(batch)
        contrastive_aux_metrics: dict[str, float] = {}
        negative_indices, safe_negative_fallbacks = self._sample_negative_indices(batch)
        negative_d_vecs = d_vecs[negative_indices]
        semantic_pair_scores = torch.stack(
            [
                torch.sum(q_vecs * d_vecs, dim=1),
                torch.sum(q_vecs * negative_d_vecs, dim=1),
            ],
            dim=1,
        )
        if self.loss_name == "contrastive":
            semantic_loss = self._compute_contrastive_loss(
                batch=batch,
                q_vecs=q_vecs,
                d_vecs=d_vecs,
                metrics=contrastive_aux_metrics,
            )
        elif self.loss_name == "focal-contrastive":
            semantic_loss = _pairwise_focal_loss(self.loss_fn, semantic_pair_scores[:, 0], semantic_pair_scores[:, 1])
        else:
            positive_labels = torch.ones(q_vecs.shape[0], device=q_vecs.device)
            negative_labels = torch.zeros(q_vecs.shape[0], device=q_vecs.device)
            semantic_loss = self.loss_fn(q_vecs, d_vecs, positive_labels) + self.loss_fn(q_vecs, negative_d_vecs, negative_labels)

        positive_sparse_scores, negative_sparse_scores = self._build_sparse_pair_scores(batch, negative_indices)
        adaptive_output = self.runner.adaptive_semantic_pair_scores(
            q_vecs,
            d_vecs,
            negative_doc_vectors=negative_d_vecs,
            return_details=True,
        )
        assert isinstance(adaptive_output, tuple)
        adaptive_semantic_pair_scores, adaptive_metrics = adaptive_output
        diag_identity_penalty = adaptive_metrics.pop("DiagGateIdentityPenalty", None)
        diag_saturation_penalty = adaptive_metrics.pop("DiagGateSaturationPenalty", None)
        if self.loss_name == "soft-contrastive":
            adaptive_semantic_pair_scores = self._scale_soft_contrastive_scores(adaptive_semantic_pair_scores)
        sparse_pair_scores = torch.stack([positive_sparse_scores, negative_sparse_scores], dim=1)
        mixed_output = self.runner.mix_scores(
            adaptive_semantic_pair_scores,
            sparse_pair_scores,
            query_vectors=q_vecs,
            return_details=True,
        )
        assert isinstance(mixed_output, tuple)
        mixed_pair_scores, mix_metrics = mixed_output
        if self.loss_name == "focal-contrastive":
            mix_loss = _pairwise_focal_loss(self.loss_fn, mixed_pair_scores[:, 0], mixed_pair_scores[:, 1])
        else:
            mix_loss = self._pairwise_similarity_loss(
                mixed_pair_scores[:, 0],
                mixed_pair_scores[:, 1],
            )

        main_loss = semantic_loss if self.alpha_train_only and self.runner.gamma_joint else mix_loss
        aux_metrics: dict[str, float] = {}
        entropy_bonus = None
        diag_identity_bonus = semantic_loss.new_zeros(())
        diag_saturation_bonus = semantic_loss.new_zeros(())
        effective_alpha_mix_weight = self._effective_alpha_mix_weight()
        if diag_identity_penalty is not None:
            assert isinstance(diag_identity_penalty, torch.Tensor)
            diag_identity_bonus = self.query_diagonal_identity_weight * diag_identity_penalty
        if diag_saturation_penalty is not None:
            assert isinstance(diag_saturation_penalty, torch.Tensor)
            diag_saturation_bonus = self.query_diagonal_saturation_weight * diag_saturation_penalty
        if self.runner.gamma_joint:
            alpha = self.runner.alpha_weights(q_vecs)
            alpha_clamped = alpha.clamp(min=1e-6, max=1.0 - 1e-6)
            alpha_entropy = -(
                alpha_clamped * torch.log(alpha_clamped) + (1.0 - alpha_clamped) * torch.log(1.0 - alpha_clamped)
            ).mean()
            entropy_bonus = self.alpha_entropy_weight * alpha_entropy
            if self.alpha_train_only:
                loss = semantic_loss + effective_alpha_mix_weight * mix_loss - entropy_bonus
            else:
                loss = mix_loss - entropy_bonus
            loss = loss + diag_identity_bonus + diag_saturation_bonus
            with torch.no_grad():
                aux_metrics["alpha_aux_loss"] = float(semantic_loss.detach().item())
                aux_metrics["alpha_mix_loss"] = float(mix_loss.detach().item())
                aux_metrics["alpha_entropy"] = float(alpha_entropy.detach().item())
                aux_metrics["alpha_entropy_bonus"] = float(entropy_bonus.detach().item())
                aux_metrics["alpha_entropy_weight"] = float(self.alpha_entropy_weight)
                aux_metrics["alpha_train_only"] = float(int(self.alpha_train_only))
                aux_metrics["alpha_mix_weight"] = float(self.alpha_mix_weight)
                aux_metrics["alpha_mix_weight_effective"] = float(effective_alpha_mix_weight)
                aux_metrics["alpha_mix_weight_warmup_steps"] = float(self.alpha_mix_weight_warmup_steps)
                aux_metrics["DiagGateIdentityPenalty"] = (
                    float(diag_identity_penalty.detach().item()) if diag_identity_penalty is not None else 0.0
                )
                aux_metrics["DiagGateSaturationPenalty"] = (
                    float(diag_saturation_penalty.detach().item()) if diag_saturation_penalty is not None else 0.0
                )
                aux_metrics["DiagGateIdentityBonus"] = float(diag_identity_bonus.detach().item())
                aux_metrics["DiagGateSaturationBonus"] = float(diag_saturation_bonus.detach().item())
                aux_metrics["DiagGateIdentityWeight"] = float(self.query_diagonal_identity_weight)
                aux_metrics["DiagGateSaturationWeight"] = float(self.query_diagonal_saturation_weight)
                aux_metrics.update(adaptive_metrics)
        else:
            loss = main_loss

        optimizer.zero_grad()
        self.manual_backward(self._adjust_loss(loss))

        grad_metrics = self._gamma_grad_metrics()
        grad_metrics["Grad_Norm_model"] = self._grad_norm(self.runner.model.parameters())
        batch_metrics = self._batch_retrieval_metrics(semantic_scores.detach())
        geom_metrics = embedding_geometry_metrics(
            q_vecs.detach(),
            d_vecs.detach(),
            tau=self.loss_fn.tau if isinstance(self.loss_fn, ContrastiveLoss) else None,
        )
        optimizer.step()
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            if isinstance(scheduler, list):
                scheduler = scheduler[0]
            if scheduler is not None:
                scheduler.step()

        payload = {
            "Loss": float(loss.detach().item()),
            "MainLoss": float(main_loss.detach().item()),
            "FreezeEncoder": int(self.freeze_encoder),
            "GammaJoint": int(self.runner.gamma_joint),
            "NegativeSamplingMode": self.negative_sampling_mode,
            "MinNegativeIDFRecall": (
                float(self.min_negative_inverse_idf_recall) if self.min_negative_inverse_idf_recall is not None else -1.0
            ),
            "MaxNegativeIDFRecall": (
                float(self.max_negative_inverse_idf_recall) if self.max_negative_inverse_idf_recall is not None else -1.0
            ),
            "HardNegativeTopK": float(self.hard_negative_top_k) if self.hard_negative_top_k is not None else -1.0,
            "ActivationFn": self.runner.activation_fn,
            "Optimizer": self.optimizer_name,
            "LossName": self.loss_name,
            "FocalGamma": self.focal_gamma,
            "ContrastiveTemperature": self.contrastive_temperature,
            "SoftContrastiveTemperature": self.soft_contrastive_temperature,
            **contrastive_aux_metrics,
            **self._gamma_metrics(),
            **aux_metrics,
            **mix_metrics,
            **batch_metrics,
            **grad_metrics,
            **geom_metrics,
            "SafeNegativeFallbacks": float(safe_negative_fallbacks),
            "SafeNegativeFallbackRate": float(safe_negative_fallbacks / max(int(q_vecs.shape[0]), 1)),
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
        loss_name: str = "soft-contrastive",
        margin: float = 0.5,
        focal_gamma: float = 2.0,
        contrastive_temperature: float = 0.1,
        soft_contrastive_temperature: float = 1.0,
        lr_encoder: float = 2e-5,
        weight_decay: float = 0.01,
        grad_acc_steps: int = 6,
        optimizer_name: str = "auto",
        contrastive_learnable_temperature: bool = True,
        contrastive_decoupled: bool = True,
        contrastive_simcse_dropout_weight: float = 0.0,
        contrastive_soft_fn_attract_weight: float = 0.0,
        contrastive_soft_fn_topk: int = 1,
        lexical_text_by_content: dict[str, str] | None = None,
        save_dir: str | Path | None = None,
        metrics_path: str | Path | None = None,
    ):
        super().__init__()
        self.runner = runner
        self.loss_name = loss_name
        self.margin = margin
        self.focal_gamma = focal_gamma
        self.contrastive_temperature = contrastive_temperature
        self.soft_contrastive_temperature = soft_contrastive_temperature
        self.lr_encoder = lr_encoder
        self.weight_decay = weight_decay
        self.grad_acc_steps = grad_acc_steps
        default_optimizer = "adafactor" if loss_name == "contrastive" else "adamw"
        self.optimizer_name = _normalize_optimizer_name(optimizer_name, default=default_optimizer)
        self.contrastive_learnable_temperature = bool(contrastive_learnable_temperature)
        self.contrastive_decoupled = bool(contrastive_decoupled)
        self.contrastive_simcse_dropout_weight = float(contrastive_simcse_dropout_weight)
        self.contrastive_soft_fn_attract_weight = float(contrastive_soft_fn_attract_weight)
        self.contrastive_soft_fn_topk = max(int(contrastive_soft_fn_topk), 1)
        if self.contrastive_simcse_dropout_weight < 0.0:
            raise ValueError("contrastive_simcse_dropout_weight must be >= 0")
        if self.contrastive_soft_fn_attract_weight < 0.0:
            raise ValueError("contrastive_soft_fn_attract_weight must be >= 0")
        self.lexical_text_by_content = lexical_text_by_content or {}
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
            return ContrastiveLoss(
                temperature=self.contrastive_temperature,
                reduction="mean",
                learnable_temperature=self.contrastive_learnable_temperature,
                decoupled=self.contrastive_decoupled,
            )
        if self.loss_name == "soft-contrastive":
            return SoftContrastiveLoss(
                margin=self.margin,
                size_average=True,
                temperature=self.soft_contrastive_temperature,
            )
        if self.loss_name == "focal-contrastive":
            return FocalLoss(gamma=self.focal_gamma, reduction="mean")
        raise ValueError(f"Unsupported loss={self.loss_name}. Use one of contrastive,soft-contrastive,focal-contrastive")

    @staticmethod
    def _model_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "pos_input_ids", "pos_attention_mask", "group_ids"}
        }

    @staticmethod
    def _sample_negative_indices(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        return _sample_safe_negative_indices(
            doc_key_ids=batch["doc_key_id"],
            content_key_ids=batch.get("content_key_id"),
            query_key_ids=batch.get("query_key_id"),
        )

    def _decode_content_from_batches(self, batch) -> tuple[list[str], list[str]]:
        tokenizer = self.runner.processor.tokenizer
        query_prefix = self.runner.processor.queries_prefix
        doc_prefix = self.runner.processor.pos_queries_prefix

        queries = [
            tokenizer.decode(q_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            .removeprefix(query_prefix)
            .strip()
            for q_tokens in batch["input_ids"]
        ]
        docs = [
            tokenizer.decode(d_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).removeprefix(doc_prefix).strip()
            for d_tokens in batch["pos_input_ids"]
        ]
        return queries, docs

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
        if self.optimizer_name == "adafactor":
            optimizer = Adafactor(
                encoder_params,
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                lr=None,
            )
            return [optimizer], [AdafactorSchedule(optimizer)]
        if self.optimizer_name == "adagrad":
            return torch.optim.Adagrad(
                [
                    {
                        "params": encoder_params,
                        "lr": self.lr_encoder,
                        "weight_decay": self.weight_decay,
                    }
                ]
            )

        return torch.optim.AdamW(
            [
                {
                    "params": encoder_params,
                    "lr": self.lr_encoder,
                    "weight_decay": self.weight_decay,
                }
            ]
        )

    def _adjust_loss(self, loss: torch.Tensor) -> torch.Tensor:
        mean_loss = loss.mean()
        if self.grad_acc_steps > 1:
            mean_loss = mean_loss / self.grad_acc_steps
        return mean_loss

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q_vecs, d_vecs = self.runner(self._model_batch(batch), norm=True)
        return q_vecs @ d_vecs.T

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        q_vecs, d_vecs = self.runner(self._model_batch(batch), norm=True)
        output = q_vecs @ d_vecs.T
        contrastive_aux_metrics: dict[str, float] = {}

        if self.loss_name == "soft-contrastive":
            negative_indices, safe_negative_fallbacks = self._sample_negative_indices(batch)
            negative_vecs = d_vecs[negative_indices]
            positive_labels = torch.ones(q_vecs.shape[0], device=q_vecs.device)
            negative_labels = torch.zeros(q_vecs.shape[0], device=q_vecs.device)
            positive_loss = self.loss_fn(q_vecs, d_vecs, positive_labels)
            negative_loss = self.loss_fn(q_vecs, negative_vecs, negative_labels)
            loss = positive_loss + negative_loss
        elif self.loss_name == "focal-contrastive":
            negative_indices, safe_negative_fallbacks = self._sample_negative_indices(batch)
            negative_vecs = d_vecs[negative_indices]
            positive_scores = torch.sum(q_vecs * d_vecs, dim=1)
            negative_scores = torch.sum(q_vecs * negative_vecs, dim=1)
            loss = _pairwise_focal_loss(self.loss_fn, positive_scores, negative_scores)
        else:
            safe_negative_fallbacks = 0
            loss_fn: ContrastiveLoss = self.loss_fn
            main_per_row = loss_fn.info_nce(q_vecs, d_vecs, reduction="none")
            extra = main_per_row.new_zeros(())
            simcse_per_row = None
            soft_fn_per_row = None
            if self.contrastive_simcse_dropout_weight > 0.0:
                q_alt_vecs, _ = self.runner(self._model_batch(batch), norm=True)
                simcse_per_row = loss_fn.simcse_term(q_vecs, q_alt_vecs, reduction="none")
                extra = extra + self.contrastive_simcse_dropout_weight * simcse_per_row.mean()
            if self.contrastive_soft_fn_attract_weight > 0.0 and q_vecs.shape[0] > 1:
                soft_fn_per_row = loss_fn.soft_fn_term(q_vecs, d_vecs, topk=self.contrastive_soft_fn_topk, reduction="none")
                extra = extra + self.contrastive_soft_fn_attract_weight * soft_fn_per_row.mean()
            loss = main_per_row.mean() + extra
            with torch.no_grad():
                contrastive_aux_metrics["ContrastiveMainLoss"] = float(main_per_row.mean().item())
                if simcse_per_row is not None:
                    contrastive_aux_metrics["ContrastiveSimCSELoss"] = float(simcse_per_row.mean().item())
                if soft_fn_per_row is not None:
                    contrastive_aux_metrics["ContrastiveSoftFNLoss"] = float(soft_fn_per_row.mean().item())
                contrastive_aux_metrics["ContrastiveSimCSEWeight"] = float(self.contrastive_simcse_dropout_weight)
                contrastive_aux_metrics["ContrastiveSoftFNWeight"] = float(self.contrastive_soft_fn_attract_weight)
                contrastive_aux_metrics["ContrastiveSoftFNTopK"] = float(self.contrastive_soft_fn_topk)
                contrastive_aux_metrics["ContrastiveLearnableTau"] = float(int(self.contrastive_learnable_temperature))
                contrastive_aux_metrics["ContrastiveDecoupled"] = float(int(self.contrastive_decoupled))

        optimizer.zero_grad()
        self.manual_backward(self._adjust_loss(loss))

        grad_norm_model = self._grad_norm(self.runner.model.parameters())
        batch_metrics = self._batch_retrieval_metrics(output.detach())
        geom_metrics = embedding_geometry_metrics(
            q_vecs.detach(),
            d_vecs.detach(),
            tau=self.loss_fn.tau if isinstance(self.loss_fn, ContrastiveLoss) else None,
        )
        optimizer.step()

        if self.loss_name == "contrastive":
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, list):
                scheduler = scheduler[0]
            if scheduler is not None:
                scheduler.step()

        payload = {
            "Loss": float(loss.detach().item()),
            "FreezeEncoder": 0,
            "TrainingMode": "encoder-only",
            "Optimizer": self.optimizer_name,
            "LossName": self.loss_name,
            "FocalGamma": self.focal_gamma,
            "ContrastiveTemperature": self.contrastive_temperature,
            "SoftContrastiveTemperature": self.soft_contrastive_temperature,
            **contrastive_aux_metrics,
            "Grad_Norm_model": grad_norm_model,
            "SafeNegativeFallbacks": float(safe_negative_fallbacks),
            "SafeNegativeFallbackRate": float(safe_negative_fallbacks / max(int(q_vecs.shape[0]), 1)),
            **batch_metrics,
            **geom_metrics,
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
