import math
import os
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
from justatom.training.memory_bank import _ContrastiveMemoryBank


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

    needs_idf = (
        max_negative_inverse_idf_recall is not None
        or min_negative_inverse_idf_recall is not None
        or negative_sampling_mode != "safe-random"
    )
    if not needs_idf:
        # Vectorized GPU fast-path for default safe-random mode: build a [B,B]
        # validity mask, sample one valid index per row via Gumbel-argmax in a
        # single launch. Avoids the per-row Python loop and 64+ syncs.
        valid = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device).fill_diagonal_(False)
        valid &= doc_key_ids.unsqueeze(0) != doc_key_ids.unsqueeze(1)
        if content_key_ids is not None:
            valid &= content_key_ids.unsqueeze(0) != content_key_ids.unsqueeze(1)
        if query_key_ids is not None:
            valid &= query_key_ids.unsqueeze(0) != query_key_ids.unsqueeze(1)
        any_valid = valid.any(dim=1)
        # Fallback rows: relax to (different doc_key_id, different content_key_id) only.
        if not bool(any_valid.all().item()):
            fallback_valid = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device).fill_diagonal_(False)
            fallback_valid &= doc_key_ids.unsqueeze(0) != doc_key_ids.unsqueeze(1)
            if content_key_ids is not None:
                fallback_valid &= content_key_ids.unsqueeze(0) != content_key_ids.unsqueeze(1)
            relaxed = ~any_valid.unsqueeze(1)
            valid = torch.where(relaxed, fallback_valid, valid)
            any_valid = valid.any(dim=1)
            if not bool(any_valid.all().item()):
                hard_fallback = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device).fill_diagonal_(False)
                fallback_count = int((~any_valid).sum().item())
                valid = torch.where(any_valid.unsqueeze(1), valid, hard_fallback)
        gumbel = -torch.log(-torch.log(torch.rand(batch_size, batch_size, device=device).clamp_min(1e-9)).clamp_min(1e-9))
        gumbel = gumbel.masked_fill(~valid, float("-inf"))
        negative_indices = gumbel.argmax(dim=1)
        return negative_indices, fallback_count

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


def _sanitize_contrastive_temperature(loss_fn, optimizer=None) -> bool:
    if not isinstance(loss_fn, ContrastiveLoss):
        return False
    changed = bool(loss_fn.clamp_temperature_())
    if changed and optimizer is not None:
        state = optimizer.state.get(loss_fn.log_tau, {})
        for value in state.values():
            if torch.is_tensor(value):
                value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    return changed


def _contrastive_temperature_grad_norm(loss_fn) -> float:
    if not isinstance(loss_fn, ContrastiveLoss):
        return 0.0
    grad = loss_fn.log_tau.grad
    if grad is None:
        return 0.0
    return float(grad.detach().float().norm(2).cpu().item())


def _assert_finite_contrastive_temperature_grad(loss_fn) -> None:
    if not isinstance(loss_fn, ContrastiveLoss):
        return
    grad = loss_fn.log_tau.grad
    if grad is None or bool(torch.isfinite(grad.detach()).all().item()):
        return
    raise RuntimeError(
        "Non-finite contrastive temperature gradient detected. "
        f"log_tau={float(loss_fn.log_tau.detach().cpu().item())}, "
        f"tau={float(loss_fn.tau.detach().cpu().item())}, "
        f"grad={float(grad.detach().cpu().item())}"
    )


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
        memory_bank_size: int = 0,
        memory_bank_warmup_steps: int = 0,
        memory_bank_mining_mode: str = "all",
        memory_bank_hard_negatives: int = 0,
        memory_bank_random_negatives: int = 0,
        memory_bank_hard_warmup_steps: int = 0,
        memory_bank_hard_ramp_steps: int = 1,
        memory_bank_too_hard_margin: float | None = None,
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
        self.memory_bank = _ContrastiveMemoryBank(
            memory_bank_size,
            warmup_steps=memory_bank_warmup_steps,
            mining_mode=memory_bank_mining_mode,
            hard_negatives=memory_bank_hard_negatives,
            random_negatives=memory_bank_random_negatives,
            hard_warmup_steps=memory_bank_hard_warmup_steps,
            hard_ramp_steps=memory_bank_hard_ramp_steps,
            too_hard_margin=memory_bank_too_hard_margin,
        )
        mode = str(contrastive_loss_alpha_gate_mode).lower()
        if mode not in {"augment", "convex"}:
            raise ValueError("contrastive_loss_alpha_gate_mode must be one of: augment, convex")
        self.contrastive_loss_alpha_gate_mode = mode
        # Phase-1 anti-collapse regularizers for alpha-gate (augment mode).
        # Default 0.0 keeps prior behaviour; set via env to activate.
        # Refs: Shazeer 2017 (load-balancing loss), Hinton/Jacobs 1991 (entropy bonus).
        self._alpha_gate_prior_weight = float(os.environ.get("ALPHA_GATE_PRIOR_WEIGHT", "0.0"))
        self._alpha_gate_prior_target = float(os.environ.get("ALPHA_GATE_PRIOR_TARGET", "0.05"))
        self._alpha_gate_entropy_weight = float(os.environ.get("ALPHA_GATE_ENTROPY_WEIGHT", "0.0"))
        self._alpha_gate_freeze_steps = max(int(os.environ.get("ALPHA_GATE_FREEZE_STEPS", "0")), 0)
        _freeze_value_env = os.environ.get("ALPHA_GATE_FREEZE_VALUE")
        self._alpha_gate_freeze_value = (
            float(_freeze_value_env) if _freeze_value_env is not None else self._alpha_gate_prior_target
        )
        # Counts training_step calls (batch units), not Lightning optimizer steps.
        # This makes ALPHA_GATE_FREEZE_STEPS independent of grad_acc_steps.
        self._train_batch_count = 0
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

    def _sample_negative_indices(
        self,
        batch: dict[str, torch.Tensor],
        *,
        queries: list[str] | None = None,
        docs: list[str] | None = None,
    ) -> tuple[torch.Tensor, int]:
        needs_text = (
            self.negative_sampling_mode != "safe-random"
            or self.max_negative_inverse_idf_recall is not None
            or self.min_negative_inverse_idf_recall is not None
        )
        if needs_text and queries is None:
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
            import os as _os
            if _os.environ.get("JUSTATOM_GRAD_CHECKPOINT", "0") == "1":
                _hf = getattr(self.runner.model, "model", None)
                if _hf is not None and hasattr(_hf, "gradient_checkpointing_enable"):
                    _hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    if hasattr(_hf, "config"):
                        _hf.config.use_cache = False

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

        # Single MPS->CPU sync per tensor; iterating the CPU tensor afterwards
        # is free, while iterating an MPS tensor would force one sync per row.
        query_token_ids = batch["input_ids"].detach().cpu()
        doc_token_ids = batch["pos_input_ids"].detach().cpu()

        queries = [
            tokenizer.decode(
                q_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            .removeprefix(query_prefix)
            .strip()
            for q_tokens in query_token_ids
        ]
        docs = [
            tokenizer.decode(
                d_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            .removeprefix(doc_prefix)
            .strip()
            for d_tokens in doc_token_ids
        ]
        return queries, docs

    def _build_sparse_pair_scores(
        self,
        batch: dict[str, torch.Tensor],
        negative_indices: torch.Tensor,
        *,
        queries: list[str] | None = None,
        docs: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if queries is None or docs is None:
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

    @staticmethod
    def _resolve_metric_tensors(metrics: dict) -> dict:
        # Batch every torch.Tensor scalar in `metrics` into a single MPS->CPU
        # transfer so we don't pay one device-sync per .item() during the hot path.
        tensor_keys = [k for k, v in metrics.items() if isinstance(v, torch.Tensor)]
        if not tensor_keys:
            return metrics
        stacked = torch.stack([metrics[k].detach().reshape(()).float() for k in tensor_keys])
        for k, f in zip(tensor_keys, stacked.cpu().tolist(), strict=True):
            metrics[k] = float(f)
        return metrics

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

            packed = torch.stack(
                [
                    (rank_positions <= 1).float().mean(),
                    (rank_positions <= min(3, batch_size)).float().mean(),
                    (rank_positions <= min(5, batch_size)).float().mean(),
                    (rank_positions > 1).sum().float(),
                    reciprocal_ranks.mean(),
                    rank_positions_f.mean(),
                    rank_positions_f.median(),
                ]
            ).cpu().tolist()
            hit_at_1, hit_at_3, hit_at_5, errors_at_1, mrr, mean_rank, median_rank = packed

            return {
                "BatchSize": float(batch_size),
                "ErrorsAt1": float(errors_at_1),
                "ErrorRateAt1": float(1.0 - hit_at_1),
                "HitRateAt1": float(hit_at_1),
                "HitRateAt3": float(hit_at_3),
                "HitRateAt5": float(hit_at_5),
                "MRR": float(mrr),
                "MeanRank": float(mean_rank),
                "MedianRank": float(median_rank),
            }

    def configure_optimizers(self):
        mixing_params = [p for p in self.runner.mixing_parameters() if p.requires_grad]
        param_groups = []
        if mixing_params:
            param_groups.append({"params": mixing_params, "lr": self.lr_gamma, "weight_decay": 0.0})

        loss_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
        if loss_params:
            param_groups.append({"params": loss_params, "lr": self.lr_encoder, "weight_decay": 0.0})

        if not self.freeze_encoder:
            excluded_param_ids = {id(p) for p in [*mixing_params, *loss_params]}
            encoder_params = [p for p in self.runner.parameters() if p.requires_grad and id(p) not in excluded_param_ids]
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

    def _is_accumulation_start(self, batch_idx: int) -> bool:
        return batch_idx % max(int(self.grad_acc_steps), 1) == 0

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        grad_acc_steps = max(int(self.grad_acc_steps), 1)
        try:
            is_last_batch = bool(getattr(self.trainer, "is_last_batch", False))
        except RuntimeError:
            is_last_batch = False
        return ((batch_idx + 1) % grad_acc_steps == 0) or is_last_batch

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
        tau_per_query = None
        if getattr(self.runner, "tau_query_conditional", False) and self.runner.tau_head is not None:
            tau_per_query = self.runner.tau_weights(q_vecs, loss_fn.tau)
            metrics["TauQueryMean"] = tau_per_query.detach().mean()
            metrics["TauQueryStd"] = tau_per_query.detach().std(unbiased=False)
            metrics["TauQueryMin"] = tau_per_query.detach().min()
            metrics["TauQueryMax"] = tau_per_query.detach().max()
        memory_negatives, memory_negative_mask, memory_metrics = self.memory_bank.get(
            batch,
            device=q_vecs.device,
            dtype=q_vecs.dtype,
            query_vectors=q_vecs,
            positive_vectors=d_vecs,
            step=int(self.global_step),
        )
        metrics.update(memory_metrics)
        main_per_row = loss_fn.info_nce(
            q_vecs,
            d_vecs,
            reduction="none",
            tau_per_query=tau_per_query,
            memory_negatives=memory_negatives,
            memory_negative_mask=memory_negative_mask,
        )
        with torch.no_grad():
            in_batch_negatives = float(max(int(q_vecs.shape[0]) - 1, 0))
            if memory_negative_mask is None:
                memory_valid_negatives = main_per_row.new_tensor(0.0)
            else:
                memory_valid_negatives = memory_negative_mask.float().sum(dim=1).mean()
            metrics["ContrastiveEffectiveNegativesMean"] = in_batch_negatives + memory_valid_negatives.detach()

        simcse_per_row = None
        if self.contrastive_simcse_dropout_weight > 0.0 and not self.freeze_encoder:
            # Second forward of the *queries* with a fresh dropout mask.
            query_batch = {
                key: batch[key]
                for key in ("input_ids", "attention_mask")
                if key in batch
            }
            (q_alt_vecs,) = self.runner(query_batch, norm=True)
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
            # N1 per-pair gate: when runner.alpha_include_doc is on, alpha sees
            # [q ; d+ ; q*d+]; otherwise it gets q only (legacy per-query gate).
            alpha_q = self.runner.alpha_weights(q_vecs, d_vecs).clamp(min=1e-6, max=1.0 - 1e-6)
            if alpha_q.dim() > 1:
                alpha_q = alpha_q.view(-1)
            alpha_frozen = bool(getattr(self.runner, "alpha_override", None) is not None)
            if self.contrastive_loss_alpha_gate_mode == "convex":
                # alpha * main + (1 - alpha) * aux  (legacy; both terms per-row).
                per_row = alpha_q * main_per_row + (1.0 - alpha_q) * aux_per_row
            else:
                # "augment" (default): main + (1 - alpha) * aux.
                # alpha(q) ~ 1 -> trust main fully (semantic-confident query);
                # alpha(q) ~ 0 -> apply full SimCSE/soft-FN regularization.
                per_row = main_per_row + (1.0 - alpha_q) * aux_per_row
            loss = per_row.mean()
            # Phase-1 anti-collapse: pull batch-mean alpha to a target prior
            # (load-balancing) and bonus query-level Bernoulli entropy.
            if self._alpha_gate_prior_weight > 0.0:
                prior_pen = self._alpha_gate_prior_weight * (alpha_q.mean() - self._alpha_gate_prior_target).pow(2)
                loss = loss + prior_pen
                metrics["AlphaGatePriorLoss"] = prior_pen.detach()
            if self._alpha_gate_entropy_weight > 0.0:
                ent = -(
                    alpha_q * alpha_q.clamp_min(1e-8).log()
                    + (1.0 - alpha_q) * (1.0 - alpha_q).clamp_min(1e-8).log()
                )
                ent_bonus = self._alpha_gate_entropy_weight * ent.mean()
                loss = loss - ent_bonus
                metrics["AlphaGateEntropyBonus"] = ent_bonus.detach()
            metrics["ContrastiveLossAlphaGate"] = 1.0
            metrics["ContrastiveLossAlphaGateAlphaMean"] = alpha_q.detach().mean()
            metrics["ContrastiveLossAlphaGateMode"] = 1.0 if self.contrastive_loss_alpha_gate_mode == "augment" else 0.0
            metrics["AlphaGatePriorWeight"] = float(self._alpha_gate_prior_weight)
            metrics["AlphaGatePriorTarget"] = float(self._alpha_gate_prior_target)
            metrics["AlphaGateEntropyWeight"] = float(self._alpha_gate_entropy_weight)
            metrics["AlphaGateFreezeSteps"] = float(self._alpha_gate_freeze_steps)
            metrics["AlphaGateFreezeValue"] = float(self._alpha_gate_freeze_value)
            metrics["AlphaGateFrozen"] = 1.0 if alpha_frozen else 0.0
        else:
            loss = main_per_row.mean()
            if aux_per_row is not None:
                loss = loss + aux_per_row.mean()
            metrics["ContrastiveLossAlphaGate"] = float(int(self.contrastive_loss_alpha_gate))

        with torch.no_grad():
            metrics["ContrastiveMainLoss"] = main_per_row.mean().detach()
            if simcse_per_row is not None:
                metrics["ContrastiveSimCSELoss"] = simcse_per_row.mean().detach()
            if soft_fn_per_row is not None:
                metrics["ContrastiveSoftFNLoss"] = soft_fn_per_row.mean().detach()
            metrics["ContrastiveSimCSEWeight"] = float(self.contrastive_simcse_dropout_weight)
            metrics["ContrastiveSoftFNWeight"] = float(self.contrastive_soft_fn_attract_weight)
            metrics["ContrastiveSoftFNTopK"] = float(self.contrastive_soft_fn_topk)
            metrics["ContrastiveLearnableTau"] = float(int(self.contrastive_learnable_temperature))
            metrics["ContrastiveDecoupled"] = float(int(self.contrastive_decoupled))
        return loss

    def training_step(self, batch, batch_idx):
        import time as _time_dbg
        _t0 = _time_dbg.perf_counter()
        _stamps: list[tuple[str, float, float]] = []
        _profile_on = os.environ.get("JUSTATOM_PROFILE_STEPS", "0") == "1"
        _mps_on = _profile_on and torch.backends.mps.is_available()

        if _profile_on:
            def _stamp(name: str) -> None:
                if _mps_on:
                    torch.mps.synchronize()
                mem_gb = (torch.mps.current_allocated_memory() / (1024 ** 3)) if _mps_on else 0.0
                _stamps.append((name, _time_dbg.perf_counter() - _t0, mem_gb))
        else:
            def _stamp(name: str) -> None:
                return None

        # Drive alpha-gate freeze via runner override so that ALL three call
        # sites that read alpha_head (SimCSE-gate, gamma-mixer direct, and
        # mix_scores) see the same constant and no gradient flows back. Counter
        # is in batch units (independent of grad_acc_steps).
        if self.runner.gamma_joint and self._alpha_gate_freeze_steps > 0:
            if self._train_batch_count < self._alpha_gate_freeze_steps:
                self.runner.alpha_override = float(self._alpha_gate_freeze_value)
            else:
                self.runner.alpha_override = None
        self._train_batch_count += 1

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        temperature_sanitized = _sanitize_contrastive_temperature(self.loss_fn, optimizer)
        if self._is_accumulation_start(batch_idx):
            optimizer.zero_grad(set_to_none=True)
        _stamp("zero_grad")
        q_vecs, d_vecs, semantic_scores = self.forward(batch)
        _stamp("forward")
        contrastive_aux_metrics: dict[str, float] = {}
        # Decode once per step and share between sparse-scoring and (optionally) negative sampling.
        decoded_queries, decoded_docs = self._decode_content_from_batches(batch)
        _stamp("decode")
        negative_indices, safe_negative_fallbacks = self._sample_negative_indices(
            batch, queries=decoded_queries, docs=decoded_docs
        )
        _stamp("neg_sample")
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
        _stamp("semantic_loss")

        positive_sparse_scores, negative_sparse_scores = self._build_sparse_pair_scores(
            batch, negative_indices, queries=decoded_queries, docs=decoded_docs
        )
        _stamp("sparse_pair")
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
            pos_doc_vectors=d_vecs,
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
            alpha = self.runner.alpha_weights(q_vecs, d_vecs)
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
                aux_metrics["alpha_aux_loss"] = semantic_loss.detach()
                aux_metrics["alpha_mix_loss"] = mix_loss.detach()
                aux_metrics["alpha_entropy"] = alpha_entropy.detach()
                aux_metrics["alpha_entropy_bonus"] = entropy_bonus.detach()
                aux_metrics["alpha_entropy_weight"] = float(self.alpha_entropy_weight)
                aux_metrics["alpha_train_only"] = float(int(self.alpha_train_only))
                aux_metrics["alpha_mix_weight"] = float(self.alpha_mix_weight)
                aux_metrics["alpha_mix_weight_effective"] = float(effective_alpha_mix_weight)
                aux_metrics["alpha_mix_weight_warmup_steps"] = float(self.alpha_mix_weight_warmup_steps)
                aux_metrics["DiagGateIdentityPenalty"] = (
                    diag_identity_penalty.detach() if diag_identity_penalty is not None else 0.0
                )
                aux_metrics["DiagGateSaturationPenalty"] = (
                    diag_saturation_penalty.detach() if diag_saturation_penalty is not None else 0.0
                )
                aux_metrics["DiagGateIdentityBonus"] = diag_identity_bonus.detach()
                aux_metrics["DiagGateSaturationBonus"] = diag_saturation_bonus.detach()
                aux_metrics["DiagGateIdentityWeight"] = float(self.query_diagonal_identity_weight)
                aux_metrics["DiagGateSaturationWeight"] = float(self.query_diagonal_saturation_weight)
                aux_metrics.update(adaptive_metrics)
        else:
            loss = main_loss

        if not torch.isfinite(loss.detach()):
            raise RuntimeError(
                "Non-finite training loss detected. "
                f"loss={float(loss.detach().cpu().item())}, "
                f"memory_bank_size={self.memory_bank.current_size}, "
                f"memory_bank_mode={self.memory_bank.mining_mode}"
            )
        self.manual_backward(self._adjust_loss(loss))
        _assert_finite_contrastive_temperature_grad(self.loss_fn)
        _stamp("backward")

        grad_metrics = self._gamma_grad_metrics()
        grad_metrics["Grad_norm_log_tau"] = _contrastive_temperature_grad_norm(self.loss_fn)
        grad_metrics["Grad_Norm_model"] = self._grad_norm(self.runner.model.parameters())
        _stamp("grad_metrics")
        batch_metrics = self._batch_retrieval_metrics(semantic_scores.detach())
        _stamp("batch_metrics")
        geom_metrics = embedding_geometry_metrics(
            q_vecs.detach(),
            d_vecs.detach(),
            tau=self.loss_fn.tau if isinstance(self.loss_fn, ContrastiveLoss) else None,
        )
        _stamp("geom")
        optimizer_stepped = self._should_step_optimizer(batch_idx)
        if optimizer_stepped:
            optimizer.step()
            temperature_sanitized = _sanitize_contrastive_temperature(self.loss_fn, optimizer) or temperature_sanitized
            scheduler = self.lr_schedulers()
            if scheduler is not None:
                if isinstance(scheduler, list):
                    scheduler = scheduler[0]
                if scheduler is not None:
                    scheduler.step()
        _stamp("opt_step")
        if self.loss_name == "contrastive":
            self.memory_bank.enqueue(d_vecs, batch)
        _stamp("memory_bank")
        grad_metrics["TemperatureSanitized"] = float(int(temperature_sanitized))

        self._resolve_metric_tensors(contrastive_aux_metrics)
        self._resolve_metric_tensors(aux_metrics)
        loss_main_floats = torch.stack([loss.detach().reshape(()).float(), main_loss.detach().reshape(()).float()]).cpu().tolist()
        payload = {
            "Loss": float(loss_main_floats[0]),
            "MainLoss": float(loss_main_floats[1]),
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
            "OptimizerStepped": float(int(optimizer_stepped)),
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

        _stamp("log")
        if _profile_on and _stamps and (batch_idx < 3 or batch_idx % 10 == 0):
            prev = 0.0
            parts = []
            for _name, _t, _mem in _stamps:
                parts.append(f"{_name}={_t - prev:.3f}s/{_mem:.2f}G")
                prev = _t
            line = f"[step {batch_idx}] total={_stamps[-1][1]:.3f}s  " + "  ".join(parts)
            print(line, flush=True)
            try:
                _log_path = os.environ.get("JUSTATOM_STEP_LOG", "/tmp/justatom_step_timing.log")
                with open(_log_path, "a") as _fh:
                    _fh.write(line + "\n")
            except Exception:
                pass

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
        memory_bank_size: int = 0,
        memory_bank_warmup_steps: int = 0,
        memory_bank_mining_mode: str = "all",
        memory_bank_hard_negatives: int = 0,
        memory_bank_random_negatives: int = 0,
        memory_bank_hard_warmup_steps: int = 0,
        memory_bank_hard_ramp_steps: int = 1,
        memory_bank_too_hard_margin: float | None = None,
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
        self.memory_bank = _ContrastiveMemoryBank(
            memory_bank_size,
            warmup_steps=memory_bank_warmup_steps,
            mining_mode=memory_bank_mining_mode,
            hard_negatives=memory_bank_hard_negatives,
            random_negatives=memory_bank_random_negatives,
            hard_warmup_steps=memory_bank_hard_warmup_steps,
            hard_ramp_steps=memory_bank_hard_ramp_steps,
            too_hard_margin=memory_bank_too_hard_margin,
        )
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
        loss_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
        param_groups = []
        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": self.lr_encoder,
                    "weight_decay": self.weight_decay,
                }
            )
        if loss_params:
            param_groups.append({"params": loss_params, "lr": self.lr_encoder, "weight_decay": 0.0})
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

    def _is_accumulation_start(self, batch_idx: int) -> bool:
        return batch_idx % max(int(self.grad_acc_steps), 1) == 0

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        grad_acc_steps = max(int(self.grad_acc_steps), 1)
        try:
            is_last_batch = bool(getattr(self.trainer, "is_last_batch", False))
        except RuntimeError:
            is_last_batch = False
        return ((batch_idx + 1) % grad_acc_steps == 0) or is_last_batch

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q_vecs, d_vecs = self.runner(self._model_batch(batch), norm=True)
        return q_vecs @ d_vecs.T

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        temperature_sanitized = _sanitize_contrastive_temperature(self.loss_fn, optimizer)
        if self._is_accumulation_start(batch_idx):
            optimizer.zero_grad(set_to_none=True)
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
            memory_negatives, memory_negative_mask, memory_metrics = self.memory_bank.get(
                batch,
                device=q_vecs.device,
                dtype=q_vecs.dtype,
                query_vectors=q_vecs,
                positive_vectors=d_vecs,
                step=int(self.global_step),
            )
            main_per_row = loss_fn.info_nce(
                q_vecs,
                d_vecs,
                reduction="none",
                memory_negatives=memory_negatives,
                memory_negative_mask=memory_negative_mask,
            )
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
                contrastive_aux_metrics.update(memory_metrics)
                in_batch_negatives = float(max(int(q_vecs.shape[0]) - 1, 0))
                memory_valid_negatives = (
                    0.0
                    if memory_negative_mask is None
                    else float(memory_negative_mask.float().sum(dim=1).mean().item())
                )
                contrastive_aux_metrics["ContrastiveEffectiveNegativesMean"] = (
                    in_batch_negatives + memory_valid_negatives
                )

        if not torch.isfinite(loss.detach()):
            raise RuntimeError(
                "Non-finite training loss detected. "
                f"loss={float(loss.detach().cpu().item())}, "
                f"memory_bank_size={self.memory_bank.current_size}, "
                f"memory_bank_mode={self.memory_bank.mining_mode}"
            )
        self.manual_backward(self._adjust_loss(loss))
        _assert_finite_contrastive_temperature_grad(self.loss_fn)

        grad_norm_model = self._grad_norm(self.runner.model.parameters())
        grad_norm_log_tau = _contrastive_temperature_grad_norm(self.loss_fn)
        batch_metrics = self._batch_retrieval_metrics(output.detach())
        geom_metrics = embedding_geometry_metrics(
            q_vecs.detach(),
            d_vecs.detach(),
            tau=self.loss_fn.tau if isinstance(self.loss_fn, ContrastiveLoss) else None,
        )
        optimizer_stepped = self._should_step_optimizer(batch_idx)
        if optimizer_stepped:
            optimizer.step()
            temperature_sanitized = _sanitize_contrastive_temperature(self.loss_fn, optimizer) or temperature_sanitized
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, list):
                scheduler = scheduler[0]
            if scheduler is not None:
                scheduler.step()
        if self.loss_name == "contrastive":
            self.memory_bank.enqueue(d_vecs, batch)

        _BaseGammaLightningTrainer._resolve_metric_tensors(contrastive_aux_metrics)
        payload = {
            "Loss": float(loss.detach().item()),
            "FreezeEncoder": 0,
            "TrainingMode": "encoder-only",
            "Optimizer": self.optimizer_name,
            "OptimizerStepped": float(int(optimizer_stepped)),
            "LossName": self.loss_name,
            "FocalGamma": self.focal_gamma,
            "ContrastiveTemperature": self.contrastive_temperature,
            "SoftContrastiveTemperature": self.soft_contrastive_temperature,
            **contrastive_aux_metrics,
            "Grad_Norm_model": grad_norm_model,
            "Grad_norm_log_tau": grad_norm_log_tau,
            "TemperatureSanitized": float(int(temperature_sanitized)),
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
