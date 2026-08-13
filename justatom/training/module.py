from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytorch_lightning as L
import torch
from torch import nn

from justatom.logging.io import CSVLogger
from justatom.training.alpha_gate import QueryAlphaGate
from justatom.training.config import MarginMode, TrainConfig, parse_train_config, train_config_to_dict
from justatom.training.gradient_projection import project_conflicting_gradients
from justatom.training.memory_bank import ContrastiveMemoryBank, QueryMarginHead
from justatom.training.methods import resolve_method
from justatom.training.objective import ContrastiveObjective, ObjectiveInputs, ObjectiveOutput
from justatom.training.reranker import build_reranker
from justatom.training.sampling import inverse_idf_recall, sample_safe_negative_indices
from justatom.training.telemetry import batch_retrieval_metrics, resolve_metric_tensors, scalar_distribution


class ContrastiveTrainingModule(L.LightningModule):
    """Single compositional training path for vanilla, atom_gate, and atomic."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        config: TrainConfig,
        objective: ContrastiveObjective,
        alpha_gate: QueryAlphaGate | None,
        memory_bank: ContrastiveMemoryBank | None,
        margin_head: QueryMarginHead | None,
        lexical_lookup: dict[str, str | list[str]] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.objective = objective
        self.alpha_gate = alpha_gate
        self.memory_bank = memory_bank
        self.margin_head = margin_head
        self.lexical_lookup = lexical_lookup
        self._last_negative_indices: torch.Tensor | None = None
        self.metrics_path = None if config.telemetry.metrics_path is None else Path(config.telemetry.metrics_path)
        self.metrics_logger = CSVLogger(self.metrics_path) if self.metrics_path is not None else None
        self.automatic_optimization = not config.gradient_projection.enabled

    @classmethod
    def build(
        cls,
        encoder: nn.Module,
        config: TrainConfig,
        *,
        lexical_lookup: dict[str, str | list[str]] | None = None,
    ) -> "ContrastiveTrainingModule":
        config = resolve_method(config)
        embedding_dim = int(getattr(encoder, "output_dims"))
        alpha_gate = QueryAlphaGate(embedding_dim, config.alpha_gate) if config.alpha_gate.enabled else None
        memory_bank = (
            ContrastiveMemoryBank(config.memory_bank, reranker=build_reranker(config.reranker))
            if config.memory_bank.enabled
            else None
        )
        margin_head = (
            QueryMarginHead(embedding_dim, config.memory_bank.margin)
            if config.memory_bank.margin.mode is MarginMode.QUERY
            else None
        )
        return cls(
            encoder=encoder,
            config=config,
            objective=ContrastiveObjective(config.objective),
            alpha_gate=alpha_gate,
            memory_bank=memory_bank,
            margin_head=margin_head,
            lexical_lookup=lexical_lookup,
        )

    @property
    def needs_simcse(self) -> bool:
        return self.config.objective.simcse_dropout_weight > 0.0

    def effective_alpha_mix_weight(self, step: int) -> float:
        target = float(self.config.alpha_gate.mix_weight)
        warmup = int(self.config.alpha_gate.mix_weight_warmup_steps)
        if warmup <= 0:
            return target
        progress = min(max(float(step), 0.0) / float(warmup), 1.0)
        return target * progress

    def adjust_loss_for_accumulation(self, loss: torch.Tensor) -> torch.Tensor:
        return loss / max(int(self.config.optimization.grad_acc_steps), 1)

    def is_accumulation_start(self, batch_idx: int) -> bool:
        return batch_idx % max(int(self.config.optimization.grad_acc_steps), 1) == 0

    def should_step_optimizer(self, batch_idx: int) -> bool:
        steps = max(int(self.config.optimization.grad_acc_steps), 1)
        try:
            is_last_batch = bool(getattr(self.trainer, "is_last_batch", False))
        except RuntimeError:
            is_last_batch = False
        return ((batch_idx + 1) % steps == 0) or is_last_batch

    def _encode_dropout_query_view(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.encoder.encode_queries(batch)

    def _negative_indices(self, batch: dict[str, Any]) -> torch.Tensor:
        if "doc_key_id" not in batch:
            batch_size = int(batch["queries"].shape[0] if "queries" in batch else batch["input_ids"].shape[0])
            indices = torch.arange(batch_size, device=self.device)
            return torch.roll(indices, shifts=1)
        indices, _ = sample_safe_negative_indices(
            doc_key_ids=batch["doc_key_id"],
            content_key_ids=batch.get("content_key_id"),
            query_key_ids=batch.get("query_key_id"),
        )
        return indices

    def _semantic_pair_scores(
        self,
        queries: torch.Tensor,
        positives: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        self._last_negative_indices = self._negative_indices(batch)
        positive_scores = (queries * positives).sum(dim=-1)
        negative_scores = (queries * positives[self._last_negative_indices]).sum(dim=-1)
        return torch.stack((positive_scores, negative_scores), dim=1)

    def _lexical_pair_scores(self, batch: dict[str, Any]) -> torch.Tensor | None:
        if self.lexical_lookup is None or self._last_negative_indices is None:
            return None
        queries = batch.get("query_text")
        documents = batch.get("content_text")
        if queries is None or documents is None:
            tokenizer = getattr(getattr(self.encoder, "processor", None), "tokenizer", None)
            if tokenizer is None or "input_ids" not in batch or "pos_input_ids" not in batch:
                return None
            queries = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            documents = tokenizer.batch_decode(batch["pos_input_ids"], skip_special_tokens=True)
            queries = [self._remove_text_prefix(text, self.config.model.query_prefix) for text in queries]
            documents = [self._remove_text_prefix(text, self.config.model.content_prefix) for text in documents]
        documents = [self.lexical_lookup.get(str(document), str(document)) for document in documents]
        negative_indices = self._last_negative_indices.detach().cpu().tolist()
        positive_scores = [inverse_idf_recall(str(query), str(document)) for query, document in zip(queries, documents)]
        negative_scores = [
            inverse_idf_recall(str(query), str(documents[negative_idx])) for query, negative_idx in zip(queries, negative_indices)
        ]
        return torch.tensor(
            list(zip(positive_scores, negative_scores)),
            device=self.device,
            dtype=torch.float32,
        )

    @staticmethod
    def _remove_text_prefix(text: str, prefix: str) -> str:
        value = str(text).strip()
        normalized_prefix = str(prefix).strip()
        if normalized_prefix and value.startswith(normalized_prefix):
            return value[len(normalized_prefix) :].strip()
        return value

    def _margin_values(self, queries: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        margin_config = self.config.memory_bank.margin
        if self.margin_head is not None:
            return self.margin_head(queries)
        if self.memory_bank is not None and margin_config.mode is MarginMode.CONSTANT:
            margin = queries.new_full((queries.shape[0],), margin_config.base)
            return margin, margin
        return None, None

    def _reranker_texts(self, batch: dict[str, Any]) -> tuple[list[str], list[str]]:
        queries = batch.get("query_text")
        documents = batch.get("content_text")
        if queries is not None and documents is not None:
            return [str(text) for text in queries], [str(text) for text in documents]
        tokenizer = getattr(getattr(self.encoder, "processor", None), "tokenizer", None)
        if tokenizer is None or "input_ids" not in batch or "pos_input_ids" not in batch:
            raise RuntimeError("reranker requires raw texts or a tokenizer capable of decoding the training batch")
        query_ids = batch["input_ids"].detach().cpu()
        document_ids = batch["pos_input_ids"].detach().cpu()
        decoded_queries = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
        decoded_documents = tokenizer.batch_decode(document_ids, skip_special_tokens=True)
        return (
            [self._remove_text_prefix(text, self.config.model.query_prefix) for text in decoded_queries],
            [self._remove_text_prefix(text, self.config.model.content_prefix) for text in decoded_documents],
        )

    def compute_training_step(self, batch: dict[str, Any], *, step: int) -> ObjectiveOutput:
        queries, positives = self.encoder.encode_pair(batch)
        alpha = None if self.alpha_gate is None else self.alpha_gate(queries)
        query_alt = self._encode_dropout_query_view(batch) if self.needs_simcse else None
        query_texts = None
        positive_texts = None
        if self.memory_bank is not None and self.memory_bank.reranker_enabled:
            query_texts, positive_texts = self._reranker_texts(batch)
        selection = (
            None
            if self.memory_bank is None
            else self.memory_bank.select(
                batch=batch,
                query_vectors=queries,
                positive_vectors=positives,
                step=step,
                query_texts=query_texts,
                positive_texts=positive_texts,
            )
        )
        raw_margin, margin = self._margin_values(queries)

        semantic_pair_scores = None
        lexical_pair_scores = None
        if alpha is not None:
            semantic_pair_scores = self._semantic_pair_scores(queries, positives, batch)
            lexical_pair_scores = self._lexical_pair_scores(batch)
            if lexical_pair_scores is None:
                semantic_pair_scores = None

        output = self.objective(
            ObjectiveInputs(
                queries=queries,
                positives=positives,
                query_alt=query_alt,
                alpha=alpha,
                memory=selection,
                raw_margin=raw_margin,
                margin=margin,
                semantic_pair_scores=semantic_pair_scores,
                lexical_pair_scores=lexical_pair_scores,
                alpha_mix_weight=self.effective_alpha_mix_weight(step),
                alpha_entropy_weight=self.config.alpha_gate.entropy_weight,
            ),
            margin_config=(self.config.memory_bank.margin if self.memory_bank is not None else None),
        )
        if not torch.isfinite(output.loss.detach()):
            raise RuntimeError(f"Non-finite loss at step={step}: {output.loss.detach().cpu().item()}")
        if not torch.isfinite(output.primary_loss.detach()):
            raise RuntimeError(f"Non-finite primary loss at step={step}: {output.primary_loss.detach().cpu().item()}")
        if output.memory_loss is not None and not torch.isfinite(output.memory_loss.detach()):
            raise RuntimeError(f"Non-finite memory loss at step={step}: {output.memory_loss.detach().cpu().item()}")
        with torch.no_grad():
            metrics: dict[str, Any] = dict(output.metrics)
            metrics.update(batch_retrieval_metrics(queries @ positives.T))
            if alpha is not None:
                metrics.update(scalar_distribution("alpha", alpha))
            if raw_margin is not None:
                metrics.update(scalar_distribution("margin/raw", raw_margin))
            if margin is not None:
                metrics.update(scalar_distribution("margin/bounded", margin))
            if selection is not None:
                metrics.update(selection.metrics)
        output = replace(output, metrics=metrics)
        if self.memory_bank is not None:
            self.memory_bank.enqueue(positives, batch, document_texts=positive_texts)
        return output

    @staticmethod
    def _optimizer_parameters(optimizer: Any) -> list[nn.Parameter]:
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        return [parameter for group in raw_optimizer.param_groups for parameter in group["params"]]

    @staticmethod
    def _capture_gradients(parameters: list[nn.Parameter]) -> list[torch.Tensor | None]:
        return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]

    def _projected_optimization_step(self, output: ObjectiveOutput, batch_idx: int) -> dict[str, float]:
        optimizer = self.optimizers()
        parameters = self._optimizer_parameters(optimizer)
        accumulated = self._capture_gradients(parameters)
        optimizer.zero_grad()

        primary_loss = self.adjust_loss_for_accumulation(output.primary_loss)
        self.manual_backward(primary_loss, retain_graph=output.memory_loss is not None)
        primary_gradients = self._capture_gradients(parameters)
        optimizer.zero_grad()

        if output.memory_loss is None:
            memory_gradients: list[torch.Tensor | None] = [None for _ in parameters]
        else:
            memory_loss = self.adjust_loss_for_accumulation(output.memory_loss)
            self.manual_backward(memory_loss)
            memory_gradients = self._capture_gradients(parameters)
            optimizer.zero_grad()

        projected_memory, stats = project_conflicting_gradients(
            primary_gradients,
            memory_gradients,
            eps=self.config.gradient_projection.eps,
        )
        memory_weight = float(self.config.gradient_projection.memory_weight)
        for parameter, previous, primary, memory in zip(
            parameters,
            accumulated,
            primary_gradients,
            projected_memory,
        ):
            combined = previous
            if primary is not None:
                combined = primary if combined is None else combined + primary
            if memory is not None and memory_weight > 0.0:
                weighted_memory = memory * memory_weight
                combined = weighted_memory if combined is None else combined + weighted_memory
            parameter.grad = combined

        if self.should_step_optimizer(batch_idx):
            optimizer.step()
            optimizer.zero_grad()
            self.objective.kernel.clamp_temperature_()
        return stats.metrics()

    def _log_training_metrics(self, metrics: dict[str, Any]) -> None:
        resolved = resolve_metric_tensors(metrics)
        numeric_metrics = {key: value for key, value in resolved.items() if isinstance(value, (int, float))}
        if self.metrics_logger is not None:
            self.metrics_logger.log_metrics(
                {
                    "step": int(self.global_step),
                    "epoch": int(self.current_epoch),
                    "method": self.config.method.value,
                    **numeric_metrics,
                }
            )
        self.log_dict(numeric_metrics, on_step=True, on_epoch=False, prog_bar=False)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self.compute_training_step(batch, step=int(self.global_step))
        metrics: dict[str, Any] = dict(output.metrics)
        if self.config.gradient_projection.enabled:
            metrics.update(self._projected_optimization_step(output, batch_idx))
        self._log_training_metrics(metrics)
        return output.loss.detach() if not self.automatic_optimization else output.loss

    def on_train_end(self) -> None:
        if self.metrics_logger is not None:
            self.metrics_logger.close_log()
        if self.memory_bank is not None:
            self.memory_bank.close()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        seen: set[int] = set()

        def unique(parameters):
            selected = []
            for parameter in parameters:
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    selected.append(parameter)
            return selected

        groups = []
        encoder_parameters = unique(self.encoder.parameters())
        if encoder_parameters:
            groups.append(
                {
                    "params": encoder_parameters,
                    "lr": self.config.optimization.lr_encoder,
                    "weight_decay": self.config.optimization.weight_decay,
                }
            )
        head_modules = [module for module in (self.alpha_gate, self.margin_head) if module is not None]
        head_parameters = unique(parameter for module in head_modules for parameter in module.parameters())
        if head_parameters:
            groups.append(
                {
                    "params": head_parameters,
                    "lr": self.config.optimization.lr_heads,
                    "weight_decay": 0.0,
                }
            )
        objective_parameters = unique(self.objective.parameters())
        if objective_parameters:
            groups.append(
                {
                    "params": objective_parameters,
                    "lr": self.config.optimization.lr_encoder,
                    "weight_decay": 0.0,
                }
            )
        return torch.optim.AdamW(groups)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        self.objective.kernel.clamp_temperature_()

    def save_deployable_encoder(self, destination: Path) -> Path:
        if self.config.model.lora.enabled:
            peft_model = self._lora_model()
            self.encoder.to("cpu")
            self.encoder.model.model = peft_model.merge_and_unload(safe_merge=True)
        destination.mkdir(parents=True, exist_ok=True)
        self.encoder.save(str(destination))
        return destination

    def _lora_model(self):
        from peft import PeftModel

        model = getattr(getattr(self.encoder, "model", None), "model", None)
        if not isinstance(model, PeftModel):
            raise RuntimeError("LoRA is enabled, but the encoder does not contain a PEFT model")
        return model

    def save_lora_adapter(self, destination: Path) -> Path | None:
        if not self.config.model.lora.enabled:
            return None
        destination.mkdir(parents=True, exist_ok=True)
        self._lora_model().save_pretrained(str(destination), safe_serialization=True)
        return destination

    def save_research_checkpoint(self, destination: Path) -> Path | None:
        if not self.config.artifacts.save_research_checkpoint:
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            optimizers = list(self.trainer.optimizers)
        except RuntimeError:
            optimizers = []
        torch.save(
            {
                "schema_version": 1,
                "resolved_config": train_config_to_dict(self.config),
                "state_dict": self.state_dict(),
                "optimizer_states": [optimizer.state_dict() for optimizer in optimizers],
                "epoch": int(self.current_epoch),
                "global_step": int(self.global_step),
            },
            temporary,
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load_research_checkpoint(
        cls,
        path: Path,
        *,
        encoder: nn.Module,
        lexical_lookup: dict[str, str | list[str]] | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple["ContrastiveTrainingModule", list[dict[str, object]]]:
        payload = torch.load(path, map_location=map_location)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported research checkpoint schema: {payload.get('schema_version')!r}")
        config = parse_train_config(payload["resolved_config"])
        module = cls.build(encoder, config, lexical_lookup=lexical_lookup)
        module.load_state_dict(payload["state_dict"], strict=True)
        return module, list(payload.get("optimizer_states", []))
