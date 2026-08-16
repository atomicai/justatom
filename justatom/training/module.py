from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytorch_lightning as L
import torch
from torch import nn

from justatom.logging.io import CSVLogger
from justatom.training.alpha_gate import QueryAlphaGate
from justatom.training.config import (
    MarginMode,
    TrainConfig,
    parse_train_config,
    train_config_to_dict,
)
from justatom.training.gradient_projection import project_conflicting_gradients
from justatom.training.memory_bank import ContrastiveMemoryBank, QueryMarginHead
from justatom.training.methods import resolve_method
from justatom.training.objective import (
    ContrastiveObjective,
    ObjectiveInputs,
    ObjectiveOutput,
)
from justatom.training.telemetry import (
    batch_retrieval_metrics,
    resolve_metric_tensors,
    retrieval_metrics_by_confidence,
    scalar_distribution,
)


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
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.objective = objective
        self.alpha_gate = alpha_gate
        self.memory_bank = memory_bank
        self.margin_head = margin_head
        self.metrics_path = None if config.telemetry.metrics_path is None else Path(config.telemetry.metrics_path)
        self.metrics_logger = CSVLogger(self.metrics_path) if self.metrics_path is not None else None
        self.automatic_optimization = not config.gradient_projection.enabled

    @classmethod
    def build(
        cls,
        encoder: nn.Module,
        config: TrainConfig,
    ) -> ContrastiveTrainingModule:
        config = resolve_method(config)
        embedding_dim = int(encoder.output_dims)
        alpha_gate = QueryAlphaGate(embedding_dim, config.alpha_gate) if config.alpha_gate.enabled else None
        memory_bank = ContrastiveMemoryBank(config.memory_bank) if config.memory_bank.enabled else None
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
        )

    @property
    def needs_simcse(self) -> bool:
        return self.config.objective.simcse_dropout_weight > 0.0

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

    def _margin_values(self, queries: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        margin_config = self.config.memory_bank.margin
        if self.margin_head is not None:
            return self.margin_head(queries)
        if self.memory_bank is not None and margin_config.mode is MarginMode.CONSTANT:
            margin = queries.new_full((queries.shape[0],), margin_config.base)
            return margin, margin
        return None, None

    def compute_training_step(self, batch: dict[str, Any], *, step: int) -> ObjectiveOutput:
        queries, positives = self.encoder.encode_pair(batch)
        alpha_logits = None if self.alpha_gate is None else self.alpha_gate.logits(queries.detach())
        alpha = None if alpha_logits is None else torch.sigmoid(alpha_logits)
        query_alt = self._encode_dropout_query_view(batch) if self.needs_simcse else None
        selection = (
            None
            if self.memory_bank is None
            else self.memory_bank.select(
                batch=batch,
                query_vectors=queries,
                positive_vectors=positives,
                step=step,
            )
        )
        raw_margin, margin = self._margin_values(queries)

        output = self.objective(
            ObjectiveInputs(
                queries=queries,
                positives=positives,
                query_alt=query_alt,
                alpha_logits=alpha_logits,
                memory=selection,
                raw_margin=raw_margin,
                margin=margin,
                alpha_supervision_weight=self.config.alpha_gate.supervision_weight,
                alpha_target_temperature=self.config.alpha_gate.target_temperature,
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
                metrics.update(scalar_distribution("alpha_aux_weight", 1.0 - alpha.detach()))
            if output.alpha_target is not None:
                metrics.update(scalar_distribution("alpha_target", output.alpha_target))
                metrics["alpha/absolute_error_mean"] = float((alpha.detach() - output.alpha_target).abs().mean().item())
                metrics.update(retrieval_metrics_by_confidence(queries @ positives.T, output.alpha_target))
            if raw_margin is not None:
                metrics.update(scalar_distribution("margin/raw", raw_margin))
            if margin is not None:
                metrics.update(scalar_distribution("margin/bounded", margin))
            if selection is not None:
                metrics.update(selection.metrics)
        output = replace(output, metrics=metrics)
        if self.memory_bank is not None:
            self.memory_bank.enqueue(positives, batch)
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
            raise TypeError("LoRA is enabled, but the encoder does not contain a PEFT model")
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
                "schema_version": 3,
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
        map_location: str | torch.device = "cpu",
    ) -> tuple[ContrastiveTrainingModule, list[dict[str, object]]]:
        payload = torch.load(path, map_location=map_location)
        schema_version = payload.get("schema_version")
        if schema_version not in {1, 2, 3}:
            raise ValueError(f"Unsupported research checkpoint schema: {payload.get('schema_version')!r}")
        resolved_config = dict(payload["resolved_config"])
        if schema_version == 1:
            objective_config = dict(resolved_config.get("objective", {}))
            objective_config.pop("pairwise_margin", None)
            resolved_config["objective"] = objective_config

            alpha_gate_config = dict(resolved_config.get("alpha_gate", {}))
            alpha_gate_config["supervision_weight"] = alpha_gate_config.pop("mix_weight", 0.3)
            alpha_gate_config.pop("mix_weight_warmup_steps", None)
            alpha_gate_config.pop("entropy_weight", None)
            resolved_config["alpha_gate"] = alpha_gate_config

            experiment_config = dict(resolved_config.get("experiment", {}))
            if resolved_config.get("method") == "atom_gate" or objective_config.get("decoupled") is True:
                experiment_config["role"] = "ablation"
            resolved_config["experiment"] = experiment_config

        if schema_version in {1, 2}:
            memory_bank_config = resolved_config.get("memory_bank")
            method_uses_bank = resolved_config.get("method") == "atomic"
            if isinstance(memory_bank_config, Mapping):
                bank_enabled = bool(memory_bank_config.get("enabled", method_uses_bank))
                has_normalization_contract = {
                    "mass_ratio",
                    "mass_ramp_steps",
                }.issubset(memory_bank_config)
            else:
                bank_enabled = method_uses_bank
                has_normalization_contract = False
            if bank_enabled and not has_normalization_contract:
                # Old bank payloads remain loadable, but current defaults cannot make them canonical.
                experiment_config = dict(resolved_config.get("experiment", {}))
                experiment_config["role"] = "ablation"
                resolved_config["experiment"] = experiment_config
        config = parse_train_config(resolved_config)
        module = cls.build(encoder, config)
        module.load_state_dict(payload["state_dict"], strict=True)
        return module, list(payload.get("optimizer_states", []))
