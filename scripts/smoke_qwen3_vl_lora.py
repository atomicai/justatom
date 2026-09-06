"""Offline real-checkpoint training smoke; not a retrieval-quality benchmark.

Run from the repository root in justatom-env:
    python scripts/smoke_qwen3_vl_lora.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

# Set offline mode before importing the HF stack. Never download in this smoke.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as L
import torch

from justatom.api.train import resolve_train_config
from justatom.processing.loader import NamedDataLoader
from justatom.processing.prime import TrainWithContrastiveProcessor
from justatom.processing.tokenizer import ITokenizer
from justatom.training.job import RunManifest, build_lightning_trainer, load_encoder, write_run_manifest
from justatom.training.module import ContrastiveTrainingModule


def smoke(mode: str, device: str, root: Path, steps: int) -> dict:
    config = resolve_train_config(config_path=f"configs/experiments/qwen3-vl-2b-lora-{mode}.yaml")
    directory = root / mode
    directory.mkdir()
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            id=None,
            name_or_path="synthetic:qwen3-vl-lora-smoke",
            labels_field="query",
            metadata={"source": "inline synthetic texts; no benchmark dataset loaded"},
        ),
        optimization=replace(config.optimization, grad_acc_steps=1, num_samples=steps * config.optimization.batch_size),
        anchor_bank=replace(config.anchor_bank, warmup_steps=1),
        runtime=replace(config.runtime, accelerator="gpu" if device == "cuda" else device),
        telemetry=replace(config.telemetry, metrics_path=str(directory / "batch_metrics.csv")),
    )
    L.seed_everything(config.experiment.seed, workers=True)
    write_run_manifest(directory, RunManifest.capture(config))
    tokenizer = ITokenizer.from_pretrained(config.model.name_or_path, revision=config.model.revision, local_files_only=True)
    processor = TrainWithContrastiveProcessor(
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        max_query_seq_len=config.model.max_query_seq_len,
        queries_prefix=config.model.query_prefix,
        pos_queries_prefix=config.model.content_prefix,
    )
    topics = ["инфляция", "нейронная сеть", "фотосинтез", "гравитация", "индекс", "компилятор", "облигация", "энтропия"]
    rows = [
        {
            "query": f"Объясни понятие {topics[i % len(topics)]}, пример {i}.",
            "content": f"Учебный текст {i}: {topics[i % len(topics)]}. Это отдельный документ для проверки обучения.",
        }
        for i in range(steps * config.optimization.batch_size)
    ]
    dataset, names, _ = processor.dataset_from_dicts(rows)
    loader = NamedDataLoader(dataset, tensor_names=names, batch_size=config.optimization.batch_size)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    encoder = load_encoder(config, processor)
    trainable = [(name, p) for name, p in encoder.model.named_parameters() if p.requires_grad]
    assert trainable and all("language_model." in name and "lora_" in name for name, _ in trainable)
    assert not any(p.requires_grad for p in encoder.model.model.visual.parameters())
    before = sum(p.detach().float().abs().sum().item() for name, p in trainable if "lora_B" in name)
    module = ContrastiveTrainingModule.build(encoder, config)
    trainer = build_lightning_trainer(config)
    trainer.fit(module, train_dataloaders=loader)
    if device == "cuda":
        torch.cuda.synchronize()
    after = sum(p.detach().float().abs().sum().item() for name, p in trainable if "lora_B" in name)
    assert after != before, "LoRA matrices did not change"
    assert all(torch.isfinite(p).all() for _, p in trainable)
    with open(directory / "batch_metrics.csv", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == steps
    if config.anchor_bank.enabled:
        assert any(float(row.get("anchor/active_rows", 0) or 0) > 0 for row in metrics)
        assert module.anchor_bank is not None
        assert not module.anchor_bank.queries.requires_grad
        assert not module.anchor_bank.documents.requires_grad
    result = {
        "status": "PASS",
        "mode": mode,
        "device": device,
        "steps": int(trainer.global_step),
        "microbatch": config.optimization.batch_size,
        "max_query_seq_len": config.model.max_query_seq_len,
        "max_seq_len": config.model.max_seq_len,
        "seconds_including_load": round(time.monotonic() - started, 3),
        "trainable_encoder_parameters": sum(p.numel() for _, p in trainable),
        "lora_B_abs_sum_before": before,
        "lora_B_abs_sum_after": after,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if device == "cuda" else None,
        "last_metrics": metrics[-1],
        "note": "Synthetic wiring smoke; accumulation=1, anchor warmup=1. No quality claim or saved weights.",
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("--steps must be >= 3 to exercise the anchor constraint after an adapter update")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(4)
    base = Path(".tmp_runs")
    base.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="qwen3-vl-lora-smoke-", dir=base))
    print(f"Environment: {sys.executable}; artifacts: {root}", flush=True)
    for mode in ("vanilla", "geometry-anchor-bank"):
        smoke(mode, args.device, root, args.steps)
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
