import json
import tempfile
from pathlib import Path

from justatom.api import train as train_api


def _write_jsonl(rows: list[dict]) -> Path:
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="train_prepare_")
    Path(path).unlink(missing_ok=True)
    out = Path(path)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def test_prepare_training_data_prefers_polars_batches_for_frame_backed_sources(
    monkeypatch,
):
    rows = [
        {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
        {"chunk_id": "b", "content": "doc-b", "queries": ["q3"]},
        {"chunk_id": "c", "content": "doc-c", "queries": ["q4"]},
    ]
    path = _write_jsonl(rows)

    def fail(*args, **kwargs):
        raise AssertionError("frame-backed datasets should avoid from_source fallback")

    monkeypatch.setattr(train_api.DatasetRecordAdapter, "from_source", fail)

    try:
        pl_data, js_data, lexical_by_content = train_api.prepare_training_data(
            dataset_name_or_path=path,
            num_samples=2,
            content_field="content",
            labels_field="queries",
            chunk_id_col="chunk_id",
        )
    finally:
        path.unlink(missing_ok=True)

    assert pl_data.height == 2
    assert len(js_data) == 2
    assert set(lexical_by_content).issubset({"doc-a", "doc-b", "doc-c"})


def test_prepare_training_data_falls_back_to_lazy_adapter_for_iterable_sources(
    monkeypatch,
):
    rows = [
        {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
        {"chunk_id": "b", "content": "doc-b", "queries": ["q3"]},
        {"chunk_id": "c", "content": "doc-c", "queries": ["q4"]},
    ]
    captured: dict[str, object] = {}

    class _FakeDataset:
        def iterator(self, **kwargs):
            return iter(rows)

    original_from_source = train_api.DatasetRecordAdapter.from_source

    def fake_named(*args, **kwargs):
        return _FakeDataset()

    def wrapped(*args, **kwargs):
        captured["lazy"] = kwargs.get("lazy")
        return original_from_source(*args, **kwargs)

    monkeypatch.setattr(train_api.DatasetApi, "named", fake_named)
    monkeypatch.setattr(train_api.DatasetRecordAdapter, "from_source", wrapped)

    pl_data, js_data, lexical_by_content = train_api.prepare_training_data(
        dataset_name_or_path="hf://dummy/dataset",
        num_samples=2,
        content_field="content",
        labels_field="queries",
        chunk_id_col="chunk_id",
    )

    assert captured["lazy"] is True
    assert pl_data.height == 2
    assert len(js_data) == 2
    assert set(lexical_by_content).issubset({"doc-a", "doc-b", "doc-c"})


def test_prepare_training_data_streams_and_bounds_sample_size():
    rows = [
        {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
        {"chunk_id": "b", "content": "doc-b", "queries": ["q3", "q4"]},
        {"chunk_id": "c", "content": "doc-c", "queries": ["q5", "q6"]},
    ]
    path = _write_jsonl(rows)

    try:
        pl_data, js_data, lexical_by_content = train_api.prepare_training_data(
            dataset_name_or_path=path,
            num_samples=3,
            content_field="content",
            labels_field="queries",
            chunk_id_col="chunk_id",
        )
    finally:
        path.unlink(missing_ok=True)

    assert pl_data.height == 3
    assert len(js_data) == 3
    assert all(row["content"] in lexical_by_content for row in js_data)
    assert all(row["lexical_text"] == lexical_by_content[row["content"]] for row in js_data)


def test_prepare_training_data_preserves_chunk_identity_for_duplicate_content():
    rows = [
        {"chunk_id": "a", "content": "same-doc", "queries": ["q1"]},
        {"chunk_id": "b", "content": "same-doc", "queries": ["q2"]},
    ]
    path = _write_jsonl(rows)

    try:
        _, js_data, lexical_by_content = train_api.prepare_training_data(
            dataset_name_or_path=path,
            num_samples=2,
            content_field="content",
            labels_field="queries",
            chunk_id_col="chunk_id",
        )
    finally:
        path.unlink(missing_ok=True)

    assert [row["chunk_id"] for row in js_data] == ["a", "b"]
    assert lexical_by_content == {"same-doc": "same-doc"}


def test_rebalance_rows_by_content_reduces_within_batch_duplicates_when_possible():
    rows = [
        {"chunk_id": "a1", "content": "same-doc", "queries": "q1", "lexical_text": "same-doc"},
        {"chunk_id": "a2", "content": "same-doc", "queries": "q2", "lexical_text": "same-doc"},
        {"chunk_id": "b1", "content": "doc-b", "queries": "q3", "lexical_text": "doc-b"},
        {"chunk_id": "c1", "content": "doc-c", "queries": "q4", "lexical_text": "doc-c"},
    ]

    before = train_api.count_batches_with_duplicate_content(rows, batch_size=3)
    rebalanced = train_api.rebalance_rows_by_content(rows, batch_size=3)
    after = train_api.count_batches_with_duplicate_content(rebalanced, batch_size=3)

    assert before == 1
    assert after == 0
    assert sorted(row["chunk_id"] for row in rebalanced) == ["a1", "a2", "b1", "c1"]


def test_rebalance_rows_by_content_keeps_all_rows_when_duplicates_are_unavoidable():
    rows = [
        {"chunk_id": "a1", "content": "same-doc", "queries": "q1", "lexical_text": "same-doc"},
        {"chunk_id": "a2", "content": "same-doc", "queries": "q2", "lexical_text": "same-doc"},
        {"chunk_id": "a3", "content": "same-doc", "queries": "q3", "lexical_text": "same-doc"},
        {"chunk_id": "b1", "content": "doc-b", "queries": "q4", "lexical_text": "doc-b"},
    ]

    rebalanced = train_api.rebalance_rows_by_content(rows, batch_size=3)

    assert sorted(row["chunk_id"] for row in rebalanced) == ["a1", "a2", "a3", "b1"]
    assert train_api.count_batches_with_duplicate_content(rebalanced, batch_size=3) == 1


def test_iterate_training_rows_applies_limit_after_query_expansion():
    rows = [
        {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
        {"chunk_id": "b", "content": "doc-b", "queries": ["q3"]},
    ]
    path = _write_jsonl(rows)

    try:
        sample_rows = list(
            train_api.iterate_training_rows(
                dataset_name_or_path=path,
                content_field="content",
                labels_field="queries",
                chunk_id_col="chunk_id",
                limit=2,
            )
        )
    finally:
        path.unlink(missing_ok=True)

    assert len(sample_rows) == 2
    assert [row["queries"] for row in sample_rows] == ["q1", "q2"]


def test_iterate_training_rows_handles_extra_source_fields_for_iterable_sources(
    monkeypatch,
):
    rows = [
        {"passage": "doc-a", "question": "q1"},
        {"passage": "doc-b", "question": "q2"},
    ]

    class _FakeDataset:
        def iterator(self, **kwargs):
            return iter(rows)

    def fake_named(*args, **kwargs):
        return _FakeDataset()

    monkeypatch.setattr(train_api.DatasetApi, "named", fake_named)

    sample_rows = list(
        train_api.iterate_training_rows(
            dataset_name_or_path="hf://dummy/boolq-like",
            content_field="passage",
            labels_field="question",
            limit=2,
        )
    )

    assert sample_rows == [
        {"content": "doc-a", "queries": "q1", "lexical_text": "doc-a"},
        {"content": "doc-b", "queries": "q2", "lexical_text": "doc-b"},
    ]


def test_create_training_job_selects_gamma_only_mode():
    job = train_api.create_training_job(
        dataset_name_or_path="dummy",
        freeze_encoder=True,
        gamma_joint=True,
        include_semantic_gamma=True,
        include_keywords_gamma=True,
        temperature=0.07,
        grad_acc_steps=4,
    )

    assert isinstance(job, train_api.GammaOnlyTrainingJob)
    assert job.training_mode == "gamma-only"
    assert job.gamma_joint is True
    assert job.loss == "contrastive"
    assert job.temperature == 0.07
    assert job.grad_acc_steps == 4


def test_create_training_job_selects_encoder_gamma_mode():
    job = train_api.create_training_job(
        dataset_name_or_path="dummy",
        freeze_encoder=False,
        include_semantic_gamma=True,
        include_keywords_gamma=True,
    )

    assert isinstance(job, train_api.EncoderGammaTrainingJob)
    assert job.training_mode == "encoder+gamma"


def test_create_training_job_selects_encoder_only_mode():
    job = train_api.create_training_job(
        dataset_name_or_path="dummy",
        freeze_encoder=False,
        include_semantic_gamma=False,
        include_keywords_gamma=False,
    )

    assert isinstance(job, train_api.EncoderOnlyTrainingJob)
    assert job.training_mode == "encoder-only"


def test_create_training_job_preserves_optimizer_and_legacy_style_params():
    job = train_api.create_training_job(
        dataset_name_or_path="dummy",
        freeze_encoder=False,
        include_semantic_gamma=True,
        include_keywords_gamma=True,
        loss="contrastive",
        optimizer="adafactor",
        grad_acc_steps=6,
        contrastive_temperature=0.1,
        soft_contrastive_temperature=10.0,
        memory_bank_size=1024,
        memory_bank_mining_mode="mixed",
        memory_bank_hard_negatives=32,
        memory_bank_random_negatives=16,
    )

    assert isinstance(job, train_api.EncoderGammaTrainingJob)
    assert job.optimizer == "adafactor"
    assert job.grad_acc_steps == 6
    assert job.contrastive_temperature == 0.1
    assert job.soft_contrastive_temperature == 10.0
    assert job.memory_bank_size == 1024
    assert job.memory_bank_mining_mode == "mixed"
    assert job.memory_bank_hard_negatives == 32
    assert job.memory_bank_random_negatives == 16


def test_resolve_train_kwargs_reads_optimizer_and_legacy_style_params():
    kwargs = train_api.resolve_train_kwargs(
        config={
            "dataset": {"name_or_path": "dummy"},
            "training": {
                "optimizer": "adafactor",
                "grad_acc_steps": 6,
                "contrastive_temperature": 0.1,
                "soft_contrastive_temperature": 10.0,
                "memory_bank_size": 2048,
                "memory_bank_mining_mode": "mixed",
                "memory_bank_hard_negatives": 32,
                "memory_bank_random_negatives": 16,
                "memory_bank_too_hard_margin": 0.01,
            },
        }
    )

    assert kwargs["optimizer"] == "adafactor"
    assert kwargs["grad_acc_steps"] == 6
    assert kwargs["contrastive_temperature"] == 0.1
    assert kwargs["soft_contrastive_temperature"] == 10.0
    assert kwargs["memory_bank_size"] == 2048
    assert kwargs["memory_bank_mining_mode"] == "mixed"
    assert kwargs["memory_bank_hard_negatives"] == 32
    assert kwargs["memory_bank_random_negatives"] == 16
    assert kwargs["memory_bank_too_hard_margin"] == 0.01


def test_resolve_train_kwargs_maps_atom_gate_production_recipe():
    kwargs = train_api.resolve_train_kwargs(
        config={
            "recipe": "atom_gate",
            "dataset": {"name_or_path": "dummy"},
            "training": {
                "batch_size": 32,
                "grad_acc_steps": 1,
                "n_epochs": 2,
            },
            "atom_gate": {
                "temperature": 0.05,
            },
            "alpha_gate": {
                "mix_weight": 0.3,
                "auxiliary": {
                    "simcse_dropout_weight": 0.1,
                },
                "alpha_query": {
                    "layers": 1,
                    "hidden_dim": "auto",
                    "dropout": 0.0,
                    "activation": "gelu",
                },
            },
            "memory_bank": {
                "enabled": True,
                "size": 256,
                "warmup_steps": 30,
                "mining": "mixed",
                "hard_negatives": 4,
                "random_negatives": 4,
            },
        }
    )

    assert kwargs["recipe"] == "atom_gate"
    assert kwargs["add_alpha_gate"] is True
    assert kwargs["loss"] == "contrastive"
    assert kwargs["freeze_encoder"] is False
    assert kwargs["gamma_joint"] is True
    assert kwargs["include_semantic_gamma"] is True
    assert kwargs["include_keywords_gamma"] is True
    assert kwargs["alpha_train_only"] is True
    assert kwargs["alpha_mix_weight"] == 0.3
    assert kwargs["optimizer"] == "adamw"
    assert kwargs["contrastive_temperature"] == 0.05
    assert kwargs["contrastive_learnable_temperature"] is True
    assert kwargs["contrastive_decoupled"] is True
    assert kwargs["contrastive_simcse_dropout_weight"] == 0.1
    assert kwargs["contrastive_loss_alpha_gate"] is True
    assert kwargs["contrastive_loss_alpha_gate_mode"] == "augment"
    assert kwargs["alpha_head_input"] == "query"
    assert kwargs["alpha_head_layers"] == 1
    assert kwargs["alpha_head_hidden_dim"] is None
    assert kwargs["alpha_head_dropout"] == 0.0
    assert kwargs["alpha_head_activation"] == "gelu"
    assert kwargs["memory_bank_size"] == 256
    assert kwargs["memory_bank_warmup_steps"] == 30
    assert kwargs["memory_bank_mining_mode"] == "mixed"
    assert kwargs["memory_bank_hard_negatives"] == 4
    assert kwargs["memory_bank_random_negatives"] == 4


def test_atom_gate_recipe_has_canonical_defaults_without_separate_config():
    kwargs = train_api.resolve_train_kwargs(
        config={
            "recipe": "atom_gate",
            "dataset": {"id": "justatom"},
        }
    )

    assert kwargs["recipe"] == "atom_gate"
    assert kwargs["add_alpha_gate"] is True
    assert kwargs["contrastive_temperature"] == 0.05
    assert kwargs["contrastive_simcse_dropout_weight"] == 0.1
    assert kwargs["alpha_mix_weight"] == 0.3
    assert kwargs["alpha_head_layers"] == 1
    assert kwargs["alpha_head_hidden_dim"] is None
    assert kwargs["alpha_head_dropout"] == 0.0


def test_create_training_job_selects_atom_gate_recipe_mode():
    job = train_api.create_training_job(
        recipe="atom_gate",
        dataset_name_or_path="dummy",
        batch_size=32,
        grad_acc_steps=1,
        n_epochs=2,
    )

    assert isinstance(job, train_api.AtomGateTrainingJob)
    assert job.training_mode == "atom-gate"
    assert job.add_alpha_gate is True
    assert job.loss == "contrastive"
    assert job.freeze_encoder is False
    assert job.gamma_joint is True
    assert job.alpha_train_only is True
    assert job.contrastive_temperature == 0.05
    assert job.contrastive_simcse_dropout_weight == 0.1
    assert job.contrastive_loss_alpha_gate is True
    assert job.contrastive_loss_alpha_gate_mode == "augment"


def test_add_alpha_gate_feature_maps_to_legacy_knobs_and_head_config():
    kwargs = train_api.resolve_train_kwargs(
        config={
            "dataset": {"name_or_path": "dummy"},
            "add_alpha_gate": True,
            "alpha_gate": {
                "mode": "augment",
                "train_only": True,
                "mix_weight": 0.35,
                "auxiliary": {
                    "simcse_dropout_weight": 0.2,
                },
                "alpha_query": {
                    "input": "query_doc",
                    "layers": 2,
                    "hidden_dim": 96,
                    "dropout": 0.15,
                    "activation": "silu",
                },
            },
        }
    )

    assert kwargs["recipe"] is None
    assert kwargs["add_alpha_gate"] is True
    assert kwargs["gamma_joint"] is True
    assert kwargs["alpha_train_only"] is True
    assert kwargs["alpha_mix_weight"] == 0.35
    assert kwargs["contrastive_loss_alpha_gate"] is True
    assert kwargs["contrastive_loss_alpha_gate_mode"] == "augment"
    assert kwargs["contrastive_simcse_dropout_weight"] == 0.2
    assert kwargs["alpha_head_input"] == "query_doc"
    assert kwargs["alpha_head_include_doc"] is True
    assert kwargs["alpha_head_layers"] == 2
    assert kwargs["alpha_head_hidden_dim"] == 96
    assert kwargs["alpha_head_dropout"] == 0.15
    assert kwargs["alpha_head_activation"] == "silu"


def test_atom_gate_recipe_preserves_legacy_training_bank_overrides():
    kwargs = train_api.resolve_train_kwargs(
        config={
            "recipe": "atom_gate",
            "dataset": {"id": "justatom"},
            "training": {
                "contrastive_temperature": 0.07,
                "memory_bank_size": 256,
                "memory_bank_mining_mode": "mixed",
                "memory_bank_hard_negatives": 4,
                "memory_bank_random_negatives": 4,
            },
        }
    )

    assert kwargs["recipe"] == "atom_gate"
    assert kwargs["dataset_name_or_path"] == "justatom"
    assert kwargs["contrastive_temperature"] == 0.07
    assert kwargs["memory_bank_size"] == 256
    assert kwargs["memory_bank_mining_mode"] == "mixed"
    assert kwargs["memory_bank_hard_negatives"] == 4
    assert kwargs["memory_bank_random_negatives"] == 4


def test_create_training_job_allows_temperature_contrastive_for_gamma_modes():
    job = train_api.create_training_job(
        dataset_name_or_path="dummy",
        freeze_encoder=False,
        include_semantic_gamma=True,
        include_keywords_gamma=True,
        loss="contrastive",
    )

    assert isinstance(job, train_api.EncoderGammaTrainingJob)
    assert job.loss == "contrastive"


def test_create_training_job_allows_focal_contrastive_for_gamma_modes():
    job = train_api.create_training_job(
        dataset_name_or_path="dummy",
        freeze_encoder=False,
        include_semantic_gamma=True,
        include_keywords_gamma=True,
        loss="focal-contrastive",
        focal_gamma=3.0,
    )

    assert isinstance(job, train_api.EncoderGammaTrainingJob)
    assert job.loss == "focal-contrastive"
    assert job.focal_gamma == 3.0
