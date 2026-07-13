from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from justatom.tooling.ir_dataset.dense import DenseIndex


class FakeEncoder:
    dimension = 3
    model_name = "test/fake"
    device = "cpu"

    def encode(self, texts, batch_size):
        vectors = {
            "alpha": [3.0, 0.0, 0.0],
            "beta": [0.0, 2.0, 0.0],
            "alpha beta": [1.0, 1.0, 0.0],
            "query: alpha": [1.0, 0.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


class TieEncoder:
    dimension = 2
    model_name = "test/ties"
    device = "cpu"

    def encode(self, texts, batch_size):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def sample_rows() -> list[tuple[str, str]]:
    return [("p1", "alpha"), ("p2", "beta"), ("p3", "alpha beta")]


def dense_fixture(tmp_path) -> DenseIndex:
    return DenseIndex.build(
        rows=sample_rows(),
        output_dir=tmp_path / "dense",
        encoder=FakeEncoder(),
        batch_size=2,
    )


def test_dense_build_normalizes_and_persists_embeddings(tmp_path):
    index = dense_fixture(tmp_path)

    matrix = index.embedding_rows([0, 1, 2])

    assert matrix.shape == (3, 3)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert index.embeddings_path.exists()


def test_blockwise_topk_matches_exact_result(tmp_path):
    index = dense_fixture(tmp_path)

    hits = index.search_embeddings(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        k=2,
        block_size=2,
        device="cpu",
    )[0]

    assert [hit.passage_id for hit in hits] == ["p1", "p3"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score > hits[1].score


def test_excluding_self_still_returns_k_neighbors(tmp_path):
    index = dense_fixture(tmp_path)

    hits = index.search_embeddings(
        index.embedding_rows([0]),
        k=2,
        block_size=2,
        exclude_ids=["p1"],
        device="cpu",
    )[0]

    assert len(hits) == 2
    assert all(hit.passage_id != "p1" for hit in hits)


def test_dense_reload_and_query_prefix_preserve_results(tmp_path):
    built = dense_fixture(tmp_path)
    expected = built.search_texts(["alpha"], k=2, device="cpu")

    loaded = DenseIndex.load(tmp_path / "dense", encoder=FakeEncoder())

    assert loaded.search_texts(["alpha"], k=2, device="cpu") == expected


def test_dense_rejects_duplicate_ids(tmp_path):
    rows = [("p1", "alpha"), ("p1", "beta")]

    try:
        DenseIndex.build(rows, tmp_path / "dense", FakeEncoder())
    except ValueError as exc:
        assert "duplicate passage_id" in str(exc)
    else:
        raise AssertionError("duplicate IDs must be rejected")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_search_does_not_wrap_read_only_memmap(tmp_path):
    index = dense_fixture(tmp_path)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        index.search_embeddings(
            index.embedding_rows([0]),
            k=2,
            block_size=2,
            exclude_ids=["p1"],
            device="mps",
        )

    assert not any("not writable" in str(item.message) for item in recorded)


def test_dense_topk_ties_are_selected_by_passage_id_before_truncation(tmp_path):
    index = DenseIndex.build(
        rows=[("a", "one"), ("b", "two"), ("c", "three"), ("d", "four")],
        output_dir=tmp_path / "dense",
        encoder=TieEncoder(),
        batch_size=2,
    )

    hits = index.search_embeddings(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        k=2,
        block_size=4,
        device="cpu",
    )[0]

    assert [hit.passage_id for hit in hits] == ["a", "b"]


def test_dense_load_rejects_corrupted_embeddings(tmp_path):
    index = dense_fixture(tmp_path)
    with index.embeddings_path.open("r+b") as stream:
        stream.seek(0)
        stream.write(b"broken")

    with pytest.raises(ValueError, match="checksum"):
        DenseIndex.load(tmp_path / "dense")
