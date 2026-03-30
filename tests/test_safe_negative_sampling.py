from __future__ import annotations

import torch

from justatom.processing.prime import TrainWithContrastiveProcessor
from justatom.running.trainer import _sample_safe_negative_indices


def test_stable_key_id_is_deterministic():
    first = TrainWithContrastiveProcessor._stable_key_id("same-value")
    second = TrainWithContrastiveProcessor._stable_key_id("same-value")
    other = TrainWithContrastiveProcessor._stable_key_id("other-value")

    assert first == second
    assert first != other


def test_safe_negative_sampling_avoids_same_doc_content_and_query_when_possible():
    doc_key_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    content_key_ids = torch.tensor([10, 10, 30, 40], dtype=torch.long)
    query_key_ids = torch.tensor([100, 200, 100, 400], dtype=torch.long)
    negative_indices, fallback_count = _sample_safe_negative_indices(
        doc_key_ids=doc_key_ids,
        content_key_ids=content_key_ids,
        query_key_ids=query_key_ids,
    )

    for idx, neg_idx in enumerate(negative_indices.tolist()):
        assert neg_idx != idx
        assert doc_key_ids[neg_idx].item() != doc_key_ids[idx].item()
        assert content_key_ids[neg_idx].item() != content_key_ids[idx].item()
        assert query_key_ids[neg_idx].item() != query_key_ids[idx].item()

    assert fallback_count == 0


def test_safe_negative_sampling_reports_fallback_when_batch_is_all_duplicates():
    negative_indices, fallback_count = _sample_safe_negative_indices(
        doc_key_ids=torch.tensor([1, 1, 1], dtype=torch.long),
        content_key_ids=torch.tensor([9, 9, 9], dtype=torch.long),
        query_key_ids=torch.tensor([7, 7, 7], dtype=torch.long),
    )

    assert fallback_count == 3
    assert all(idx != neg_idx for idx, neg_idx in enumerate(negative_indices.tolist()))


def test_safe_negative_sampling_filters_too_close_candidates_by_inverse_idf_recall_threshold():
    score_by_query_doc = {
        ("q0", "close"): 0.9,
        ("q0", "medium"): 0.7,
        ("q0", "far"): 0.1,
        ("q1", "positive-0"): 0.9,
        ("q1", "medium"): 0.8,
        ("q1", "far"): 0.2,
        ("q2", "positive-0"): 0.4,
        ("q2", "close"): 0.9,
        ("q2", "far"): 0.7,
        ("q3", "positive-0"): 0.8,
        ("q3", "close"): 0.3,
        ("q3", "medium"): 0.9,
    }

    negative_indices, fallback_count = _sample_safe_negative_indices(
        doc_key_ids=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        content_key_ids=torch.tensor([10, 20, 30, 40], dtype=torch.long),
        query_key_ids=torch.tensor([100, 200, 300, 400], dtype=torch.long),
        queries=["q0", "q1", "q2", "q3"],
        docs=["positive-0", "close", "medium", "far"],
        inverse_idf_recall_fn=lambda query, doc_text: score_by_query_doc.get((str(query), str(doc_text)), 1.0),
        max_negative_inverse_idf_recall=0.5,
    )

    assert fallback_count == 0
    assert negative_indices.tolist() == [3, 3, 0, 1]
