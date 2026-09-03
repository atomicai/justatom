import numpy as np
import pytest

from justatom.running.qrels import exact_single_positive_ranks, single_positive_metrics


def test_exact_single_positive_ranks_are_block_invariant_and_ties_follow_corpus_order():
    corpus = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    queries = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    positives = np.asarray([3, 1, 0])

    one_block = exact_single_positive_ranks(queries, corpus, positives, corpus_block_size=4)
    many_blocks = exact_single_positive_ranks(
        queries,
        corpus,
        positives,
        query_batch_size=1,
        corpus_block_size=1,
    )

    assert one_block.tie_policy == "corpus_order"
    assert one_block.ranks.tolist() == [2, 1, 3]
    assert many_blocks.ranks.tolist() == one_block.ranks.tolist()


def test_single_positive_metrics_use_cutoff_for_mrr_and_ndcg():
    metrics = single_positive_metrics(np.asarray([1, 2, 11]))

    assert metrics["queries"] == 3
    assert metrics["hit_at_1"] == pytest.approx(1 / 3)
    assert metrics["recall_at_5"] == pytest.approx(2 / 3)
    assert metrics["recall_at_10"] == pytest.approx(2 / 3)
    assert metrics["mrr_at_10"] == pytest.approx(0.5)
    assert metrics["ndcg_at_10"] == pytest.approx((1.0 + 1.0 / np.log2(3.0)) / 3.0)
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5 + 1.0 / 11.0) / 3.0)
    assert metrics["mean_rank"] == pytest.approx(14 / 3)
    assert metrics["median_rank"] == 2.0


@pytest.mark.parametrize(
    ("queries", "corpus", "positives", "message"),
    [
        (np.ones((2, 3)), np.ones((2, 4)), [0, 1], "dimensions"),
        (np.ones((2, 3)), np.ones((2, 3)), [0], "shape"),
        (np.ones((2, 3)), np.ones((2, 3)), [0, 2], "outside"),
    ],
)
def test_exact_single_positive_ranks_reject_invalid_contracts(queries, corpus, positives, message):
    with pytest.raises(ValueError, match=message):
        exact_single_positive_ranks(queries, corpus, positives)
