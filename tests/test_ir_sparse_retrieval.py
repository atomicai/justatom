from __future__ import annotations

from justatom.tooling.ir_dataset.sparse import BM25Index


def sample_rows() -> list[tuple[str, str]]:
    return [
        (
            "p1",
            "passage: Rootless Docker\nПорты\n\n"
            "Rootless Docker не может открыть непривилегированный порт без настройки системного параметра.",
        ),
        (
            "p2",
            "passage: PostgreSQL\nРепликация\n\n" "Журнал WAL используется для потоковой репликации PostgreSQL между узлами.",
        ),
        (
            "p3",
            "passage: Контейнеры\nОсновы\n\n" "Контейнеризация изолирует приложение и его зависимости от основной системы.",
        ),
    ]


def test_bm25_ranks_rare_technical_terms_first(tmp_path):
    index = BM25Index.build(sample_rows(), tmp_path / "bm25")

    hits = index.search(["почему rootless docker не открывает порт"], k=2)[0]

    assert hits[0].passage_id == "p1"
    assert hits[0].score >= hits[1].score
    assert [hit.rank for hit in hits] == [1, 2]


def test_bm25_mmap_reload_preserves_results(tmp_path):
    built = BM25Index.build(sample_rows(), tmp_path / "bm25")
    expected = built.search(["postgresql wal"], k=3)

    loaded = BM25Index.load(tmp_path / "bm25", mmap=True)

    assert loaded.search(["postgresql wal"], k=3) == expected


def test_bm25_rejects_duplicate_ids(tmp_path):
    rows = [sample_rows()[0], sample_rows()[0]]

    try:
        BM25Index.build(rows, tmp_path / "bm25")
    except ValueError as exc:
        assert "duplicate passage_id" in str(exc)
    else:
        raise AssertionError("duplicate IDs must be rejected")


def test_bm25_k_is_capped_by_corpus_size(tmp_path):
    index = BM25Index.build(sample_rows(), tmp_path / "bm25")

    hits = index.search(["несуществующий термин"], k=20)[0]

    assert len(hits) == 3
    assert {hit.passage_id for hit in hits} == {"p1", "p2", "p3"}


def test_bm25_does_not_attach_sentence_punctuation_to_terms(tmp_path):
    index = BM25Index.build(
        [
            ("p-other", "Контейнер запускается через podman."),
            ("p-docker", "Контейнер запускается через docker."),
        ],
        tmp_path / "bm25",
    )

    hits = index.search(["docker"], k=2)[0]

    assert hits[0].passage_id == "p-docker"
    assert hits[0].score > 0.0
