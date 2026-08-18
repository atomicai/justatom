from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml

from justatom.tooling.ir_dataset.artifacts import sha256_file
from justatom.tooling.ir_dataset.release import CORPUS_COLUMNS, PAIR_SCHEMA, QREL_SCHEMA, RELEASE_ARTIFACT_PATHS
from research.habr_ir.release import PUBLIC_SCHEMA, combine_releases, load_release_config


def _pair(*, suffix: str, passage_id: str, article_id: str, split: str) -> dict[str, object]:
    return {
        "pair_id": f"pair-{suffix}",
        "query_id": f"q-{suffix}",
        "article_id": article_id,
        "positive_passage_id": passage_id,
        "split": split,
        "query": f"Как работает механизм {suffix}?",
        "answer": f"Ответ {suffix}",
        "evidence": f"Текст {suffix}",
        "positive_passage": f"Заголовок {suffix}\n\nТекст {suffix}",
        "requested_intent": "how_to",
        "actual_intent": "how_to",
        "title": f"Заголовок {suffix}",
        "section": "Раздел",
        "url": f"https://example.test/{article_id}",
        "topic_flows": ["develop"],
        "topic_hubs": ["python"],
        "tags": ["test"],
        "generator_model": "test-model",
        "generator_prompt_hash": f"prompt-{suffix}",
        "generation_attempt": 1,
        "generation_custom_id": f"request-{suffix}",
        "generation_batch_id": f"batch-{suffix}",
    }


def _corpus() -> pl.DataFrame:
    rows = []
    for index, suffix in enumerate(("a", "b"), start=1):
        rows.append(
            {
                "passage_id": f"p-{suffix}",
                "article_id": f"article-{suffix}",
                "title": f"Заголовок {suffix}",
                "section": "Раздел",
                "content": f"Текст {suffix}",
                "serialized_passage": f"Заголовок {suffix}\n\nТекст {suffix}",
                "url": f"https://example.test/article-{suffix}",
                "flows": ["develop"],
                "hubs": ["python"],
                "tags": ["test"],
                "char_count": 7,
                "token_count": 2,
                "corpus_rank": index,
                "source_hash": f"source-{suffix}",
            }
        )
    return pl.DataFrame(rows).select(CORPUS_COLUMNS)


def _write_release(root: Path, *, suffix: str, passage_id: str, article_id: str, split: str) -> None:
    pair = pl.DataFrame([_pair(suffix=suffix, passage_id=passage_id, article_id=article_id, split=split)], schema=PAIR_SCHEMA)
    corpus = _corpus().with_columns((pl.col("passage_id") == passage_id).alias("is_positive"))
    qrel = pl.DataFrame(
        [{"query_id": f"q-{suffix}", "passage_id": passage_id, "relevance": 1}],
        schema=QREL_SCHEMA,
    )
    for current_split in ("train", "validation", "test"):
        pair_path = root / "data/pairs" / f"{current_split}.parquet"
        qrel_path = root / "data/qrels" / f"{current_split}.parquet"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        qrel_path.parent.mkdir(parents=True, exist_ok=True)
        pair.filter(pl.col("split") == current_split).write_parquet(pair_path)
        qrel.filter(pl.lit(current_split == split)).write_parquet(qrel_path)
    corpus_path = root / "data/corpus-100k/corpus.parquet"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_parquet(corpus_path)
    audit_path = root / "audit/pilot-review.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "generation_custom_id": [f"request-{suffix}"],
            "automatic_status": ["accepted"],
            "query": [f"Как работает механизм {suffix}?"],
        }
    ).write_csv(audit_path, include_bom=True)
    (root / "README.md").write_text("source release\n", encoding="utf-8")

    artifacts = []
    for relative_path in sorted(RELEASE_ARTIFACT_PATHS):
        path = root / relative_path
        artifacts.append({"path": relative_path, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {
        "version": 1,
        "fingerprint": f"fingerprint-{suffix}",
        "source": {
            "corpus_fingerprint": "corpus-fingerprint",
            "passages_sha256": "source-passages-sha256",
            "chunker_version": 4,
        },
        "generation": {"model": "test-model"},
        "git": {"sha": f"git-{suffix}", "dirty": False},
        "counts": {
            "pairs": 1,
            "pair_splits": {name: int(name == split) for name in ("train", "validation", "test")},
            "corpus": 2,
            "qrels": 1,
            "audit_rows": 1,
        },
        "artifacts": artifacts,
    }
    manifest_path = root / "data/manifests/release-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


def test_combine_releases_merges_validated_artifacts_and_recomputes_positive_corpus(tmp_path: Path):
    first = tmp_path / "release-a"
    second = tmp_path / "release-b"
    output = tmp_path / "combined"
    _write_release(first, suffix="a", passage_id="p-a", article_id="article-a", split="train")
    _write_release(second, suffix="b", passage_id="p-b", article_id="article-b", split="test")

    result = combine_releases([first, second], output, git_sha="combined-git", git_dirty=False)

    assert result.pair_count == 2
    assert result.qrel_count == 2
    assert result.corpus_count == 2
    assert result.review_count == 2
    assert result.reused is False
    train = pl.read_parquet(output / "data/train-00000-of-00001.parquet")
    dev = pl.read_parquet(output / "data/dev-00000-of-00001.parquet")
    test = pl.read_parquet(output / "data/test-00000-of-00001.parquet")
    corpus = pl.read_parquet(output / "data/corpus-00000-of-00001.parquet")
    assert train.schema == dev.schema == test.schema == corpus.schema == pl.Schema(PUBLIC_SCHEMA)
    assert train.select("query_id", "positive_doc_id", "query", "positive", "doc_id", "content").to_dicts() == [
        {
            "query_id": "q-a",
            "positive_doc_id": "p-a",
            "query": "Как работает механизм a?",
            "positive": "Заголовок a\n\nТекст a",
            "doc_id": "",
            "content": "",
        }
    ]
    assert dev.is_empty()
    assert test["query_id"].to_list() == ["q-b"]
    assert corpus["doc_id"].to_list() == ["p-a", "p-b"]
    assert corpus["query"].to_list() == ["", ""]
    assert corpus["is_positive"].to_list() == [True, True]
    assert pl.read_csv(output / "audit/pilot-review.csv").height == 2
    assert (output / "artifacts/qrels/train.parquet").exists()
    assert (output / "artifacts/qrels/dev.parquet").exists()
    assert (output / "artifacts/qrels/test.parquet").exists()
    dataset_card = (output / "README.md").read_text(encoding="utf-8")
    assert "- text-retrieval" in dataset_card
    assert "- information-retrieval" not in dataset_card
    metadata = yaml.safe_load(dataset_card.split("---", 2)[1])
    assert metadata["configs"] == [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": "data/train-*.parquet"},
                {"split": "dev", "path": "data/dev-*.parquet"},
                {"split": "test", "path": "data/test-*.parquet"},
                {"split": "corpus", "path": "data/corpus-*.parquet"},
            ],
        }
    ]

    assert result.manifest_path == output / "manifest.json"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract"] == "habr-ir-retrieval-release-v1"
    assert manifest["git"] == {"sha": "combined-git", "dirty": False}
    assert manifest["counts"]["splits"] == {"train": 1, "dev": 0, "test": 1, "corpus": 2}
    assert [source["fingerprint"] for source in manifest["releases"]] == ["fingerprint-a", "fingerprint-b"]

    reused = combine_releases([first, second], output, git_sha="combined-git", git_dirty=False)
    assert reused.reused is True
    assert reused.fingerprint == result.fingerprint


def test_combine_releases_rejects_duplicate_positive_passage(tmp_path: Path):
    first = tmp_path / "release-a"
    second = tmp_path / "release-b"
    _write_release(first, suffix="a", passage_id="p-a", article_id="article-a", split="train")
    _write_release(second, suffix="b", passage_id="p-a", article_id="article-a", split="train")

    try:
        combine_releases([first, second], tmp_path / "combined", git_sha="combined-git", git_dirty=False)
    except ValueError as exc:
        assert "duplicate positive_passage_id" in str(exc)
    else:
        raise AssertionError("duplicate positive passage was accepted")


def test_release_config_controls_sources_layout_and_publication(tmp_path: Path):
    path = tmp_path / "release.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "release": {
                    "repo_id": "justatom/habr-ir",
                    "private": False,
                    "config_name": "default",
                    "layout": "retrieval",
                    "source_releases": ["first", "second"],
                    "output_root": "combined",
                    "split_map": {"train": "train", "validation": "dev", "test": "test", "corpus": "corpus"},
                    "include_audit": True,
                    "include_qrels_artifacts": True,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_release_config(path)

    assert config.repo_id == "justatom/habr-ir"
    assert config.private is False
    assert config.config_name == "default"
    assert config.layout == "retrieval"
    assert config.source_releases == (Path("first"), Path("second"))
    assert config.output_root == Path("combined")
    assert config.split_map == {"train": "train", "validation": "dev", "test": "test", "corpus": "corpus"}
    assert config.include_audit is True
    assert config.include_qrels_artifacts is True
