from justatom.processing import tokenizer as module


def test_tokenizer_forwards_pinned_hub_revision(monkeypatch):
    sentinel = object()
    captured = {}

    def fake_ignite(where, revision=None):
        captured.update(where=where, revision=revision)
        return sentinel

    monkeypatch.setattr(module, "ignite_hf_tokenizer", fake_ignite)

    tokenizer = module.ITokenizer.from_pretrained(
        "owner/model",
        revision="snapshot-123",
    )

    assert tokenizer is sentinel
    assert captured == {"where": "owner/model", "revision": "snapshot-123"}
