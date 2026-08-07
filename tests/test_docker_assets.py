from pathlib import Path


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def test_api_image_is_model_free_and_runs_production_entrypoint():
    dockerfile = _read("Dockerfile.api")
    assert ".[serve]" in dockerfile
    assert "justatom.api.serve" in dockerfile
    assert "torch" not in dockerfile.lower()
    assert "Qwen" not in dockerfile


def test_embedding_images_share_endpoint_and_select_platform_torch():
    cpu = _read("Dockerfile.embedder.cpu")
    cuda = _read("Dockerfile.embedder.cuda")
    assert "justatom.api.serve_embeddings" in cpu
    assert "justatom.api.serve_embeddings" in cuda
    assert "download.pytorch.org/whl/cpu" in cpu
    assert "download.pytorch.org/whl/cu128" in cuda
    assert "EMBEDDING_DEVICE=cpu" in cpu
    assert "EMBEDDING_DEVICE=cuda:0" in cuda


def test_dockerignore_excludes_credentials_weights_and_worktrees():
    ignored = _read(".dockerignore").splitlines()
    assert ".env" in ignored
    assert ".cache/" in ignored
    assert "weights/" in ignored
    assert "models/" in ignored
    assert "*.safetensors" in ignored
    assert "*.gguf" in ignored
    assert ".worktrees/" in ignored
    assert ".tmp_runs/" in ignored
    assert "phd.paper/" in ignored
