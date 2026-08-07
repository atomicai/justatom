from pathlib import Path

import yaml


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


def test_compose_defines_api_and_mutually_exclusive_embedding_profiles():
    compose = yaml.safe_load(_read("docker-compose.yaml"))
    services = compose["services"]
    assert services["api"]["build"]["dockerfile"] == "Dockerfile.api"
    assert services["api"]["ports"] == ["${JUSTATOM_API_PORT:-5555}:5555"]
    assert services["api"]["extra_hosts"] == ["host.docker.internal:host-gateway"]
    assert services["weaviate"]["ports"] == [
        "${WEAVIATE_HTTP_PORT:-2211}:2211",
        "${WEAVIATE_GRPC_PORT:-50051}:50051",
    ]
    assert services["embedder-cpu"]["profiles"] == ["cpu"]
    assert services["embedder-cuda"]["profiles"] == ["cuda"]
    assert services["api"]["depends_on"]["embedder-cpu"]["required"] is False
    assert services["api"]["depends_on"]["embedder-cuda"]["required"] is False
    assert services["embedder-cuda"]["deploy"]["resources"]["reservations"]["devices"][0][
        "capabilities"
    ] == ["gpu"]
    assert "huggingface-cache" in compose["volumes"]


def test_docker_serve_config_uses_internal_weaviate_and_embedding_alias():
    config = yaml.safe_load(_read("configs/serve.docker.yaml"))
    retrieval = config["retrieval"]
    assert retrieval["embedding"]["backend"] == "openai-compatible"
    assert retrieval["embedding"]["base_url"] == "${EMBEDDING_BASE_URL}"
    assert retrieval["store"]["url"] == "http://weaviate:2211"
