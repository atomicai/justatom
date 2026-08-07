import re
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


def test_cpu_smoke_script_uses_launcher_cleanup_and_utf8_assertions():
    script = _read("scripts/smoke_containerized_retrieval.sh")
    assert "set -euo pipefail" in script
    assert 'export COMPOSE_PROJECT_NAME="$PROJECT"' in script
    assert "scripts/services.sh cpu up -d --build weaviate embedder-cpu api" in script
    assert "scripts/services.sh cpu logs --no-color" in script
    assert "scripts/services.sh cpu down -v --remove-orphans" in script
    assert "docker compose -p" not in script
    assert "COMPOSE_PROFILES=cpu" not in script
    assert "JUSTATOM_API_PORT" in script
    assert "EMBEDDING_PORT" in script
    assert "WEAVIATE_HTTP_PORT" in script
    assert "WEAVIATE_GRPC_PORT" in script
    assert "Loading from huggingface hub via" in script
    assert "\\\\u" in script
    assert '.docs[0].meta.topic == "storage"' in script


def test_cpu_smoke_script_bounds_readiness_and_inference_requests():
    script = _read("scripts/smoke_containerized_retrieval.sh")
    assert "curl --connect-timeout 2 --max-time 5 --fail --silent --show-error" in script
    assert script.count("curl --connect-timeout 5 --max-time 300 --fail --silent --show-error") == 3
    assert script.count("--connect-timeout") == 4
    assert script.count("--max-time") == 4


def test_external_backend_smoke_uses_real_api_image_without_torch():
    script = _read("scripts/smoke_api_external_backend.sh")
    fixture = _read("tests/fixtures/openai_embedding_stub.py")

    assert "set -euo pipefail" in script
    assert 'export COMPOSE_PROJECT_NAME="$PROJECT"' in script
    assert "host.docker.internal" in script
    assert "scripts/services.sh external up -d --build weaviate api" in script
    assert "scripts/services.sh external logs --no-color" in script
    assert "scripts/services.sh external down -v --remove-orphans" in script
    assert "docker compose" not in script
    assert "COMPOSE_PROFILES" not in script
    assert 'find_spec("torch") is None' in script
    assert "JUSTATOM_API_PORT" in script
    assert "WEAVIATE_HTTP_PORT" in script
    assert "WEAVIATE_GRPC_PORT" in script
    assert "FAKE_EMBEDDING_PORT" in script
    assert 'PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert 'kill -0 "$FAKE_PID"' in script
    assert ".docs[0].meta.topic == \"retrieval\"" in script
    assert ".docs[0].meta.topic == \"storage\"" in script
    assert "банк негативов" in script
    assert "\\\\u" in script
    assert "/v1/embeddings" in fixture
    assert '"/health"' in fixture
    assert '"/v1/models"' in fixture
    assert "fixture-embedding-model" in fixture
    assert 'app.config.setdefault("PROVIDE_AUTOMATIC_OPTIONS", True)' in fixture


def test_external_backend_smoke_cleanup_enforces_owned_resource_and_isolation_audits():
    script = _read("scripts/smoke_api_external_backend.sh")

    assert "local main_status=$?" in script
    assert "trap - EXIT INT TERM" in script
    assert "if ! scripts/services.sh external down -v --remove-orphans" in script
    assert "cleanup_failed=1" in script
    assert "after_projects" in script
    assert "before_projects" in script
    assert '[[ "$after_projects" != "$before_projects" ]]' in script
    assert "if (( main_status == 0 && cleanup_failed )); then" in script
    assert 'exit "$main_status"' in script


def test_external_backend_smoke_monitors_and_reaps_the_host_stub_for_long_commands():
    script = _read("scripts/smoke_api_external_backend.sh")

    assert 'CHILD_PID=""' in script
    assert "run_with_stub_watch()" in script
    assert 'kill -0 "$CHILD_PID"' in script
    assert 'kill "$CHILD_PID"' in script
    assert 'wait "$CHILD_PID"' in script
    assert "run_with_stub_watch scripts/services.sh external up -d --build weaviate api" in script
    assert script.count("run_with_stub_watch curl --connect-timeout 5 --max-time 60") == 3


def test_external_backend_smoke_watches_final_docker_checks_before_success():
    script = _read("scripts/smoke_api_external_backend.sh")

    assert "API_CONTAINER_FILE" in script
    assert 'run_with_stub_watch docker ps -q' in script
    assert 'run_with_stub_watch docker exec "$api_container" python -c' in script
    assert 'ensure_fake_embedding_is_alive\nprintf \'model-free API smoke passed' in script


def test_external_backend_smoke_static_contract_rejects_isolation_and_timeout_mutations():
    script = _read("scripts/smoke_api_external_backend.sh")

    assert 'PROJECT="justatom-api-smoke-$(date +%s)-$$"' in script
    assert 'JUSTATOM_API_PORT="${JUSTATOM_API_PORT:-15556}"' in script
    assert 'WEAVIATE_HTTP_PORT="${WEAVIATE_HTTP_PORT:-13212}"' in script
    assert 'WEAVIATE_GRPC_PORT="${WEAVIATE_GRPC_PORT:-15052}"' in script
    assert 'FAKE_EMBEDDING_PORT="${FAKE_EMBEDDING_PORT:-18001}"' in script
    assert 'trap cleanup EXIT' in script
    assert "trap 'exit 130' INT" in script
    assert "trap 'exit 143' TERM" in script
    assert script.count("scripts/services.sh external up -d --build weaviate api") == 1
    assert "docker compose" not in script
    assert "COMPOSE_PROFILES" not in script
    assert " -p " not in script
    assert "--project-name" not in script
    assert script.count("curl --connect-timeout 2 --max-time 5 --fail --silent --show-error") == 2
    assert script.count("run_with_stub_watch curl --connect-timeout 5 --max-time 60") == 3
    assert script.count('"content":') == 3
    assert "cleanup warning" not in script
    assert "cleanup failure: smoke project resources remain" in script
    assert "cleanup failure: one or more smoke ports remain occupied" in script
    assert "cleanup failure: pre-existing Compose projects changed" in script


def test_external_backend_smoke_rejects_managed_embedders_and_couples_fatal_branches():
    script = _read("scripts/smoke_api_external_backend.sh")
    launcher = re.search(
        r"^if ! run_with_stub_watch scripts/services\.sh external up -d --build weaviate api; then$",
        script,
        flags=re.MULTILINE,
    )
    cleanup = re.search(r"cleanup\(\) \{(?P<body>.*?)^\}", script, flags=re.MULTILINE | re.DOTALL)

    assert launcher is not None
    assert not re.search(
        r"run_with_stub_watch scripts/services\.sh external up[^\n;]*(?:embedder-cpu|embedder-cuda)",
        script,
    )
    assert cleanup is not None
    body = cleanup.group("body")
    for condition, failure in [
        (r'elif \[\[ -n "\$remaining" \]\]; then', "smoke project resources remain"),
        (r"if ! check_ports_are_free; then", "one or more smoke ports remain occupied"),
        (r'elif \[\[ "\$after_projects" != "\$before_projects" \]\]; then', "pre-existing Compose projects changed"),
    ]:
        branch = re.search(
            rf"{condition}(?P<branch>.*?)(?=^  (?:elif|else|fi)$)",
            body,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert branch is not None
        assert f"cleanup failure: {failure}" in branch.group("branch")
        assert "cleanup_failed=1" in branch.group("branch")
