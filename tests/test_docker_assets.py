import copy
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def _launcher_json(mode, command, *args):
    result = subprocess.run(
        ["bash", "scripts/services.sh", mode, command, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _compose_profile_json(profile):
    env = os.environ.copy()
    env["COMPOSE_PROFILES"] = profile
    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _dockerfile_instructions(dockerfile):
    instructions = []
    logical_line = ""
    for raw_line in dockerfile.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        logical_line += line
        if logical_line.endswith("\\"):
            logical_line = f"{logical_line[:-1].rstrip()} "
            continue
        instruction, value = logical_line.split(maxsplit=1)
        instructions.append((instruction.upper(), value))
        logical_line = ""
    assert not logical_line
    return instructions


def _assert_dockerfile_contract(path, dockerfile):
    instructions = _dockerfile_instructions(dockerfile)
    values = lambda name: [value for instruction, value in instructions if instruction == name]
    environment = {}
    for value in values("ENV"):
        for assignment in shlex.split(value):
            key, setting = assignment.split("=", maxsplit=1)
            environment[key] = setting

    assert [tuple(shlex.split(value)) for value in values("COPY")] == [
        ("pyproject.toml", "README.md", "./"),
        ("justatom", "./justatom"),
    ]
    assert (".", ".") not in [tuple(shlex.split(value)) for value in values("COPY")]
    assert values("USER") == ["10001:10001"]

    if path == "Dockerfile.api":
        assert ".[serve]" in dockerfile
        assert environment == {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "JUSTATOM_CONFIG": "/etc/justatom/serve.yaml",
            "JUSTATOM_START_MQ": "false",
        }
        assert values("EXPOSE") == ["5555"]
        assert values("HEALTHCHECK") == [
            "--interval=10s --timeout=3s --start-period=10s --retries=6 "
            'CMD ["python", "-c", "import urllib.request; '
            "urllib.request.urlopen('http://127.0.0.1:5555/', timeout=2)\"]"
        ]
        assert values("CMD") == ['["python", "-m", "justatom.api.serve"]']
    else:
        assert ".[embedder]" in dockerfile
        device = "cpu" if path == "Dockerfile.embedder.cpu" else "cuda:0"
        assert environment == {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/cache/huggingface",
            "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
            "EMBEDDING_DEVICE": device,
            "EMBEDDING_BATCH_SIZE": "8",
            "EMBEDDING_MAX_LENGTH": "512",
        }
        assert values("EXPOSE") == ["8000"]
        assert values("HEALTHCHECK") == [
            "--interval=10s --timeout=3s --start-period=120s --retries=30 "
            'CMD ["python", "-c", "import urllib.request; '
            "urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)\"]"
        ]
        assert values("CMD") == ['["python", "-m", "justatom.api.serve_embeddings"]']


@pytest.mark.parametrize(
    ("path", "mutation"),
    [
        ("Dockerfile.embedder.cpu", lambda value: value.replace("Qwen/Qwen3-Embedding-0.6B", "other/model", 1)),
        ("Dockerfile.embedder.cpu", lambda value: value.replace("EMBEDDING_BATCH_SIZE=8", "EMBEDDING_BATCH_SIZE=16", 1)),
        ("Dockerfile.embedder.cpu", lambda value: value.replace("HF_HOME=/cache/huggingface", "HF_HOME=/tmp/cache", 1)),
        ("Dockerfile.api", lambda value: value.replace("USER 10001:10001", "USER root", 1)),
        ("Dockerfile.embedder.cuda", lambda value: value.replace("EXPOSE 8000", "EXPOSE 9000", 1)),
        ("Dockerfile.api", lambda value: value.replace("http://127.0.0.1:5555/", "http://127.0.0.1:9999/", 1)),
        ("Dockerfile.embedder.cpu", lambda value: value.replace("COPY justatom ./justatom", "COPY . .", 1)),
    ],
)
def test_dockerfile_contract_rejects_default_and_copy_mutations(path, mutation):
    dockerfile = _read(path)
    _assert_dockerfile_contract(path, dockerfile)

    with pytest.raises(AssertionError):
        _assert_dockerfile_contract(path, mutation(dockerfile))


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
    assert services["embedder-cuda"]["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]
    assert "huggingface-cache" in compose["volumes"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "expected_services"),
    [
        ("external", ["api", "weaviate"]),
        ("cpu", ["api", "embedder-cpu", "weaviate"]),
        ("cuda", ["api", "embedder-cuda", "weaviate"]),
    ],
)
def test_launcher_modes_render_exact_retrieval_services_without_redis(mode, expected_services):
    compose = _launcher_json(mode, "config", "--format", "json")

    assert sorted(compose["services"]) == expected_services
    assert "redis" not in compose["services"]


@pytest.mark.integration
def test_cuda_compose_and_bake_pin_linux_amd64_platform():
    compose = _launcher_json("cuda", "config", "--format", "json")
    bake = _launcher_json("cuda", "build", "--print", "embedder-cuda")

    assert compose["services"]["embedder-cuda"]["platform"] == "linux/amd64"
    assert bake["target"]["embedder-cuda"]["platforms"] == ["linux/amd64"]


def _rendered_compose_contract():
    return {
        "external": _launcher_json("external", "config", "--format", "json"),
        "cpu": _launcher_json("cpu", "config", "--format", "json"),
        "cuda": _launcher_json("cuda", "config", "--format", "json"),
        "legacy": _compose_profile_json("legacy"),
    }


def _assert_rendered_compose_contract(rendered):
    assert sorted(rendered["external"]["services"]) == ["api", "weaviate"]
    assert sorted(rendered["cpu"]["services"]) == ["api", "embedder-cpu", "weaviate"]
    assert sorted(rendered["cuda"]["services"]) == ["api", "embedder-cuda", "weaviate"]
    assert rendered["cuda"]["services"]["embedder-cuda"]["platform"] == "linux/amd64"
    assert sorted(rendered["legacy"]["services"]) == ["api", "redis", "weaviate"]

    for mode, config in rendered.items():
        api = config["services"]["api"]
        weaviate = config["services"]["weaviate"]
        assert api["environment"]["JUSTATOM_START_MQ"] == "false"
        assert api.get("restart") == "unless-stopped"
        assert len(api["volumes"]) == 1
        api_config = api["volumes"][0]
        assert api_config == {
            "type": "bind",
            "source": api_config["source"],
            "target": "/etc/justatom/serve.yaml",
            "read_only": True,
            "bind": {},
        }
        assert Path(api_config["source"]).resolve() == Path("configs/serve.docker.yaml").resolve()
        assert weaviate["restart"] == "on-failure:0"
        assert weaviate["volumes"] == [
            {
                "type": "volume",
                "source": "weaviatedb",
                "target": "/var/lib/weaviate",
                "volume": {},
            }
        ]
        expected_volumes = {"weaviatedb"} if mode in {"external", "legacy"} else {"huggingface-cache", "weaviatedb"}
        assert set(config["volumes"]) == expected_volumes
        for volume_name, volume in config["volumes"].items():
            assert volume["name"].endswith(f"_{volume_name}")

    expected_embedders = {"cpu": "embedder-cpu", "cuda": "embedder-cuda"}
    for mode, service_name in expected_embedders.items():
        embedder = rendered[mode]["services"][service_name]
        assert embedder["networks"] == {"default": {"aliases": ["embedder"]}}
        assert embedder["volumes"] == [
            {
                "type": "volume",
                "source": "huggingface-cache",
                "target": "/cache/huggingface",
                "volume": {},
            }
        ]
        assert embedder["restart"] == "unless-stopped"

    cuda_device = rendered["cuda"]["services"]["embedder-cuda"]["deploy"]["resources"]["reservations"]["devices"]
    assert cuda_device == [{"capabilities": ["gpu"], "driver": "nvidia", "count": 1}]

    redis = rendered["legacy"]["services"]["redis"]
    assert redis["profiles"] == ["legacy"]
    assert redis["image"] == "redis:latest"
    assert redis["command"] == ["redis-server", "/redis.conf"]
    assert redis["ports"] == [{"mode": "ingress", "target": 6379, "published": "6379", "protocol": "tcp"}]
    assert len(redis["volumes"]) == 1
    redis_config = redis["volumes"][0]
    assert redis_config == {
        "type": "bind",
        "source": redis_config["source"],
        "target": "/redis.conf",
        "bind": {},
    }
    assert Path(redis_config["source"]).resolve() == Path("redis.conf").resolve()


@pytest.mark.integration
@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["cpu"]["services"]["embedder-cpu"]["networks"]["default"].update(aliases=[]),
        lambda value: value["cuda"]["services"]["embedder-cuda"].update(volumes=[]),
        lambda value: value["external"]["services"]["api"]["volumes"][0].update(read_only=False),
        lambda value: value["cpu"]["services"]["api"]["environment"].update(JUSTATOM_START_MQ="true"),
        lambda value: value["cuda"]["services"]["embedder-cuda"]["deploy"]["resources"]["reservations"]["devices"][0].update(
            driver="other", count=2, capabilities=["compute"]
        ),
        lambda value: value["external"]["services"]["api"].pop("restart"),
        lambda value: value["legacy"]["services"]["redis"].update(profiles=[]),
        lambda value: value["legacy"]["services"]["redis"].update(command=["redis-server"]),
        lambda value: value["cpu"].update(volumes={}),
    ],
)
def test_rendered_compose_contract_rejects_topology_mutations(mutation):
    rendered = _rendered_compose_contract()
    _assert_rendered_compose_contract(rendered)
    mutated = copy.deepcopy(rendered)
    mutation(mutated)

    with pytest.raises(AssertionError):
        _assert_rendered_compose_contract(mutated)


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


def _assert_cpu_smoke_cleanup_contract(script):
    assert 'PROJECT="justatom-smoke-$(date +%s)-$$"' in script
    assert 'export COMPOSE_PROJECT_NAME="$PROJECT"' in script
    assert "source scripts/smoke_docker_audit.sh" in script
    assert "check_ports_are_free()" in script
    assert 'before_projects=""' in script
    assert "before_projects_ready=false" in script
    assert 'preexisting_project_resources="$(project_resources)"' in script
    assert '[[ -n "$preexisting_project_resources" ]]' in script
    assert 'before_projects="$(list_preexisting_compose_projects)"' in script
    assert "before_projects_ready=true" in script
    assert script.count("check_ports_are_free") == 3

    cleanup = re.search(r"cleanup\(\) \{(?P<body>.*?)^\}", script, flags=re.MULTILINE | re.DOTALL)
    assert cleanup is not None
    body = cleanup.group("body")
    assert "local main_status=$?" in body
    assert "local cleanup_failed=0" in body
    assert "trap - EXIT INT TERM" in body
    assert "|| true" not in body
    for condition, failure in [
        (
            r"if ! scripts/services\.sh cpu down -v --remove-orphans[^;]*; then",
            "launcher teardown failed",
        ),
        (r'elif \[\[ -n "\$remaining" \]\]; then', "smoke project resources remain"),
        (r"if ! check_ports_are_free; then", "one or more smoke ports remain occupied"),
        (
            r'elif \[\[ "\$after_projects" != "\$before_projects" \]\]; then',
            "pre-existing Compose projects changed",
        ),
    ]:
        branch = re.search(
            rf"{condition}(?P<body>.*?{re.escape(failure)}.*?cleanup_failed=1)",
            body,
            flags=re.DOTALL,
        )
        assert branch is not None
    assert "if (( main_status == 0 && cleanup_failed )); then" in body
    assert re.search(r"if \(\( main_status == 0 && cleanup_failed \)\); then\n    exit 1\n  fi", body)
    assert 'exit "$main_status"' in body
    assert "trap cleanup EXIT" in script
    assert "trap 'exit 130' INT" in script
    assert "trap 'exit 143' TERM" in script


def test_cpu_smoke_cleanup_is_fatal_and_audits_isolation():
    _assert_cpu_smoke_cleanup_contract(_read("scripts/smoke_containerized_retrieval.sh"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda script: script.replace("local main_status=$?", "local main_status=0", 1),
        lambda script: script.replace(
            "if ! scripts/services.sh cpu down -v --remove-orphans >/dev/null 2>&1; then",
            "scripts/services.sh cpu down -v --remove-orphans >/dev/null 2>&1 || true\n  if false; then",
            1,
        ),
        lambda script: script.replace('elif [[ -n "$remaining" ]]; then', "elif false; then", 1),
        lambda script: script.replace("if ! check_ports_are_free; then", "if false; then", 1),
        lambda script: script.replace('elif [[ "$after_projects" != "$before_projects" ]]; then', "elif false; then", 1),
        lambda script: script.replace(
            "if (( main_status == 0 && cleanup_failed )); then\n    exit 1\n  fi",
            "if (( main_status == 0 && cleanup_failed )); then\n    exit 0\n  fi",
            1,
        ),
        lambda script: script.replace("trap 'exit 143' TERM", "trap 'exit 0' TERM", 1),
    ],
)
def test_cpu_smoke_cleanup_contract_rejects_mutations(mutation):
    script = _read("scripts/smoke_containerized_retrieval.sh")
    _assert_cpu_smoke_cleanup_contract(script)
    with pytest.raises(AssertionError):
        _assert_cpu_smoke_cleanup_contract(mutation(script))


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
    assert '.docs[0].meta.topic == "retrieval"' in script
    assert '.docs[0].meta.topic == "storage"' in script
    assert "банк негативов" in script
    assert "\\\\u" in script
    assert "/v1/embeddings" in fixture
    assert '"/health"' in fixture
    assert '"/v1/models"' in fixture
    assert "fixture-embedding-model" in fixture
    assert 'app.config.setdefault("PROVIDE_AUTOMATIC_OPTIONS", True)' in fixture


def test_external_backend_smoke_cleanup_enforces_owned_resource_and_isolation_audits():
    script = _read("scripts/smoke_api_external_backend.sh")

    assert "source scripts/smoke_docker_audit.sh" in script
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
    assert "run_with_stub_watch docker ps -q" in script
    assert 'run_with_stub_watch docker exec "$api_container" python -c' in script
    assert "ensure_fake_embedding_is_alive\nprintf 'model-free API smoke passed" in script


def test_external_backend_smoke_static_contract_rejects_isolation_and_timeout_mutations():
    script = _read("scripts/smoke_api_external_backend.sh")

    assert 'PROJECT="justatom-api-smoke-$(date +%s)-$$"' in script
    assert 'JUSTATOM_API_PORT="${JUSTATOM_API_PORT:-15556}"' in script
    assert 'WEAVIATE_HTTP_PORT="${WEAVIATE_HTTP_PORT:-13212}"' in script
    assert 'WEAVIATE_GRPC_PORT="${WEAVIATE_GRPC_PORT:-15052}"' in script
    assert 'FAKE_EMBEDDING_PORT="${FAKE_EMBEDDING_PORT:-18001}"' in script
    assert "trap cleanup EXIT" in script
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


def _assert_deployment_docs_contract(documents):
    readme = documents["README.md"]
    guide = documents["docs/launch-guide.md"]
    architecture = documents["docs/architecture.md"]
    runtime = documents["docs/modules/runtime.md"]

    assert "scripts/services.sh cpu up -d --build" in guide
    assert "scripts/services.sh cuda up -d --build" in guide
    assert "EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1" in guide
    assert "scripts/services.sh external up -d --build api weaviate" in guide
    assert "EMBEDDING_DEVICE=mps" in guide
    assert "conda run -n justatom python -m justatom.api.serve_embeddings" in guide
    assert "Dockerfile.api" in readme
    assert "Dockerfile.embedder.cpu" in guide
    assert "Dockerfile.embedder.cuda" in guide
    assert "no Torch or model" in guide
    assert "portable CPU" in guide
    assert "Linux/NVIDIA only" in guide
    assert "Linux x86_64/amd64" in guide
    assert "cannot expose MPS to containers" in guide
    assert "vLLM" in guide
    assert "Triton" in guide
    assert "llama.cpp" in guide
    assert "One embedding service process owns one model instance" in guide
    assert "do not reload it" in guide
    assert "does not preserve the model in RAM" in guide
    assert "CUDA mode never falls back to CPU." in guide
    assert "client -> justatom-api -> OpenAICompatibleEmbedder -> embedding HTTP service" in architecture
    assert "`-> WeaviateDocumentStore -> Weaviate" in architecture
    assert "Query and document prefixes" in architecture
    assert "batching" in architecture
    assert "before HTTP inference" in architecture
    assert "tokenization and model execution only" in architecture
    assert "OpenAICompatibleEmbedder" in runtime

    for document in (readme, guide, architecture, runtime):
        assert "docker compose" not in document
        assert "COMPOSE_PROFILES" not in document
        assert "--profile" not in document
        assert not re.search(r"(^|\s)-p(?:\s|=)", document)


def test_deployment_docs_use_the_launcher_and_describe_runtime_boundaries():
    _assert_deployment_docs_contract(
        {
            "README.md": _read("README.md"),
            "docs/launch-guide.md": _read("docs/launch-guide.md"),
            "docs/architecture.md": _read("docs/architecture.md"),
            "docs/modules/runtime.md": _read("docs/modules/runtime.md"),
        }
    )


@pytest.mark.parametrize("project_argument", ["-p demo", "-p=demo"])
def test_deployment_docs_contract_rejects_manual_compose_project_selection(project_argument):
    documents = {
        "README.md": _read("README.md"),
        "docs/launch-guide.md": _read("docs/launch-guide.md") + f"\n{project_argument}\n",
        "docs/architecture.md": _read("docs/architecture.md"),
        "docs/modules/runtime.md": _read("docs/modules/runtime.md"),
    }

    with pytest.raises(AssertionError):
        _assert_deployment_docs_contract(documents)


def _assert_native_mps_smoke_contract(script):
    assert '"$(uname -s)" != "Darwin"' in script
    assert '"$(uname -m)" != "arm64"' in script
    assert "Apple Silicon macOS is required" in script
    assert 'EMBEDDING_PORT="${EMBEDDING_PORT:-18002}"' in script
    assert "EMBEDDING_DEVICE=mps" in script
    assert "torch.backends.mps.is_available()" in script
    assert 'serve_app(build_embedding_app(), host="127.0.0.1"' in script
    assert "docker compose" not in script
    assert "scripts/services.sh" not in script
    port_check = re.search(
        r"check_embedding_port_is_free\(\) \{(?P<body>.*?)^\}",
        script,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert port_check is not None
    assert 'listener.bind(("127.0.0.1", int(sys.argv[1])))' in port_check.group("body")
    assert script.count("check_embedding_port_is_free") == 3
    assert re.search(
        r"if ! check_embedding_port_is_free; then\n" r'  fail "port \$\{EMBEDDING_PORT\} is already in use"\nfi',
        script,
    )
    assert re.search(
        r"for attempt in \$\(seq 1 360\); do\n"
        r'    if curl --connect-timeout 2 --max-time 5 --fail --silent --show-error "\$url" >/dev/null 2>&1; then\n'
        r"      return 0\n    fi\n    sleep 1\n  done\n"
        r'  fail "\$name did not become ready at \$url"',
        script,
    )
    cleanup = re.search(r"cleanup\(\) \{(?P<body>.*?)^\}", script, flags=re.MULTILINE | re.DOTALL)
    assert cleanup is not None
    cleanup_body = cleanup.group("body")
    assert "local main_status=$?" in cleanup_body
    assert "local cleanup_failed=0" in cleanup_body
    assert "trap - EXIT INT TERM" in cleanup_body
    assert "|| true" not in cleanup_body
    for condition, failure in [
        (r"if ! terminate_server; then", "embedding server could not be terminated and reaped"),
        (r'if \[\[ -n "\$SERVER_PID" \]\] && kill -0', "embedding server remains alive"),
        (r"if ! check_embedding_port_is_free; then", "embedding port remains occupied"),
        (r'if ! rm -f "\$LOG_FILE"; then', "embedding log could not be removed"),
    ]:
        branch = re.search(
            rf"{condition}(?P<body>.*?{re.escape(failure)}.*?cleanup_failed=1)",
            cleanup_body,
            flags=re.DOTALL,
        )
        assert branch is not None
    assert re.search(
        r"if \(\( main_status == 0 && cleanup_failed \)\); then\n    exit 1\n  fi",
        cleanup_body,
    )
    assert 'exit "$main_status"' in cleanup_body
    termination = re.search(r"terminate_server\(\) \{(?P<body>.*?)^\}", script, flags=re.MULTILINE | re.DOTALL)
    assert termination is not None
    termination_body = termination.group("body")
    assert 'kill -TERM "$SERVER_PID"' in termination_body
    assert "for attempt in $(seq 1 10); do" in termination_body
    assert 'kill -KILL "$SERVER_PID"' in termination_body
    assert 'wait "$SERVER_PID"' in termination_body
    assert "0|137|143" in termination_body
    assert "trap cleanup EXIT" in script
    assert "trap 'exit 130' INT" in script
    assert "trap 'exit 143' TERM" in script
    assert 'wait_http "embedding health"' in script
    assert 'wait_http "embedding models"' in script
    assert "http://127.0.0.1:${EMBEDDING_PORT}/v1/models" in script
    assert script.count("curl --connect-timeout 5 --max-time 300 --fail --silent --show-error") == 2
    assert script.count("http://127.0.0.1:${EMBEDDING_PORT}/v1/embeddings") == 2
    assert "русский запрос" in script
    assert "$data[0].index == 0" in script
    assert "$data[1].index == 1" in script
    assert script.count("select(.model == $model)") == 2
    assert 'first_dimension="$(' in script
    assert '--argjson dimension "$first_dimension"' in script
    assert "== $dimension" in script
    assert "Loading from huggingface hub via" in script
    assert '[[ "$model_loads" == "1" ]]' in script
    assert "ensure_server_is_alive\nprintf 'native MPS embedding smoke passed" in script
    for value in ("first_request", "second_request", "first_response", "second_response"):
        assert f"printf '%s' \"${value}\" | grep -Eq" in script
    assert script.count("escaped readable UTF-8") == 4


def test_native_mps_smoke_has_a_bounded_host_only_lifecycle_and_contract_checks():
    _assert_native_mps_smoke_contract(_read("scripts/smoke_native_embedding.sh"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda script: script.replace("if ! check_embedding_port_is_free; then", "if false; then", 1),
        lambda script: script.replace(
            'listener.bind(("127.0.0.1", int(sys.argv[1])))', 'listener.bind(("0.0.0.0", int(sys.argv[1])))', 1
        ),
        lambda script: script.replace("for attempt in $(seq 1 360); do", "while true; do", 1),
        lambda script: script.replace('exit "$main_status"', "exit 0", 1),
        lambda script: re.sub(
            r'if printf \'%s\' "\$(?:first|second)_(?:request|response)".*?^fi\n', "", script, flags=re.MULTILINE | re.DOTALL
        ),
        lambda script: re.sub(r'if printf \'%s\' "\$first_response".*?^fi\n', "", script, count=1, flags=re.MULTILINE | re.DOTALL),
        lambda script: script.replace("select(.model == $model)", 'select(.model == "wrong")', 1),
        lambda script: script.replace('--argjson dimension "$first_dimension"', '--argjson dimension "0"', 1),
    ],
)
def test_native_mps_smoke_contract_rejects_safety_mutations(mutation):
    script = _read("scripts/smoke_native_embedding.sh")
    _assert_native_mps_smoke_contract(script)
    with pytest.raises(AssertionError):
        _assert_native_mps_smoke_contract(mutation(script))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda script: script.replace('kill -TERM "$SERVER_PID"', ":", 1),
        lambda script: script.replace("for attempt in $(seq 1 10); do", "while true; do", 1),
        lambda script: script.replace('kill -KILL "$SERVER_PID"', ":", 1),
        lambda script: script.replace('wait "$SERVER_PID"', "wait_status=0", 1),
        lambda script: script.replace('if [[ -n "$SERVER_PID" ]] && kill -0', 'if [[ -n "$SERVER_PID" ]] && false && kill -0', 1),
        lambda script: script.replace(
            "if (( main_status == 0 && cleanup_failed )); then\n    exit 1\n  fi",
            "if (( main_status == 0 && cleanup_failed )); then\n    exit 0\n  fi",
            1,
        ),
        lambda script: script.replace(
            "ensure_server_is_alive\nprintf 'native MPS embedding smoke passed",
            "printf 'native MPS embedding smoke passed",
            1,
        ),
    ],
)
def test_native_mps_smoke_cleanup_contract_rejects_mutations(mutation):
    script = _read("scripts/smoke_native_embedding.sh")
    _assert_native_mps_smoke_contract(script)
    with pytest.raises(AssertionError):
        _assert_native_mps_smoke_contract(mutation(script))
