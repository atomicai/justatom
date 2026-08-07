# Task 9 Report: Model-free API Image Smoke Test

Date: 2026-08-07
Worktree: `/Users/thebat/IProject/justatom/.worktrees/retrieval-api-qwen`
Branch: `feature/retrieval-runtime`

## Scope

Created a deterministic host-side OpenAI-compatible embedding stub and an
external-backend API image smoke. The smoke starts only a uniquely named,
external-mode Weaviate/API project through `scripts/services.sh`, leaves model
ownership on the host fixture, indexes three Russian documents, verifies two
deterministic vector rankings and readable UTF-8, and checks that the running
API container cannot import Torch.

Files changed:

- Created `tests/fixtures/openai_embedding_stub.py`.
- Created executable `scripts/smoke_api_external_backend.sh`.
- Modified `tests/test_docker_assets.py` with mutation-sensitive static
  coverage for the launcher boundary, project/port isolation, stub endpoints,
  scoped import path, startup liveness, rankings, UTF-8, and Torch absence.

## RED/GREEN Evidence

### Initial missing artifacts

RED command:

```bash
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_external_backend_smoke_uses_real_api_image_without_torch -q
```

RED output:

```text
FAILED ... FileNotFoundError: scripts/smoke_api_external_backend.sh
1 failed in 0.03s
```

After adding the stub, smoke script, and static assertions:

```bash
chmod +x scripts/smoke_api_external_backend.sh
bash -n scripts/smoke_api_external_backend.sh
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_external_backend_smoke_uses_real_api_image_without_torch -q
```

GREEN output:

```text
1 passed in 0.01s
```

### Fixture import path and fast liveness failure

The first fixture-backed run failed because executing
`python tests/fixtures/openai_embedding_stub.py` puts `tests/fixtures`, not the
repository root, on `sys.path`:

```text
ModuleNotFoundError: No module named 'justatom'
```

That run took `155.45s` because the original readiness loop waited for an
already-exited child. Its cleanup trap reported zero smoke resources and all
selected ports free.

Added the following static requirements before the fix:

```python
assert 'PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"' in script
assert 'kill -0 "$FAKE_PID"' in script
```

RED output:

```text
FAILED ... AssertionError
1 failed in 0.02s
```

The fixture is now launched with a process-scoped worktree `PYTHONPATH`; the
fake-endpoint waiter checks `kill -0 "$FAKE_PID"` before every request and
prints the fake-server log immediately when it has exited. The focused test and
shell syntax check then passed:

```text
1 passed in 0.01s
```

### Quart route configuration

The next run failed in `2.45s`, immediately because the liveness check detected
the child exit. The fixture log showed:

```text
KeyError: 'PROVIDE_AUTOMATIC_OPTIONS'
```

This Quart version requires the setting before registering routes. The retrieval
app already establishes it, so the stub now does the same. A new static
assertion was added first:

```python
assert 'app.config.setdefault("PROVIDE_AUTOMATIC_OPTIONS", True)' in fixture
```

RED output:

```text
FAILED ... AssertionError
1 failed in 0.02s
```

After adding the setting, the focused test and `bash -n` were GREEN:

```text
1 passed in 0.01s
```

## Real Smoke

Command:

```bash
/usr/bin/time -p conda run -n justatom bash scripts/smoke_api_external_backend.sh
```

The `justatom` environment is required because the host-side fixture imports
Quart and current-worktree JustAtom code. Successful output included:

```text
model-free API smoke passed: project=justatom-api-smoke-1786120880-76267
cleanup evidence: project=justatom-api-smoke-1786120880-76267 containers/volumes/networks=none
cleanup evidence: ports free=15556,13212,15052,18001
real 19.20
user 1.54
sys 0.56
```

The successful run used only these lifecycle commands:

```bash
scripts/services.sh external up -d --build weaviate api
scripts/services.sh external logs --no-color ...
scripts/services.sh external down -v --remove-orphans
```

It did not invoke raw Compose, select a profile directly, or pass a Compose
project option. It exported `COMPOSE_PROJECT_NAME` as
`justatom-api-smoke-1786120880-76267` and set high, overridable ports
`15556`, `13212`, `15052`, and `18001`.

Inline smoke assertions passed:

- Host stub exposes `/health`, `/v1/models`, and `/v1/embeddings` for
  `fixture-embedding-model`.
- API uses `http://host.docker.internal:18001/v1` and indexes exactly three
  Russian documents.
- `банк негативов` ranks the `retrieval` document first; the storage query
  ranks the `storage` document first.
- The returned Russian document body is readable UTF-8 with no `\\uNNNN`
  escaping.
- `importlib.util.find_spec("torch") is None` succeeds in the running API
  container.

No local Qwen/embedder image or model download was started by this task.

## Isolation and Cleanup Evidence

Before the successful smoke, the discovered Compose project labels were:

```text
ci-pipelines
clearml
deploy
justatom
production-training
uniai
```

All smoke ports were free before startup:

```text
15556,13212,15052,18001
```

After the smoke, an independent direct-Docker label audit returned no IDs for
the smoke project in all three classes:

```text
docker ps -aq --filter label=com.docker.compose.project=justatom-api-smoke-1786120880-76267
docker volume ls -q --filter label=com.docker.compose.project=justatom-api-smoke-1786120880-76267
docker network ls -q --filter label=com.docker.compose.project=justatom-api-smoke-1786120880-76267
```

The pre-existing project-label list was identical after teardown. A post-smoke
bind check also returned:

```text
15556: free
13212: free
15052: free
18001: free
```

The two failed pre-success smoke projects, `justatom-api-smoke-1786120616-75104`
and `justatom-api-smoke-1786120826-76080`, also emitted cleanup evidence with
no remaining containers, volumes, networks, or occupied smoke ports.

## Verification

Focused asset test and syntax:

```bash
bash -n scripts/smoke_api_external_backend.sh
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_external_backend_smoke_uses_real_api_image_without_torch -q
```

Output: `1 passed in 0.01s`.

Full suite:

```bash
/usr/bin/time -p conda run -n justatom python -m pytest -q
```

Output:

```text
448 passed, 9 warnings in 26.05s
real 28.31
user 11.91
sys 2.88
```

The nine warnings are the existing TensorFlow/ParametricUMAP availability,
namespace deprecation, Weaviate-client deprecation, and Lightning environment
or data-loader warnings.

`git diff --check` passed with no output.

## Concerns

- Direct `bash scripts/smoke_api_external_backend.sh` requires an already
  activated Python environment with Quart and Hypercorn. The documented and
  verified invocation is `conda run -n justatom bash ...`.
- Docker Compose emits its existing warning that the top-level `version`
  attribute is obsolete. Task 9 did not modify Compose.
