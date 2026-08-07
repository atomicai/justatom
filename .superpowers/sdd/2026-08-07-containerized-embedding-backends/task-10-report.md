# Task 10 Report: User Documentation and Final Verification

Date: 2026-08-07
Worktree: `/Users/thebat/IProject/justatom/.worktrees/retrieval-api-qwen`
Branch: `feature/retrieval-runtime`

## Delivered

- Documented the launcher-only CPU, CUDA, external, and native MPS flows in
  `README.md` and `docs/launch-guide.md`.
- Added the API/embedder/Weaviate process edges and prefix/batching ownership
  to `docs/architecture.md` and `docs/modules/runtime.md`.
- Added executable `scripts/smoke_native_embedding.sh`. It skips outside Apple
  Silicon macOS, uses `EMBEDDING_PORT` (default `18002`), validates MPS,
  readiness, model metadata, two bounded UTF-8 embedding requests, top-level
  response model identity, ordering, cross-call dimensions, one model load,
  and signal-safe PID/log cleanup.
- Added mutation-sensitive documentation and native-smoke assertions to
  `tests/test_docker_assets.py`.

## TDD Evidence

RED:

```bash
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_deployment_docs_use_the_launcher_and_describe_runtime_boundaries \
  tests/test_docker_assets.py::test_native_mps_smoke_has_a_bounded_host_only_lifecycle_and_contract_checks -q
```

Result: `2 failed in 0.05s`: launcher documentation was absent and the native
script did not exist.

GREEN after the scoped edits:

```bash
bash -n scripts/smoke_native_embedding.sh
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_deployment_docs_use_the_launcher_and_describe_runtime_boundaries \
  tests/test_docker_assets.py::test_native_mps_smoke_has_a_bounded_host_only_lifecycle_and_contract_checks -q
```

Result: `2 passed in 0.01s`.

### Review-fix RED/GREEN

The review-fix static contract added a finite-loop assertion, a port-preflight
and error-path assertion, status-preserving `EXIT`/`INT`/`TERM` cleanup,
request and response UTF-8 guards, both response model assertions, and the
second-call dimension comparison. It also mutates the source in memory to
remove the bind preflight, replace the loop with `while true`, force `exit 0`,
remove all UTF-8 guards, remove a response model selector, remove the
cross-call dimension argument, and add `-p demo` to the guide.

RED command:

```bash
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_deployment_docs_use_the_launcher_and_describe_runtime_boundaries \
  tests/test_docker_assets.py::test_deployment_docs_contract_rejects_manual_compose_project_selection \
  tests/test_docker_assets.py::test_native_mps_smoke_has_a_bounded_host_only_lifecycle_and_contract_checks \
  tests/test_docker_assets.py::test_native_mps_smoke_contract_rejects_safety_mutations -q
```

Result: `2 failed, 7 passed in 0.05s`, because the CUDA no-fallback statement
and both response-model checks were absent.

### Re-review Fix Round 2

The documentation mutation test is parameterized for both `-p demo` and
`-p=demo`. The native port-preflight contract now captures the Python heredoc
and requires its loopback `listener.bind(("127.0.0.1", int(sys.argv[1])))`
statement. Its mutation changes that actual bind to `0.0.0.0`, which the same
helper rejects.

Command:

```bash
conda run -n justatom python -m pytest \
  tests/test_docker_assets.py::test_deployment_docs_use_the_launcher_and_describe_runtime_boundaries \
  tests/test_docker_assets.py::test_deployment_docs_contract_rejects_manual_compose_project_selection \
  tests/test_docker_assets.py::test_native_mps_smoke_has_a_bounded_host_only_lifecycle_and_contract_checks \
  tests/test_docker_assets.py::test_native_mps_smoke_contract_rejects_safety_mutations -q
```

Result: `12 passed in 0.04s`.

## Verification Matrix

| Command | Result | Timing / evidence |
| --- | --- | --- |
| `conda run -n justatom python -m pytest tests/test_docker_assets.py tests/test_services_launcher.py -q` | pass | `64 passed in 7.60s` (`real 8.70s`) |
| `conda run -n justatom python -m pytest tests -q` | pass | `464 passed, 9 warnings in 26.11s` (`real 28.29s`) |
| `conda run -n justatom make format-check` | pass | after `conda run -n justatom make fix-format` reformatted all seven branch-owned Python files with outstanding Black/isort changes |
| `conda run -n justatom mkdocs build --strict` | pass | `real 1.33s` |
| `bash -n` for the launcher and all three smoke scripts | pass | no output |
| `git diff --check` | pass | no output |
| `scripts/services.sh external config --quiet` | pass | `real 0.04s` |
| `scripts/services.sh cpu config --quiet` | pass | `real 0.04s` |
| `scripts/services.sh cuda config --quiet` | pass | `real 0.05s` |
| `scripts/services.sh cuda up -d --build` | skipped as designed | macOS preflight rejected it in `0.00s`: Linux host required |
| `docker build -f Dockerfile.api -t justatom-api:verify .` | pass | fresh `linux/arm64` image created at `2026-08-07T17:20:17Z`; build output was truncated by the harness |
| `docker build -f Dockerfile.embedder.cpu -t justatom-embedder-cpu:verify .` | pass | `sha256:6d965...`, `real 16.54s` |
| `docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.api .` | attempted | BuildKit traversed both target stages; the harness truncated its final completion line, so no multi-platform image was loaded locally |
| `scripts/services.sh cuda build embedder-cuda` | fail, recorded | macOS arm64 build reached the CUDA wheel index, which has no arm64 `torch==2.8.0+cu128` candidate |
| `docker build --platform linux/amd64 -f Dockerfile.embedder.cuda -t justatom-embedder-cuda:verify .` | timed out, recorded | stopped after 10 minutes with no image artifact; no CUDA inference was attempted |
| `conda run -n justatom bash scripts/smoke_native_embedding.sh` | pass | both response models and equal cross-call dimensions verified; `model_loads=1`, `real 14.62s` |
| `conda run -n justatom bash scripts/smoke_api_external_backend.sh` | pass | API contains no Torch, deterministic Russian indexing/search/UTF-8 checks and isolation cleanup passed; `real 24.46s` |
| `conda run -n justatom bash scripts/smoke_containerized_retrieval.sh` | evidence partial | Qwen loaded once and CPU server reached `Running on http://0.0.0.0:8000`; the tool lost the command transcript after the process exited. Direct audits found no smoke-labeled resources or smoke ports. |

The full suite warnings are existing optional TensorFlow/ParametricUMAP,
namespace-package, Weaviate-client, and Lightning environment/dataloader
warnings. Formatting was branch-owned final-verification debt: the Makefile
formatter corrected the seven affected files, and the required format-check
now passes.

## Image and Security Evidence

Fresh image inspection:

```text
justatom-api:verify            227,975,533 bytes  linux/arm64
justatom-embedder-cpu:verify   321,484,051 bytes  linux/arm64
justatom-embedder-cuda:test  4,156,153,982 bytes  linux/amd64 (prior verified artifact)
```

These are `docker image inspect` content-size values. Separately, `docker image
ls` reports Docker's virtual image sizes: `1.04GB` for the API,
`1.6GB` for the CPU embedder, and `4.16GB` for the prior CUDA artifact. The
two measurements are distinct Docker metrics; the CLI values are not rounded
forms of the inspect values.

```bash
docker run --rm justatom-api:verify python -c \
  'import importlib.util; assert importlib.util.find_spec("torch") is None'
docker run --rm justatom-embedder-cpu:verify python -m pip check
```

Both passed; the CPU image reported `No broken requirements found.` A
`docker history --no-trunc` scan of both fresh images found no API key, Hugging
Face token, `.env`, `/Users/thebat` local path, weight format, or model-weight
layer markers. Image histories contain the expected source copies and package
installation layers only.

Docker Scout `1.20.4` is installed but both critical-CVE invocations returned
status `1` because Scout requires Docker authentication. ShellCheck is not
installed (`command -v shellcheck` status `1`).

## Platform and Smoke Notes

- Native MPS was executed on macOS arm64. It used port `18002`, polled health
  and models, accepted two UTF-8 requests and responses, verified both
  top-level response models, ordered non-empty equal dimensions within the
  first call and against the second call, and found exactly one loader message.
- CUDA inference is skipped: this is macOS arm64, not a Linux/NVIDIA runner.
  The launcher rejected `cuda up` before it could invoke a workload.
- The external smoke used only its unique project and ports `15556`, `13212`,
  `15052`, and `18001`; cleanup reported no resources, free ports, and an
  unchanged pre-existing Compose-project snapshot.
- The first CPU smoke used only its unique project and ports `15555`, `18000`,
  `13211`, and `15051`. It logged exactly one Qwen load and server startup;
  post-exit audits found no project resources or ports. The result transcript
  is the only incomplete evidence item because the local tool dropped its live
  handle.

## Concerns

1. The current CUDA package index lacks the pinned arm64 CUDA wheel and the
   explicit amd64 emulated build stalled beyond ten minutes. This must be
   rechecked on a Linux/NVIDIA runner; do not interpret either macOS result as
   CUDA inference coverage.
2. Docker Scout requires authentication in this environment.
