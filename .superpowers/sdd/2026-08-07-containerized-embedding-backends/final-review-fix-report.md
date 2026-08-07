# Final Whole-Branch Review Fix Report

Date: 2026-08-07
Branch: `feature/retrieval-runtime`
Worktree: `/Users/thebat/IProject/justatom/.worktrees/retrieval-api-qwen`
Initial fix commit: `98b896e fix: close retrieval final review`
Re-review fix commit: `22ac4f5 fix: close final re-review blockers`

## Delivered

1. `HuggingFaceEmbedder` now owns a one-permit `asyncio.Lock` around all
   batched local inference for one call. Lifecycle acquisition remains outside
   the permit so close sees accepted queued work; queued cancellation reaches
   the existing `finally` release; started thread work remains shielded; one
   encoder instance and close-once behavior are unchanged.
2. Redis is now behind the explicit non-default `legacy` profile. Real
   launcher renders contain exactly `api,weaviate` for external and add only
   the selected CPU or CUDA embedder for managed modes.
3. `embedder-cuda` is pinned to `linux/amd64`. `cuda up` rejects non-Linux,
   non-`x86_64`/`amd64`, and missing or failing `nvidia-smi` before Compose.
   CUDA `config` and `build --print` remain available on macOS and render the
   amd64 platform. Documentation states Linux x86_64/amd64 and no CPU fallback.
4. CPU smoke cleanup now preflights its unique project and all ports, snapshots
   other Compose projects, treats launcher teardown as fatal, audits owned
   containers/volumes/networks and ports, and makes cleanup failure fatal after
   a successful main path while preserving an existing nonzero or signal
   status. Native MPS cleanup now performs bounded TERM, KILL fallback and
   reap, then audits server death, port release, and log removal under the same
   status precedence.
5. Dockerfile tests parse logical instructions and pin model, device, batch,
   max length, cache, user, exposed port, healthcheck, command, and the exact
   narrow COPY list while rejecting `COPY . .`. Parsed Compose tests pin exact
   service lists, aliases, cache/config/Weaviate/Redis mounts, read-only API
   config, MQ false, restart policies, exact NVIDIA driver/count/capabilities,
   legacy Redis topology, CUDA platform, and rendered named volumes.

## TDD Evidence

| Slice | RED | GREEN |
| --- | --- | --- |
| Inference serialization/lifecycle | `2 failed in 1.53s`: the second encoder call overlapped and queued cancellation timed out | complete embedder file: `8 passed in 0.48s` |
| Redis topology/CUDA platform | `5 failed in 2.22s`: Redis rendered in all modes, Linux arm64 reached Docker, platform missing | launcher plus focused Compose/docs: `46 passed in 8.20s` |
| CPU cleanup | `8 failed in 0.07s`: base contract and all mutations rejected the old best-effort cleanup | `10 passed in 0.01s`, plus Bash syntax |
| Native cleanup | `16 failed in 0.10s`: base contract and all lifecycle mutations rejected the old unbounded cleanup | `16 passed in 0.02s`, plus Bash syntax |
| Deferred Dockerfile/Compose gaps | `16 failed in 1.53s`: every reviewed mutation survived the intentionally incomplete helpers | full asset file: `60 passed in 1.71s` |

The combined changed-area gate passed `109 tests in 11.19s` (`real 12.46`,
`user 3.03`, `sys 1.41`). The concurrency RED run is also the direct mutation
check for removing the one-permit inference lock: it observed the second real
worker-thread encode enter while the first was blocked.

## Verification Matrix

| Command | Result | Timing / evidence |
| --- | --- | --- |
| `conda run -n justatom python -m pytest tests -q` | pass | final fresh run: `505 passed, 9 warnings in 30.42s`; `real 32.59`, `user 12.99`, `sys 3.73` |
| `conda run -n justatom make format-check` | pass | `171 files would be left unchanged`; `real 1.55` |
| `conda run -n justatom mkdocs build --strict` | pass | MkDocs build `0.15s`; `real 1.23` |
| Bash syntax for launcher and all three smokes | pass | no output |
| `git diff --check` and cached diff check | pass | no output before fix commit |
| `scripts/services.sh external config --format json` | pass | exact services `api,weaviate`; `real 0.04` |
| `scripts/services.sh cpu config --format json` | pass | exact services `api,embedder-cpu,weaviate`; `real 0.04` |
| `scripts/services.sh cuda config --format json` | pass | exact services `api,embedder-cuda,weaviate`; platform `linux/amd64`; `real 0.04` |
| `scripts/services.sh cuda build --print embedder-cuda` | pass | Bake platform `linux/amd64`; `real 0.05` |
| `scripts/services.sh cuda up -d` on macOS arm64 | expected rejection | status `2`: `CUDA mode requires a Linux host`; Compose was not invoked |

The full-suite warnings remain the existing optional TensorFlow/ParametricUMAP,
namespace-package, Weaviate-client, and Lightning environment/data-loader
warnings.

## Live Smokes

### Native MPS

Command:

```bash
/usr/bin/time -p conda run -n justatom bash scripts/smoke_native_embedding.sh
```

Result: pass in `real 16.14` (`user 10.21`, `sys 2.40`). Both UTF-8 embedding
calls, model identity, ordering, and cross-call dimensions passed with
`model_loads=1`. Cleanup proved the server stopped and reaped, port `18002`
free, and the generated log removed.

### CPU Qwen Retrieval

Command:

```bash
/usr/bin/time -p conda run -n justatom bash scripts/smoke_containerized_retrieval.sh
```

Result: pass in `real 234.74` (`user 1.77`, `sys 0.91`) using project
`justatom-smoke-1786127364-4938`. The current API and CPU images built, Qwen
loaded exactly once, three documents indexed, the retrieval query ranked
`retrieval` first, and the required second live query ranked `storage` first.
Readable Russian UTF-8 checks passed.

CPU teardown reported no owned containers, volumes, or networks; ports
`15555,18000,13211,15051` were free; and its label-complete pre-existing
Compose-project snapshot was unchanged. A separate post-audit found no
`justatom-smoke-*` or `justatom-api-smoke-*` resources and successfully bound
ports `15555,18000,13211,15051,18002`. `docker compose ls --all` was identical
before and after: `ci-pipelines` created, `clearml` running, `justatom` mixed
exited/running, `production-training` running, and `uniai` running.

## Residual Environment Gaps

- CUDA inference was not run: the host is Darwin arm64 and Docker runs Linux
  arm64. Per instruction, no slow emulated CUDA image build was attempted;
  Compose and Bake validation prove the requested `linux/amd64` target and the
  launcher proves macOS rejection.
- ShellCheck is not installed. Bash parsing and subprocess behavior are covered
  by the syntax and launcher tests.
- The prior branch-wide Docker Scout critical-CVE scan gap remains dependent on
  Docker Scout authentication; it was outside these final-review fixes and was
  not rerun.

## Final Re-review Fix Round

### Delivered

1. `_encode_batch` now drains its shielded `to_thread` task through any finite
   sequence of caller cancellations. It records the first cancellation,
   consumes worker completion or failure, and only then re-raises cancellation,
   keeping the inference permit and lifecycle count held until thread work is
   over. A three-cancellation regression proves `close()` stays pending and the
   encoder is not closed while active, followed by exactly one close.
2. CPU and external smoke scripts now source one Docker audit helper. Every
   container, volume, and network listing and every volume/network inspection
   captures and propagates its own status before any combined output is
   accepted. Executable fake-Docker tests exercise all 13 failure paths both
   directly and under literal `if !` condition context.
3. Only the 13 tests that execute real Compose config or Bake rendering in
   `tests/test_docker_assets.py` are marked `integration`; all 47 pure
   Dockerfile, YAML, and static mutation cases remain standard. CI now has a
   dedicated Ubuntu/Python 3.11 Docker Compose contract job. A characterization
   puts a status-97 fake Docker first on `PATH` and proves the standard asset
   slice never invokes it.

### TDD Evidence

| Slice | RED | GREEN |
| --- | --- | --- |
| Repeated cancellation | `1 failed in 0.47s`; the caller completed after its second cancellation while the worker was still blocked | full embedder file: `9 passed in 0.47s` |
| Docker audit propagation | `13 failed in 0.13s`; missing shared helper/source left every fake-Docker failure or static contract red | CPU/external audit gate: `18 passed in 4.48s`; final literal-`if !` focused set included below |
| Docker test classification/CI | `2 failed in 1.76s`; the fake-Docker run exposed 13 unmarked failures and the CI gate was absent | classification: `2 passed in 0.19s`; real Docker contracts: `13 passed, 47 deselected in 1.66s` |

The final combined regression slice passed `27 tests in 6.00s` (`real 7.21`,
`user 1.91`, `sys 0.65`). It includes all nine embedder lifecycle tests, all 16
fake-Docker/shared-helper tests, and both no-Docker/CI classification tests.

### Verification Matrix

| Command | Result | Timing / evidence |
| --- | --- | --- |
| Docker-hidden standard suite: `pytest tests -m "not integration and not network" -q` with `/usr/local/bin` absent from `PATH` | pass | `507 passed, 17 deselected, 6 warnings in 24.12s`; `real 25.33`, `user 9.93`, `sys 2.41`; `docker` was undiscoverable while Bash 5 remained available |
| Local Docker contract gate: `pytest tests/test_docker_assets.py -m integration -q` | pass | `13 passed, 47 deselected in 1.66s`; `real 2.72`, `user 2.02`, `sys 0.84` |
| Final fresh `pytest tests -q` | pass | `524 passed, 8 warnings in 35.02s`; `real 37.17`, `user 13.53`, `sys 4.23` |
| `make format-check` | pass | `173 files would be left unchanged`; `real 1.72`, `user 4.57`, `sys 1.18` |
| `python -m mkdocs build --strict` | pass | MkDocs build `0.18s`; `real 1.39`, `user 1.08`, `sys 0.27` |
| Bash syntax for every tracked `*.sh` | pass | no output |
| Working and staged `git diff --check` | pass | no output before implementation commit |
| external/cpu/cuda launcher configs | pass | exact services `api,weaviate`; `api,embedder-cpu,weaviate`; and `api,embedder-cuda,weaviate`, each in `real 0.04` |
| CUDA Compose/Bake platform | pass | config `linux/amd64` in `real 0.04`; Bake `[linux/amd64]` in `real 0.06` |
| macOS `cuda up -d` | expected rejection | status `2` before Compose in `real 0.00`: Linux host required |

### Smoke and Cleanup Evidence

Per the re-review instruction, the CPU Qwen and native MPS model smokes were
not rerun. Their successful live outcomes and cleanup evidence remain recorded
in the preceding **Live Smokes** section. This round covers the production
inference change with deterministic blocking-thread lifecycle tests and covers
both smoke-audit changes with executable fake-Docker subprocess tests.

Post-verification audit found no `justatom-smoke-*` or
`justatom-api-smoke-*` Compose projects. Ports
`15555,18000,13211,15051,15556,13212,15052,18001,18002` all accepted a fresh
loopback bind. `docker compose ls --all` still matched the earlier snapshot:
`ci-pipelines` created, `clearml` running, `justatom` mixed exited/running,
`production-training` running, and `uniai` running.

### Residual Environment Gaps

- CUDA inference remains unverified on this Darwin arm64 host; real Compose and
  Bake contracts passed and `cuda up` rejected the host before Compose.
- CPU Qwen and native MPS live smokes were intentionally not repeated in this
  round, as requested; the prior successful runs remain documented above.
- ShellCheck remains unavailable, and Docker Scout remains unauthenticated.
  Bash parsing/subprocess behavior and all requested local Docker contracts
  passed.
