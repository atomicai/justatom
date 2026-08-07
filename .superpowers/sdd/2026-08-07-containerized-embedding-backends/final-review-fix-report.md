# Final Whole-Branch Review Fix Report

Date: 2026-08-07
Branch: `feature/retrieval-runtime`
Worktree: `/Users/thebat/IProject/justatom/.worktrees/retrieval-api-qwen`
Fix commit: `98b896e fix: close retrieval final review`

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
