#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/services.sh <external|cpu|cuda> <compose-command> [compose args...]

Examples:
  scripts/services.sh cpu up -d --build
  EMBEDDING_BASE_URL=http://host.docker.internal:8000/v1 scripts/services.sh external up -d
  scripts/services.sh cuda config
EOF
}

if (( $# < 2 )); then
  usage >&2
  exit 2
fi

mode="$1"
shift

case "$mode" in
  external|cpu|cuda) ;;
  *)
    echo "expected exactly one embedding mode: external, cpu, or cuda" >&2
    exit 2
    ;;
esac

for arg in "$@"; do
  case "$arg" in
    --profile|--profile=*)
      echo "--profile is not supported; select the embedding mode with the first argument" >&2
      exit 2
      ;;
    external|cpu|cuda)
      echo "expected exactly one embedding mode; select it only with the first argument" >&2
      exit 2
      ;;
  esac
done

compose_command="$1"
export COMPOSE_PROFILES="$mode"

if [[ "$mode" == "external" ]]; then
  case "$compose_command" in
    up|create|start|restart|run|exec|watch)
      if [[ -z "${EMBEDDING_BASE_URL:-}" ]]; then
        echo "EMBEDDING_BASE_URL is required for external mode with '$compose_command'" >&2
        exit 2
      fi
      ;;
  esac
fi

if [[ "$mode" == "cuda" && "$compose_command" == "up" ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "CUDA mode requires a Linux host; use 'cuda config' or 'cuda build' for validation only" >&2
    exit 2
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi must be available and succeed before CUDA services can start" >&2
    exit 2
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

exec docker compose "$@"
