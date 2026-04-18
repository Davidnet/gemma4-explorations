#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ENV_FILE"
  set +a
fi

: "${HF_TOKEN:?HF_TOKEN must be set in .env or the environment.}"

vllm serve google/gemma-4-26B-A4B-it \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
