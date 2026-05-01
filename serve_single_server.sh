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

export MATH_SIDECAR_IPC="${MATH_SIDECAR_IPC:-ipc:///tmp/gemma4-math.sock}"
LAYERS="${LAYERS:-0 14 29}"

SOCKET_PATH="${MATH_SIDECAR_IPC#ipc://}"
rm -f "$SOCKET_PATH"

uv run python math_sidecar.py &
SIDECAR_PID=$!

cleanup() {
  if kill -0 "$SIDECAR_PID" 2>/dev/null; then
    kill -TERM "$SIDECAR_PID" 2>/dev/null || true
    wait "$SIDECAR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_sidecar() {
  MATH_SIDECAR_IPC="$MATH_SIDECAR_IPC" uv run python - <<'PY'
import os
import sys
import time

import msgpack
import zmq

endpoint = os.environ["MATH_SIDECAR_IPC"]
ctx = zmq.Context.instance()
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.setsockopt(zmq.SNDTIMEO, 200)
    try:
        sock.connect(endpoint)
        sock.send(msgpack.packb({"ping": True}, use_bin_type=True))
        reply = msgpack.unpackb(sock.recv(), raw=False)
        if reply.get("pong"):
            sys.exit(0)
    except zmq.error.ZMQError:
        pass
    finally:
        sock.close(linger=0)
    time.sleep(0.1)
sys.exit(1)
PY
}

if ! wait_for_sidecar; then
  echo "math sidecar did not become ready on $MATH_SIDECAR_IPC" >&2
  exit 1
fi

exec vllm serve google/gemma-4-26B-A4B-it \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --extract-activation-layers $LAYERS
