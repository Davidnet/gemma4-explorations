"""POC math sidecar for the Gemma 4 single-server setup.

Receives per-layer last-prompt-token activations from the patched vLLM over a
ZeroMQ REQ/REP IPC socket and replies with one stringified scalar per layer.

The "math" for the POC is just the per-layer mean. The real CPU model can
replace `kernel(...)` later without changing the wire protocol.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

import msgpack
import numpy as np
import zmq

DEFAULT_ENDPOINT = "ipc:///tmp/gemma4-math.sock"
LOGGER = logging.getLogger("math_sidecar")


def kernel(layer_idx: int, tensor: np.ndarray) -> str:
    """POC kernel: per-layer mean as a string. Swap for the real model later."""
    return str(float(tensor.mean()))


def _decode_tensor(payload: dict) -> np.ndarray:
    shape = tuple(payload["shape"])
    dtype = np.dtype(payload["dtype"])
    return np.frombuffer(payload["data"], dtype=dtype).reshape(shape)


def _handle(request: dict) -> dict:
    if request.get("ping"):
        return {"pong": True}

    req_id = request.get("req_id", "")
    activations = request.get("activations") or {}

    try:
        result: dict[str, str] = {}
        for layer_idx, payload in activations.items():
            tensor = _decode_tensor(payload)
            result[str(int(layer_idx))] = kernel(int(layer_idx), tensor)
    except Exception as exc:  # noqa: BLE001 - report any decode/kernel failure
        LOGGER.exception("kernel failed for req_id=%s", req_id)
        return {"req_id": req_id, "error": f"{type(exc).__name__}: {exc}"}

    return {"req_id": req_id, "math_result": result}


def _unlink_ipc_path(endpoint: str) -> None:
    if not endpoint.startswith("ipc://"):
        return
    path = Path(endpoint[len("ipc://") :])
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            LOGGER.warning("failed to unlink stale socket %s: %s", path, exc)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    endpoint = os.environ.get("MATH_SIDECAR_IPC", DEFAULT_ENDPOINT)
    _unlink_ipc_path(endpoint)

    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REP)
    socket.bind(endpoint)
    LOGGER.info("math sidecar bound to %s", endpoint)
    LOGGER.info("math sidecar ready (kernel=mean)")

    stop = False

    def _shutdown(signum, _frame):
        nonlocal stop
        LOGGER.info("received signal %s, shutting down", signum)
        stop = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    try:
        while not stop:
            events = dict(poller.poll(timeout=500))
            if socket not in events:
                continue
            raw = socket.recv()
            try:
                request = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("failed to unpack request")
                socket.send(msgpack.packb({"error": f"unpack: {exc}"}, use_bin_type=True))
                continue
            reply = _handle(request)
            socket.send(msgpack.packb(reply, use_bin_type=True))
    finally:
        socket.close(linger=0)
        ctx.term()
        _unlink_ipc_path(endpoint)
        LOGGER.info("math sidecar stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
