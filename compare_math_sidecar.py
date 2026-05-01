"""Offline comparison helper for the math sidecar.

Run vLLM twice with the same prompt:

  1. Without the math sidecar (so `serialize_activations` returns raw layer
     tensors). Save the response JSON to e.g. `raw.json`.
  2. With the math sidecar enabled. Save to `math.json`.

Then run:

  uv run python compare_math_sidecar.py raw.json math.json

For each layer in `raw.json.choices[0].activations`, the script computes
`str(float(np.array(values).mean()))` and asserts it matches the matching
entry in `math.json.choices[0].activations.math_result`. Use `temperature: 0`
on both runs (which `test_vllm.sh` already does) so the prompt-token hidden
state is deterministic.

This is a test utility; the runtime path is patched serialize_activations +
math_sidecar.py.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def per_layer_means(raw_activations: dict[str, list[float]]) -> dict[str, str]:
    return {
        str(int(layer)): str(float(np.asarray(values, dtype=np.float32).mean()))
        for layer, values in raw_activations.items()
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <raw.json> <math.json>", file=sys.stderr)
        return 2

    raw = json.loads(Path(argv[1]).read_text())
    sidecar = json.loads(Path(argv[2]).read_text())

    raw_acts = raw["choices"][0]["activations"]
    side_acts = sidecar["choices"][0]["activations"]

    if "math_result" not in side_acts or side_acts["math_result"] is None:
        print(f"sidecar response has no math_result: {side_acts}", file=sys.stderr)
        return 1

    expected = per_layer_means(raw_acts)
    got = side_acts["math_result"]

    bad = 0
    for layer, exp in expected.items():
        actual = got.get(layer)
        if actual is None:
            print(f"layer {layer}: missing in sidecar response", file=sys.stderr)
            bad += 1
            continue
        if not math.isclose(float(exp), float(actual), rel_tol=1e-6, abs_tol=1e-7):
            print(
                f"layer {layer}: expected {exp} got {actual} "
                f"(diff={float(exp) - float(actual):.3e})",
                file=sys.stderr,
            )
            bad += 1
        else:
            print(f"layer {layer}: ok ({actual})")

    if bad:
        print(f"{bad} layers mismatched", file=sys.stderr)
        return 1
    print(f"all {len(expected)} layers match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
