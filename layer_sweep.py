from __future__ import annotations

import inspect
from pathlib import Path

from transformers import AutoModelForCausalLM


MODEL_ID = "google/gemma-4-26B-A4B-it"
OUTPUT_PATH = Path("layers.md")
LAYER_PATH = "model.model.language_model.layers"


def render_markdown(layer_path: str, layer_count: int, layers) -> str:
    lines = [
        "# Layer Forward Sweep",
        "",
        f"- Model: `{MODEL_ID}`",
        f"- Resolved layer path: `{layer_path}`",
        f"- Total layers: `{layer_count}`",
        "",
    ]

    for layer_index, block in enumerate(layers):
        lines.extend(
            [
                f"## Layer {layer_index}",
                "",
                f"- Block type: `{type(block).__name__}`",
                f"- Forward signature: `{inspect.signature(block.forward)}`",
                "",
                "```python",
                inspect.getsource(block.forward).rstrip(),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    print(f"Loading model for {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    layer_path = LAYER_PATH
    # Gemma 4 exposes the text decoder blocks under the wrapped language model.
    layers = model.model.language_model.layers
    print(f"Recording forward methods for {len(layers)} layers...")

    OUTPUT_PATH.write_text(
        render_markdown(layer_path, len(layers), layers),
        encoding="utf-8",
    )
    print(f"Wrote layer sweep to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
