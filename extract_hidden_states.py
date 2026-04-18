from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_PROMPT = "The capital of France is"
LAYER_PATH = "model.model.language_model.layers"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and serialize the hidden state emitted by a specific "
            "Gemma 4 decoder layer for a prompt."
        )
    )
    parser.add_argument(
        "layer",
        type=int,
        help="Zero-based decoder layer index to capture.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to run through the model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Path to the serialized .pt output. Defaults to "
            "layer_<layer>_hidden_state.pt in the current directory."
        ),
    )
    return parser


def build_chat_prompt(processor: AutoProcessor, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def move_inputs_to_model_device(
    inputs: dict[str, torch.Tensor],
    model: AutoModelForCausalLM,
) -> dict[str, torch.Tensor]:
    input_device = next(model.parameters()).device
    return {key: value.to(input_device) for key, value in inputs.items()}


def resolve_output_path(layer_index: int, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return Path(f"layer_{layer_index}_hidden_state.pt")


def main() -> None:
    args = build_parser().parse_args()
    output_path = resolve_output_path(args.layer, args.output)

    print(f"Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"Loading model for {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    layers = model.model.language_model.layers
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(
            f"Layer index {args.layer} is out of range for {len(layers)} layers "
            f"at {LAYER_PATH}."
        )

    prompt = build_chat_prompt(processor, args.prompt)
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = move_inputs_to_model_device(inputs, model)

    captured_hidden_state: dict[str, torch.Tensor] = {}

    def capture_hidden_state(
        _module: torch.nn.Module,
        _module_inputs: tuple[Any, ...],
        module_output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        hidden_state = (
            module_output[0] if isinstance(module_output, tuple) else module_output
        )
        captured_hidden_state["tensor"] = hidden_state.detach().to("cpu")

    handle = layers[args.layer].register_forward_hook(capture_hidden_state)
    try:
        print(
            f"Running forward pass for layer {args.layer} with "
            f"{inputs['input_ids'].shape[-1]} prompt tokens..."
        )
        with torch.inference_mode():
            model(**inputs)
    finally:
        handle.remove()

    hidden_state = captured_hidden_state.get("tensor")
    if hidden_state is None:
        raise RuntimeError(f"Failed to capture hidden state for layer {args.layer}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "layer_index": args.layer,
        "layer_path": f"{LAYER_PATH}[{args.layer}]",
        "prompt_text": args.prompt,
        "prompt": prompt,
        "input_ids": inputs["input_ids"].detach().to("cpu"),
        "attention_mask": inputs.get("attention_mask", None),
        "hidden_state": hidden_state,
        "shape": tuple(hidden_state.shape),
        "dtype": str(hidden_state.dtype),
    }
    if isinstance(payload["attention_mask"], torch.Tensor):
        payload["attention_mask"] = payload["attention_mask"].detach().to("cpu")

    torch.save(payload, output_path)

    print(f"Captured hidden state shape: {tuple(hidden_state.shape)}")
    print(f"Serialized hidden state to {output_path}")


if __name__ == "__main__":
    main()
