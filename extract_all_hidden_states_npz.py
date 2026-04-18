from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_ID = "google/gemma-4-26B-A4B-it"
PROMPT = "The capital of France is"
LAYER_PATH = "model.model.language_model.layers"
OUTPUT_PATH = Path("all_layers_hidden_states.npz")


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


def main() -> None:
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
    prompt = build_chat_prompt(processor, PROMPT)
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = move_inputs_to_model_device(inputs, model)

    captured_hidden_states: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_capture_hidden_state(layer_index: int):
        def capture_hidden_state(
            _module: torch.nn.Module,
            _module_inputs: tuple[Any, ...],
            module_output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            hidden_state = (
                module_output[0] if isinstance(module_output, tuple) else module_output
            )
            # Save each layer output on CPU as float32 so it can be written to NPZ
            # portably. The stacked array has shape:
            # (num_layers, batch_size, sequence_length, hidden_size).
            # For the default prompt here that is expected to be:
            # (30, 1, 18, 2816).
            captured_hidden_states[layer_index] = (
                hidden_state.detach().to(device="cpu", dtype=torch.float32)
            )

        return capture_hidden_state

    for layer_index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_capture_hidden_state(layer_index)))

    try:
        print(
            f"Running forward pass across {len(layers)} layers with "
            f"{inputs['input_ids'].shape[-1]} prompt tokens..."
        )
        with torch.inference_mode():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    if len(captured_hidden_states) != len(layers):
        raise RuntimeError(
            f"Captured {len(captured_hidden_states)} layer outputs for {len(layers)} "
            "decoder layers."
        )

    hidden_states = torch.stack(
        [captured_hidden_states[layer_index] for layer_index in range(len(layers))],
        dim=0,
    ).numpy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        hidden_states=hidden_states,
        input_ids=inputs["input_ids"].detach().to("cpu").numpy(),
        attention_mask=inputs["attention_mask"].detach().to("cpu").numpy(),
        layer_indices=np.arange(len(layers), dtype=np.int64),
        model_id=np.array(MODEL_ID),
        layer_path=np.array(LAYER_PATH),
        prompt_text=np.array(PROMPT),
        prompt=np.array(prompt),
    )

    print(f"Captured hidden states array shape: {hidden_states.shape}")
    print(
        "Shape meaning: "
        "(num_layers, batch_size, sequence_length, hidden_size)"
    )
    print(f"Serialized all layer hidden states to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
