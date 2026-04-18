import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "google/gemma-4-26B-A4B-it"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 256


def main() -> None:
    print(f"Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"Loading model for {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    print("Model and processor loaded successfully.")
    print(f"Processor type: {type(processor).__name__}")
    print(f"Model type: {type(model).__name__}")

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": PROMPT}],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(text=prompt, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    print("\nGenerating response...\n")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = generated[0][prompt_length:]
    response = processor.decode(new_tokens, skip_special_tokens=True)
    print(response.strip())


if __name__ == "__main__":
    main()
