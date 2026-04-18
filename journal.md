# Journal

## Python 3.12 and vLLM

- Constrained the project to Python `3.12.x` with `requires-python = ">=3.12,<3.13"`.
- Updated `tool.uv` to use `prerelease = "allow"` and `index-strategy = "unsafe-best-match"`.
- Pointed `uv` at the CUDA 13 nightly `vllm` index and refreshed the lockfile and environment.
- Removed `.venv` and rebuilt it with `uv sync --no-cache` to avoid using cached packages.
- Verified with both `./.venv/bin/python` and `uv run python`.
- Installed version: `0.19.1rc1.dev386+g55842a8d6`

## Gemma 4 Layer Sweep

- Confirmed the Gemma 4 text decoder stack is exposed at `model.model.language_model.layers`.
- Recorded `30` decoder layers in `layers.md`.
- Verified all `30` recorded layer entries share the same `Gemma4TextDecoderLayer.forward` signature and source (`mismatches=0`).

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    per_layer_input: torch.Tensor = None,
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    position_embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    **kwargs,
) -> torch.Tensor:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        shared_kv_states=shared_kv_states,
        position_ids=position_ids,
        past_key_values=past_key_values,
        **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)

    if self.enable_moe_block:
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        # Take hidden states before MLP here
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        _, top_k_weights, top_k_index = self.router(hidden_states_flat)
        hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
        hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        # Combine mlp and moe outputs
        hidden_states = hidden_states_1 + hidden_states_2

    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    if self.hidden_size_per_layer_input:
        residual = hidden_states
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states *= self.layer_scalar
    return hidden_states
```

## Gemma 4 vLLM Activation Extraction

- Treated the existing `custom-patches/*.patch` files as `vLLM 0.18` design references only and implemented the Gemma 4 support directly against the installed `vLLM 0.19.1rc1.dev386+g55842a8d6`.
- Initialized a nested git repository in `.venv/lib/python3.12/site-packages/vllm` to track the installed-package edits; baseline commit: `ed87373`.
- Added public `extract_activation_layers` plumbing through `ModelConfig`, `EngineArgs`, and `LLM`, then propagated prompt activations through the V1 engine and OpenAI-compatible response objects.
- For Gemma 4, aligned activation capture to decoder-layer outputs `0..29` on the non-fast-prefill path by capturing the post-layer hidden state using `self.start_layer + layer_idx`.
- Explicitly rejected Gemma 4 activation extraction when fast prefill is enabled. The verified path is eager / non-fast-prefill execution.
- Reviewed the older Nemotron and Gemma 3 patches for missed optimizations. They only add inline aux-hidden-state capture in a normal decoder loop; they do not preserve a special fast-prefill path.
- Gemma 4 differs because its fast-prefill path is structurally split into `self_decoder` and `cross_decoder` wrappers, both calling `_run_decoder_layers(...)`, and that helper only returns the final hidden state rather than intermediate per-layer outputs.
- Gemma 4 fast prefill also reduces the token set passed through the cross-decoder using `logits_indices_padded` / `num_padded`, so exact full-prompt activation reconstruction for all layers is not obviously available from the current optimized path.
- Kept the vLLM-side activation tensor shape as `(prompt_tokens, hidden_size)` and handled the Hugging Face reference batch dimension in `compare_vllm_hidden_states.py` by adding `unsqueeze(0)` during comparison only.
- Verified against `all_layers_hidden_states.npz` generated by `extract_all_hidden_states_npz.py` with `uv run python compare_vllm_hidden_states.py`.
- Verification result: all `30` layers were returned, each compared at shape `(1, 18, 2816)`, and the minimum cosine similarity was `0.99858379` on layer `29`.
- Exported the installed-package diff as `custom-patches/vllm-0.19rc1-gemma4-hidden-states.patch` and updated `custom-patches/apply-patch.sh` to prefer the `vllm-0.19*.patch` path over the older numbered patches.


### TBC
In this part: Gemma 4 differs because its fast-prefill path is structurally split into self_decoder and
 cross_decoder wrappers, both calling _run_decoder_layers(...), and that helper only returns the final hidden state
 rather than intermediate per-layer outputs.
 isn't that what we want the final hidden state?


• Not in this case.

 What we want is:

 - the final hidden state of each decoder layer
 - for all requested layers 0..29
 - over the prompt tokens

 _run_decoder_layers(...) currently returns only:

 - the final hidden state after the whole block of layers it ran

 So if it runs layers 0..14, you only get the output of layer 14, not layers 0..13.
 If it runs layers 15..29, you only get the output of layer 29, not layers 15..28.

 That is enough for:

 - normal inference
 - logits computation
 - maybe a single terminal representation

 But it is not enough for our use case, which is:

 - extract all layer hidden states
 - compare each layer against the HF reference NPZ

 So the issue is not “it returns a final hidden state” versus “we want a final hidden state.”
 The issue is:

 - it returns only one final hidden state for the whole decoder segment
 - we need one final hidden state per layer

 That is why the old Gemma 3 / Nemotron style loop works better for extraction: inside the loop, after each layer, you
 can capture that layer’s final output.

 For Gemma 4 fast-prefill, there is a second complication:

 - the cross_decoder path may only run on a reduced token subset

 So even if we modified _run_decoder_layers(...) to emit per-layer outputs, we would still need to check whether those
 outputs exist for all prompt tokens in the cross-decoder half.

 So your intuition is right that “final hidden state” is the right kind of tensor. The missing part is that we need
 the final hidden state after each layer, not just after the entire wrapper finishes.
 ### 
