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
