# Local vLLM patch workflow

## Requirements
Install `vllm` into this project's virtual environment, for example:

```bash
uv pip install vllm --torch-backend=cu128
```

This repository applies local patches to the installed package at:

`.venv/lib/python3.12/site-packages/vllm`

Two patch sets are maintained, one per supported vLLM minor release.
`apply-patch.sh` reads the installed dist-info name and picks the right set
automatically — no manual selection needed.

## Patch files

For vLLM **0.20** (current; verified against `0.20.1rc1.dev136+g529c671e8`):

- `custom-patches/vllm-0.20-gemma4-hidden-states.patch`
- `custom-patches/vllm-0.20-math-sidecar.patch`

For vLLM **0.19.1rc1** (verified against `0.19.1rc1.dev386+g55842a8d6`):

- `custom-patches/vllm-0.19rc1-gemma4-hidden-states.patch`
- `custom-patches/vllm-0.19rc1-math-sidecar.patch`

## Pinning a vLLM commit by URL

The rolling `wheels.vllm.ai/nightly/cu130` index only retains the latest commit,
so depending on it lets `uv sync` walk past the version a patch set was tested
against. Pin instead by direct wheel URL — `wheels.vllm.ai` keeps per-commit
subdirectories indefinitely:

```
s3://vllm-wheels/<full-40-char-commit-sha>/
  ├── vllm-*.whl                # wheels live at this level
  ├── vllm/index.html           # PEP-503 default-variant index
  ├── cu129/vllm/index.html     # cu129-variant index (points back to root wheels)
  ├── cu130/vllm/index.html     # cu130-variant index
  └── cpu/vllm/index.html       # cpu variant
```

Resolve a short hash from a build like `0.20.1rc1.dev136+g529c671e8` to its
full SHA via the GitHub API:

```bash
curl -sS https://api.github.com/repos/vllm-project/vllm/commits/529c671e8 \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"
# -> 529c671e8075d265a48b72e0eaaeb5e30d2f1630
```

The 0.20 pin currently in `pyproject.toml` uses this pattern (the `+` in the
version must be URL-encoded as `%2B`):

```toml
"vllm @ https://wheels.vllm.ai/529c671e8075d265a48b72e0eaaeb5e30d2f1630/vllm-0.20.1rc1.dev136%2Bg529c671e8-cp38-abi3-manylinux_2_34_x86_64.whl ; sys_platform == 'linux' and platform_machine == 'x86_64'"
```

For a one-shot install without editing `pyproject.toml`:

```bash
uv pip install --reinstall <wheel-url>
```

The same pattern works for the 0.19 baseline if you need to switch back; the
0.19 wheel filename has a `.cu130` infix and lives at
`wheels.vllm.ai/55842a8d6961204f5adbcb5ca07da8cf79d85c33/`.

Historical reference patches from the older `0.18`-era layout:

- `custom-patches/01_allow_extract_hidden_states.patch`
- `custom-patches/02_support_nemotron_models.patch`
- `custom-patches/03_support_gemma3_models.patch`

The old numbered patches are kept as references for intent. They are not the
preferred apply path for the current environment.

The current `0.19rc1` patch adds:

- `extract_activation_layers` plumbing through `ModelConfig`, `EngineArgs`, and `LLM`
- prompt activation capture and request/output propagation in the V1 engine
- OpenAI-compatible `activations` serialization for chat and completion outputs
- Gemma 4 decoder-layer activation capture aligned with the Hugging Face
  `extract_all_hidden_states_npz.py` reference

`extract_activation_layers` is an exact `list[int]` API. For example,
`[12, 20]` captures only layers `12` and `20`.

Gemma 4 activation extraction now returns the last prompt-token hidden state for
each requested layer. That narrower contract is validated against the saved
Hugging Face `.npz` by slicing the reference activations at the last prompt
token during comparison, and it is intended to work on the fast-prefill path
without forcing eager mode.

## Apply the patch

Run from the repository root:

```bash
./custom-patches/apply-patch.sh
```

`apply-patch.sh` prefers `custom-patches/vllm-0.19*.patch`. If no current-version
patch is present, it falls back to the numbered legacy patches.

The script copies `custom-patches/` into the installed `vllm` tree as
`<site-packages>/vllm/patches/`, then applies the selected patch files with
`patch -p1`.

## Verification

After patching, verify the Gemma 4 activations against the saved Hugging Face
reference:

```bash
uv run python compare_vllm_hidden_states.py
```

The current comparison helper tolerates the missing batch dimension on the
vLLM-side activation tensor by squeezing singleton batch dimensions during
comparison only. The default pass threshold is `0.998` cosine similarity across
all layers.

## Assumptions

- Python is installed in `.venv`.
- The installed package path resolves to `.venv/lib/python3.12/site-packages/vllm`.
- The current patch file targets the installed `vllm 0.19rc1` code layout.
