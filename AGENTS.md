# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python 3.12 `uv` project for experimenting with `google/gemma-4-26B-A4B-it` through `transformers`.

- `main.py`: loads the processor and model, then runs a sample prompt.
- `extract_hidden_states.py`: captures and serializes the hidden state for one decoder layer and prompt.
- `extract_all_hidden_states_npz.py`: captures hidden states for all decoder layers and writes a compressed `.npz`.
- `layer_sweep.py`: walks all layers and writes a report to `layers.md`.
- `layers.md`: generated markdown report of decoder layer forward signatures and source.
- `journal.md`: project notes and recorded findings from environment and model inspection.
- `README.md`: minimal usage notes.
- `pyproject.toml` and `uv.lock`: dependency and environment definitions.

Keep new exploratory scripts at the repo root unless a clear subpackage emerges. Write generated artifacts to tracked markdown files or an explicit output path.
Prefer not to commit generated binary artifacts such as `.pt` or `.npz` by default; only track them when they are intentional reference artifacts. Git LFS is unnecessary for small one-off artifacts and should only be considered if binary snapshots become numerous or large.

## Build, Test, and Development Commands
- `uv sync --dev`: install project and dev dependencies into the local environment.
- `uv run python main.py`: run the base Gemma 4 inference example.
- `uv run python extract_hidden_states.py 29`: capture one layer's hidden state for the default prompt.
- `uv run python extract_all_hidden_states_npz.py`: capture hidden states for all decoder layers into a compressed `.npz`.
- `uv run python layer_sweep.py`: regenerate `layers.md` from the current model layout.
- `uv run python -m py_compile main.py layer_sweep.py extract_hidden_states.py extract_all_hidden_states_npz.py`: quick syntax validation.

These scripts load a large model. Expect GPU, Hugging Face access, and substantial memory requirements.

## Coding Style & Naming Conventions
Follow standard Python conventions:

- Use 4-space indentation and type hints on public functions.
- Prefer `UPPER_SNAKE_CASE` for module-level constants like `MODEL_ID`.
- Use `snake_case` for functions and variables.
- Keep scripts direct and inspectable; avoid unnecessary abstractions in one-off exploration code.

No formatter or linter is configured yet. Match the existing style and keep imports grouped and minimal.

## Testing Guidelines
There is no automated test suite yet. For changes, run the relevant script plus `py_compile` before opening a PR. If you add reusable logic, add `pytest`-style tests in a future `tests/` directory and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects, for example: `Add Gemma 4 uv demo project`. Follow that pattern.

PRs should include:
- a brief summary of behavior changes,
- the commands you ran to validate them,
- any model, GPU, or Hugging Face prerequisites,
- sample output or regenerated artifacts when relevant.
