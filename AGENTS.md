# Repository Guidelines

## Project Structure & Module Organization
- `run.py` is the single entry point; it loads YAML configs from `configs/` and dispatches into stage functions in `src/nbglm/pipelines.py`.
- Core modules live under `src/nbglm/`: `model.py` (modeling), `data_io.py` and `dataset.py` (data loading), `eval.py` (metrics), and `utils.py` (shared helpers). Extend the file that matches your change instead of introducing duplicate logic.
- Store large raw inputs in `data/` (not committed). Generated artifacts land in timestamped subfolders under `outputs/`; delete them locally after review.
- Mirror code layout when adding tests by placing them under `tests/` with matching module paths.

## Build, Test, and Development Commands
- Create a clean environment: `python -m venv .venv && source .venv/bin/activate`.
- Install dependencies as documented: `pip install torch scanpy anndata numpy pandas tqdm pyyaml`.
- Run the default pipeline: `python run.py`; select a config via `python run.py --config configs/default.yaml`.
- Override hyperparameters inline, e.g. `python run.py --set pipeline.mode=evaluate_only --set paths.pred_h5ad=...`.

## Coding Style & Naming Conventions
- Use 4-space indentation and keep code Black-compatible; prefer explicit type hints across modules.
- Stick to snake_case for variables, functions, and config keys; reserve PascalCase for classes.
- Keep docstrings concise with an English summary sentence; add comments only when clarifying tensor shapes or non-obvious math.
- Follow module boundaries rather than creating monolithic scripts.

## Testing Guidelines
- No automated suite ships yet; validate logic by running the relevant pipeline stages.
- Reproduce evaluation-only behavior with `python run.py --set pipeline.mode=evaluate_only --set paths.pred_h5ad=...`.
- When adding unit tests, mirror `src/nbglm/` under `tests/` and execute `python -m pytest`.

## Commit & Pull Request Guidelines
- Write commit subjects in present tense under 60 characters (e.g., `Modify sample method`); explain motivation in the body when helpful.
- Squash noisy fixups before opening a PR and outline the problem, solution, config/data expectations, and reproduction commands.
- Include metric snapshots from `outputs/.../logs/run.log` and tag owners of affected pipeline stages; call out follow-up tasks explicitly.

## Security & Configuration Tips
- Parameterize sensitive paths through environment variables referenced in YAML instead of hard-coding absolute routes.
- Do not commit artifacts produced by `tools/precompute_true_de_cache.py`; document regeneration steps for reviewers instead.
