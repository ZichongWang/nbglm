# Repository Guidelines

## Project Structure & Module Organization
`run.py` is the single entry point; it reads YAML in `configs/` and dispatches to the stage-specific functions inside `src/nbglm/pipelines.py`. Modeling code (`model.py`), dataloading (`data_io.py`, `dataset.py`), metrics (`eval.py`), and shared utilities (`utils.py`) are tightly scoped—extend the component that matches your change instead of adding new monolith scripts. Large inputs live under `data/` (kept out of git) and generated runs write into timestamped folders inside `outputs/`; delete those locally after inspection.

## Build, Test, and Development Commands
Use a fresh environment (`python -m venv .venv && source .venv/bin/activate`) and install the scientific stack noted in `README.md` (`pip install torch scanpy anndata numpy pandas tqdm pyyaml`). Run the default pipeline with `python run.py` or a specific config via `python run.py --config configs/default.yaml`. Override hyperparameters inline during experiments, e.g. `python run.py --set pipeline.mode=sample_eval --set pipeline.pretrained_ckpt=outputs/baseline_nbglm__20250930_021716/ckpt/model.pt`.

## Coding Style & Naming Conventions
Match the existing modules: four-space indentation, Black-compatible formatting, and docstrings that open with a brief English summary. Use snake_case for functions, variables, and config keys; reserve PascalCase for classes. Prefer explicit type hints across module boundaries and keep comments focused on tensor shapes or tricky numerical steps.

## Testing Guidelines
No automated suite ships yet, so exercise new logic through the pipelines: `python run.py --set pipeline.mode=evaluate_only --set paths.pred_h5ad=...` should reproduce metrics without retraining. When adding unit coverage, mirror the `src/nbglm/` layout under `tests/` and run `python -m pytest`. Capture key logs or plots in your PR description so reviewers can verify scientific changes without raw data.

## Commit & Pull Request Guidelines
Recent history uses concise, present-tense subjects (`Modify sample method`), so keep commit titles under 60 characters and explain motivation in the body when needed. Squash noisy fixups before opening a PR. Each PR should outline the problem, summarize the solution, list config or data expectations, and include reproduction commands plus metric snapshots from `outputs/.../logs/run.log`. Tag owners of the affected pipeline stage and call out any follow-up tasks explicitly.

## Configuration & Data Handling
Parameterize sensitive paths through environment variables consumed in YAML instead of hard-coding absolute routes. Limit new configs to the knobs you legitimately change—inherit defaults via YAML anchors or documented comments rather than copying entire blocks. Do not commit artifacts produced by `tools/precompute_true_de_cache.py`; instead, describe how to regenerate them when reviewer validation is necessary.
