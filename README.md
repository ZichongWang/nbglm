# nbglm

A small, configuration-driven project structure for **Low-Rank NB-GLM** on perturb-seq style data.

> 目标（Goal）  
> 将单脚本重构为**简洁可维护**的小型工程，配置（YAML）驱动，入口统一（`python run.py`），
> 支持“训练 + 采样 + 评估”等组合流程，并为后续 **k-fold CV**、**内存直连评估（采样→评估）** 留好接口。

---

## Run eval
```
python run.py \
   --config configs/default.yaml \
   --set pipeline.mode=evaluate_only \
   --set paths.pred_h5ad=/absolute/path/to/your_predictions.h5ad \
   --set experiment.name=eval_existing_pred \
   --set sampling.sampler=gp_mixture
```
## Directory

```
nbglm/
├── config/
│ └── default.yaml
├── src/nbglm/
│ ├── init.py
│ ├── data_io.py
│ ├── dataset.py
│ ├── model.py
│ ├── eval.py
│ ├── pipelines.py
│ ├── utils.py
│ └── types.py
├── run.py
├── README.md
└── requirements.txt
```


## Quickstart

1. Prepare data paths in `config/default.yaml`:
   - training h5ad(s)
   - test h5ad (for evaluation)
   - embeddings (gene / perturbation)
   - validation list CSV (`target_gene, n_cells, median_umi_per_cell`)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run:

```
python run.py
# or
python run.py --config path/to/exp.yaml
```

Outputs will be placed under `outputs/{experiment_name}__{timestamp}`/ with:

+ `ckpt`/ model checkpoint
+ preds/ sampled predictions (pred.h5ad)
+ metrics/ evaluation metrics (metrics.json)
+ logs/ logs

## Configuration Tips

+ `pipeline.mode`: one of
   + `train_sample_eval`
   + `train_sample`
   + `sample_eval` (requires `pipeline.pretrained_ckpt`)
   + `sample`
   + `evaluate_only` (requires `paths.pred_h5ad`)
   + `multi_seed` (provide `pipeline.seeds` and optional `pipeline.multi_seed_base_mode`)

  When `pipeline.mode=multi_seed`, list the seeds you want to sweep, e.g.:

  ```yaml
  pipeline:
    mode: multi_seed
    seeds: [301, 302, 303]
    multi_seed_base_mode: train_sample_eval  # defaults to train_sample_eval if omitted
    multi_seed_devices: [0, 1, 2]            # optional: map each seed to a GPU (spawned in parallel)
    multi_seed_max_workers: 3                # optional: cap concurrent processes (default = device count)
  ```

  Multi-seed runs spawn independent subprocesses, each with its own `outputs/.../seed_{seed}/` bundle (config snapshot, logs, checkpoints, metrics). An aggregated `metrics/multi_seed_metrics.json` is emitted at the parent level summarizing per-seed scores. Keep `pipeline.persist_intermediate=true` so predictions are written to disk instead of shared in-memory objects.

## Multi-Seed Usage

1. **Configure seeds and GPUs**
   ```yaml
   experiment:
     name: baseline_nbglm_multi_seed
   pipeline:
     mode: multi_seed
     seeds: [101, 202, 303, 404]
     multi_seed_devices: [0, 1, 2, 3]     # optional GPU mapping per worker
     multi_seed_max_workers: 4            # optional cap on concurrent workers
     persist_intermediate: true           # must stay true so each worker writes outputs
     multi_seed_base_mode: train_sample_eval
   ```

2. **Launch from the CLI**
   ```bash
   python run.py --config configs/default.yaml \\
     --set pipeline.mode=multi_seed \\
     --set pipeline.seeds=[101,202,303,404] \\
     --set pipeline.multi_seed_devices=[0,1,2,3]
   ```
   - Drop `pipeline.multi_seed_devices` to inherit the current `CUDA_VISIBLE_DEVICES` (or fall back to CPU).
   - Use `--set pipeline.multi_seed_max_workers=2` to limit concurrency without changing the seed list.

3. **Inspect the outputs**
   - Parent folder: `outputs/{experiment_name}__{timestamp}/`.
   - Each seed writes to `seed_{seed}/` with its own `config.yaml`, `logs/run.log`, `ckpt/`, `preds/`, and `metrics/`.
   - Aggregated statistics live at `metrics/multi_seed_metrics.json`, including per-seed metrics, assigned devices, and mean/std/min/max summaries.

4. **Reproducibility tips**
   - Set `experiment.seed` in addition to `pipeline.seeds` if downstream logic uses a global seed.
   - Make sure GPU indices match the device ordering from `nvidia-smi`; strings like `"cuda:2"` are normalized to `2`, while `"cpu"`/`-1` forces a CPU run.

+ To switch **whole-cell** vs **pseudo-bulk** training:
   + `train.fit_mode: whole | concise`

+ To use cell cycle covariate:
   + `model.use_cycle: true` and ensure `data.phase_column` exists.


## Notes

+ The project uses **row-wise L2 normalized embeddings** by default.

+ Checkpoints include `G/P` matrices, `mu_control`, and `theta` to ensure portability.

