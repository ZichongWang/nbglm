# nbglm

A small, configuration-driven project structure for **Low-Rank NB-GLM** on perturb-seq style data.

---
## Quick start
1. Manually split dataset(code not provided). I split the training dataset, 120 perts for training (`train.h5ad`), 30 perts for testing(`test.h5ad`). Keep control cells in both files. And create a csv file containing pert, cell num, median UMI(`test_pert_info.csv`).
2. Run 
   ```
   python tools/vcc_build_ground_truth_cache.py \
         --real /path/to/test.h5ad \
         --output /path/to/output
   ```
   The output is a folder, contains three files. Use `data/test_de_cache` as example.

3. Modify configs. Use `configs/multi_seed_poisson.yaml` as example, which is the most frequent config file I use.
   + Set `paths.training_h5ad` as the file VCC official provide.
   + Set `paths.train_split_h5ad` as `path/to/train.h5ad`
   + Set `paths.test_h5ad` as `path/to/test.h5ad`
   + Set `paths.val_list_csv` as `path/to/test_pert_info.csv`
   + The rest three path configs are easy to understand.
   + Set `evaluate.true_de_cache` as `data/test_de_cache`.

## Run eval
```
python run.py \
   --config configs/default.yaml \
   --set pipeline.mode=evaluate_only \
   --set paths.pred_h5ad=/absolute/path/to/your_predictions.h5ad \
   --set experiment.name=eval_existing_pred \
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

+ `evaluate.backend`: choose `"legacy"` for the original evaluator (`src/nbglm/eval.py`) or `"vcc"` for the VCC metrics (`src/nbglm/vcc_eval.py`). The default is `"legacy"`.

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

## MLP Hyperparameters for Embeddings

The `LowRankNB_GLM` can process gene (`G`) and perturbation (`P`) embeddings through configurable MLP stacks before applying the low-rank kernel. Configure them in your YAML under `model`:

```yaml
model:
  g_mlp_hidden: [384]   # hidden widths for the gene-side MLP ([] = identity)
  g_out_dim: 192        # optional final dimension for processed gene embeddings
  p_mlp_hidden: [384]   # hidden widths for the perturbation-side MLP
  p_out_dim: 192        # optional final dimension for processed perturbations
  gp_activation: "gelu" # activation name (relu/gelu/silu/tanh/identity)
  gp_norm: "layernorm"  # normalization per hidden layer: none|batchnorm|layernorm
  gp_dropout: 0.1       # dropout applied after each hidden layer (0-1)
```

Pipelines parse these fields automatically: training builds the MLPs via `build_mlp` and stores the configuration inside the checkpoint; sampling/evaluation rebuild the same architecture when the checkpoint is loaded. Override the defaults per experiment to explore deeper stacks or alternative activations without code edits.

## Whole-cell vs Pseudo-bulk

- `train.fit_mode: whole | concise`

## Cell Cycle Covariate

- Enable with `model.use_cycle: true` and ensure `data.phase_column` exists in the AnnData `obs`.


## Notes

+ The project uses **row-wise L2 normalized embeddings** by default.

+ Checkpoints include `G/P` matrices, `mu_control`, and `theta` to ensure portability.
