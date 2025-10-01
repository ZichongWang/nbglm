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
   --set experiment.name=eval_existing_pred
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

+ To switch **whole-cell** vs **pseudo-bulk** training:
   + `train.fit_mode: whole | concise`

+ To use cell cycle covariate:
   + `model.use_cycle: true` and ensure `data.phase_column` exists.


## Notes

+ The project uses **row-wise L2 normalized embeddings** by default.

+ Checkpoints include `G/P` matrices, `mu_control`, and `theta` to ensure portability.