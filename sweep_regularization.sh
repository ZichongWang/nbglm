#!/usr/bin/env bash

# Regularization sweep for nbglm using a base multi-seed config.
# - Uses configs/multi_seed_poisson.yaml by default (already defines seeds/devices/mode)
# - Runs all three metrics: ["MAE","PDS","DES"]
# - Writes runs under outputs/<GROUP>/... where GROUP is a sweep folder
# - Generates a final markdown summary with per-run tables and 4 metric grids
#
# Usage:
#   bash sweep_regularization.sh [config_path] [group_name]
#     config_path: defaults to configs/multi_seed_poisson.yaml
#     group_name:  defaults to reg_sweep_YYYYmmdd_HHMMSS

set -euo pipefail

CONFIG_PATH=${1:-configs/multi_seed_gpm.yaml}
GROUP_NAME=${2:-gpm_reg_sweep_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT="./outputs"
OUT_GROUP_DIR="${OUT_ROOT}/${GROUP_NAME}"
mkdir -p "${OUT_GROUP_DIR}"
echo "[sweep] Group folder: ${OUT_GROUP_DIR}"

# Search grids (edit as needed)
L1_VALUES=(0 1e-6 1e-5 1e-4 5e-4 1e-3)
L2_VALUES=(0 1e-6 1e-5 5e-5 1e-4 5e-4 1e-3)

for l1 in "${L1_VALUES[@]}"; do
  for l2 in "${L2_VALUES[@]}"; do
    run_name="l1_${l1}_l2_${l2}"
    echo "[sweep] Running: ${run_name}"
    python run.py --config "${CONFIG_PATH}" \
      --set model.regularization.l1="${l1}" \
      --set model.regularization.l2="${l2}" \
      --set evaluate.metrics='["MAE","PDS","DES"]' \
      --set experiment.name="${GROUP_NAME}/${run_name}"
  done
done

echo "[sweep] Completed grid over L1 and L2. Building summary..."

# --------------------------------------
# Build a combined markdown summary file
# --------------------------------------
SUM_PATH="${OUT_GROUP_DIR}/sweep_summary.md"

L1S="${L1_VALUES[*]}"
L2S="${L2_VALUES[*]}"
export OUT_GROUP_DIR L1S L2S

python - << 'PY'
import os, re, json, sys, datetime

out_dir = os.environ['OUT_GROUP_DIR']
l1_vals = os.environ['L1S'].split()
l2_vals = os.environ['L2S'].split()

# Collect latest run dir per (l1,l2)
pair_latest = {}
pat = re.compile(r'^l1_(?P<l1>[^_]+)_l2_(?P<l2>[^_]+)__(?P<ts>\d{8}_\d{6})$')
if not os.path.isdir(out_dir):
    sys.exit(0)
for name in os.listdir(out_dir):
    path = os.path.join(out_dir, name)
    if not os.path.isdir(path):
        continue
    m = pat.match(name)
    if not m:
        continue
    l1 = m.group('l1')
    l2 = m.group('l2')
    ts = m.group('ts')
    key = (l1, l2)
    if key not in pair_latest or ts > pair_latest[key][0]:
        pair_latest[key] = (ts, path)

# Helper to read individual MD and aggregated means
def read_md(run_dir: str):
    md1 = os.path.join(run_dir, 'metrics', 'metrics_summary.md')
    md2 = os.path.join(run_dir, 'metrics', 'metrics.md')
    for p in (md1, md2):
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return f.read()
    return None

def read_agg(run_dir: str):
    jp = os.path.join(run_dir, 'metrics', 'multi_seed_metrics.json')
    agg = {}
    if os.path.exists(jp):
        with open(jp, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        agg = obj.get('aggregate', {}) or {}
    return agg

# Prepare summary content
lines = []
lines.append('# Regularization Sweep Summary')
lines.append('')
lines.append(f'- Group folder: `{out_dir}`')
lines.append(f'- Generated at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
lines.append('')

# Individual MD tables
lines.append('## Individual Reports')

for l1 in l1_vals:
    for l2 in l2_vals:
        key = (l1, l2)
        if key not in pair_latest:
            continue
        ts, run_dir = pair_latest[key]
        md = read_md(run_dir)
        if not md:
            continue
        lines.append(f'### l1={l1}, l2={l2}')
        lines.append('')
        lines.append(md.strip())
        lines.append('')

# Build metric grids from aggregated means
metrics = {'DES': {}, 'PDS': {}, 'MAE': {}, 'Score': {}}
for key, (_, run_dir) in pair_latest.items():
    agg = read_agg(run_dir)
    def get_mean(metric, alt=None):
        if metric in agg and isinstance(agg[metric], dict):
            return agg[metric].get('mean')
        if alt and alt in agg and isinstance(agg[alt], dict):
            return agg[alt].get('mean')
        return None
    mdes = get_mean('DES')
    mpds = get_mean('PDS')
    mmae = get_mean('MAE')
    mscore = get_mean('Score', 'Overall')
    metrics['DES'][key] = mdes
    metrics['PDS'][key] = mpds
    metrics['MAE'][key] = mmae
    metrics['Score'][key] = mscore

def fmt(x):
    return '-' if x is None else f'{x:.4f}'

def grid_table(title, grid):
    lines = []
    lines.append(f'## {title} (mean across seeds)')
    # header
    header = '| l1 \\ l2 | ' + ' | '.join(l2_vals) + ' |'
    sep = '| --- | ' + ' | '.join(['---'] * len(l2_vals)) + ' |'
    lines.append(header)
    lines.append(sep)
    for l1 in l1_vals:
        row_cells = [f'{l1}']
        for l2 in l2_vals:
            val = grid.get((l1, l2))
            row_cells.append(fmt(val))
        lines.append('| ' + ' | '.join(row_cells) + ' |')
    return lines

lines.append('## Aggregated Grids')
for name in ('DES', 'PDS', 'MAE', 'Score'):
    lines += grid_table(name, metrics[name])
    lines.append('')

with open(os.path.join(out_dir, 'sweep_summary.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines).rstrip() + '\n')

print(f"[summary] Wrote {os.path.join(out_dir, 'sweep_summary.md')}")
PY

echo "[sweep] Summary written to: ${SUM_PATH}"
