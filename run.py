# run.py
# -*- coding: utf-8 -*-
"""
统一入口（Entry Point）
======================

使用方式：
----------
1) 直接运行默认配置：
   $ python run.py

2) 指定配置文件（YAML）：
   $ python run.py --config path/to/exp.yaml
   或设置环境变量：
   $ NBGLM_CONFIG=path/to/exp.yaml python run.py
"""

from __future__ import annotations

import os
import argparse
import json
from typing import Dict, Any

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

from src.nbglm import data_io, pipelines, utils


def _load_cfg(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[run] 配置文件不存在: {path}")
    if _HAS_YAML and (path.endswith(".yaml") or path.endswith(".yml")):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="nbglm runner")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg_path = args.config or os.environ.get("NBGLM_CONFIG", "config/default.yaml")
    cfg = _load_cfg(cfg_path)

    # 输出目录与日志
    run_dirs = data_io.make_run_dirs(
        outputs_root=cfg["paths"].get("outputs_root", "./outputs"),
        experiment_name=cfg["experiment"].get("name", "nbglm_experiment"),
    )
    data_io.save_config_snapshot(cfg, run_dirs["run_dir"])
    logger = utils.get_logger(run_dirs["run_dir"])
    utils.set_seed(int(cfg["experiment"].get("seed", 2025)))
    logger.info(f"Loaded config from: {cfg_path}")

    # 分发 pipeline
    mode = cfg.get("pipeline", {}).get("mode", "train_sample_eval").lower()
    logger.info(f"Pipeline mode = {mode}")

    if mode == "train_sample_eval":
        out = pipelines.run_train_sample_eval(cfg, run_dirs)
    elif mode == "train_sample":
        out = pipelines.run_train_sample(cfg, run_dirs)
    elif mode == "sample_eval":
        out = pipelines.run_sample_eval(cfg, run_dirs)
    elif mode == "sample":
        out = pipelines.run_sample_only(cfg, run_dirs)
    elif mode == "evaluate_only":
        out = pipelines.run_evaluate_only(cfg, run_dirs)
    else:
        raise ValueError(f"[run] 未知 pipeline.mode: {mode}")

    logger.info(f"Done. Summary: {json.dumps(out, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
