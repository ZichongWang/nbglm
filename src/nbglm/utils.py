# src/nbglm/utils.py
# -*- coding: utf-8 -*-
"""
通用工具（Utilities）
====================

提供：
- 随机种子（random seed）统一设置
- 设备选择（device selection）
- 日志系统（logging）封装
- 简易文件/JSON 辅助函数
"""

from __future__ import annotations

from typing import Optional, Dict
import os
import sys
import json
import random
from pathlib import Path
import logging
import time

import numpy as np

try:
    import torch
except Exception:
    torch = None  # 允许在无 torch 环境下导入本模块的非训练逻辑


# -----------------------------
# 随机种子
# -----------------------------
def set_seed(seed: int) -> None:
    """
    统一设置随机种子（random / numpy / torch）。
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 设备选择
# -----------------------------
def get_device(cfg: Dict) -> str:
    """
    从 cfg.experiment.device 选择设备：
    - "auto"（默认）：有 CUDA 则 "cuda" 否则 "cpu"
    - "cuda" / "cuda:0" / "cpu"
    """
    dev = str(cfg.get("experiment", {}).get("device", "auto")).lower()
    if dev == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return dev


# -----------------------------
# 日志系统
# -----------------------------
def get_logger(run_dir: str, name: str = "nbglm") -> logging.Logger:
    """
    配置并返回一个 logger，同时输出到控制台与文件（logs/run.log）。
    """
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "run.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 避免重复输出

    # 清理旧 handler（避免多次创建）
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logger initialized. Log file: {log_path}")
    return logger


# -----------------------------
# 其他小工具
# -----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def json_dump(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
