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
import torch
import torch.nn as nn
from typing import List, Optional, Literal, Tuple, Union, Any, Dict
import numpy as np
from numbers import Number
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

    class _NoWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno != logging.WARNING

    sh.addFilter(_NoWarningFilter())
    logger.addHandler(sh)

    # 文件
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.debug(f"Logger initialized. Log file: {log_path}")
    return logger

        

def build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: Optional[int] = None,
    activation: Optional[nn.Module] = None,
    dropout: float = 0.0,
    norm: Literal["none", "batchnorm", "layernorm"] = "none",
):
    """
    构建一个 MLP（用于 embedding 前置变换, pre-net）。
    规则：
      - 若 hidden_dims=[] 且 (out_dim is None or out_dim==in_dim) => Identity
      - 否则按 [Linear -> Norm? -> Act -> Dropout?] 重复，末层输出 out_dim（若给定）
    """
    if activation is None:
        activation = nn.ReLU()

    dims = [in_dim] + list(hidden_dims)
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1], bias=False))
        if norm == "batchnorm":
            layers.append(nn.BatchNorm1d(dims[i+1]))
        elif norm == "layernorm":
            layers.append(nn.LayerNorm(dims[i+1]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    final_out = dims[-1] if out_dim is None else out_dim
    # 若没有隐藏层且 out_dim 与 in_dim 一样 => Identity
    if len(layers) == 0 and (out_dim is None or out_dim == in_dim):
        return nn.Identity(), in_dim

    # 需要末层映射到 out_dim（若指定且不同）
    if out_dim is not None and final_out != dims[-1]:
        # 单独加一个线性输出层（不再叠激活/归一化）
        layers.append(nn.Linear(dims[-1], out_dim, bias=False))
        final_out = out_dim

    mlp = nn.Sequential(*layers)
    # Kaiming 或 Xavier 初始化（兼容 ReLU/GELU/Tanh）
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return mlp, final_out


def _format_metrics_markdown(entries: List[Tuple[str, Dict[str, Any]]]) -> str:
    """Render evaluation results into Markdown sections."""

    def _to_float(value: Any) -> Optional[float]:
        if isinstance(value, Number):
            return float(value)
        try:
            return float(value)
        except Exception:
            return None

    def _fmt_float(value: Optional[float]) -> str:
        return f"{value:.4f}" if value is not None else "-"

    def _fmt_int(value: Optional[int]) -> str:
        return str(int(value)) if value is not None else "-"

    def _as_int(metrics: Dict[str, Any], key: str) -> Optional[int]:
        value = metrics.get(key)
        if value is None:
            return None
        try:
            return int(round(float(value)))
        except Exception:
            return None

    def _safe_ratio(num: Optional[int], den: Optional[int]) -> Optional[float]:
        if num is None or den is None or den == 0:
            return None
        return float(num) / float(den)

    primary_specs = [
        ("DES", "DES"),
        ("PDS", "PDS"),
        ("MAE", "MAE"),
        ("Score", "Score"),
    ]

    header = "| seed | " + " | ".join(label for label, _ in primary_specs) + " |"
    separator = "| " + " | ".join(["---"] * (len(primary_specs) + 1)) + " |"
    rows: List[str] = []
    collected: Dict[str, List[float]] = {key: [] for _, key in primary_specs}

    for seed_label, metrics in entries:
        metrics = dict(metrics or {})
        if "Score" not in metrics:
            score_val = metrics.get("Score") or metrics.get("Overall")
            if score_val is None:
                des = _to_float(metrics.get("DES"))
                pds = _to_float(metrics.get("PDS"))
                mae = _to_float(metrics.get("MAE"))
                if None not in (des, pds, mae):
                    des_base, pds_base, mae_base = 0.0761, 0.52, 0.0269
                    try:
                        des_scaled = max(0.0, min(1.0, ((des - des_base) / (1 - des_base)))) if des is not None else None
                        pds_scaled = max(0.0, min(1.0, ((pds - pds_base) / (1 - pds_base)))) if pds is not None else None
                        mae_scaled = max(0.0, min(1.0, ((mae_base - mae) / mae_base))) if mae is not None else None
                        if None not in (des_scaled, pds_scaled, mae_scaled):
                            score_val = 100.0 * (des_scaled + pds_scaled + mae_scaled) / 3.0
                    except Exception:
                        score_val = None
            if score_val is not None:
                metrics["Score"] = score_val

        row_values = []
        for _, key in primary_specs:
            value = metrics.get(key)
            if key == "Score" and value is None:
                value = metrics.get("Overall")
            float_value = _to_float(value)
            if float_value is not None:
                collected[key].append(float_value)
            row_values.append(_fmt_float(float_value))
        rows.append("| " + " | ".join([seed_label, *row_values]) + " |")

    def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not values:
            return None, None
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return mean, variance ** 0.5

    mean_row = ["Mean"]
    std_row = ["Std Dev"]
    for _, key in primary_specs:
        mean_val, std_val = _mean_std(collected[key])
        mean_row.append(_fmt_float(mean_val))
        std_row.append(_fmt_float(std_val))

    sections: List[str] = []
    primary_table = "\n".join([header, separator, *rows, "| " + " | ".join(mean_row) + " |", "| " + " | ".join(std_row) + " |"])
    sections.append("**Primary Metrics**\n" + primary_table)

    for seed_label, metrics in entries:
        metrics = dict(metrics or {})
        tp = _as_int(metrics, "DE_TP")
        fp = _as_int(metrics, "DE_FP")
        fn = _as_int(metrics, "DE_FN")
        tn = _as_int(metrics, "DE_TN")

        if any(value is None for value in (tp, fp, fn, tn)):
            continue

        total = _as_int(metrics, "DE_m")
        if total is None and None not in (tp, fp, fn, tn):
            total = tp + fp + fn + tn

        rejections = _as_int(metrics, "DE_R")
        if rejections is None and None not in (tp, fp):
            rejections = tp + fp

        true_nulls = _as_int(metrics, "DE_m0")
        if true_nulls is None and None not in (tn, fp):
            true_nulls = tn + fp

        precision = _to_float(metrics.get("DE_PPV"))
        if precision is None:
            precision = _safe_ratio(tp, rejections)

        recall = _to_float(metrics.get("DE_TPR"))
        if recall is None:
            recall = _safe_ratio(tp, tp + fn if tp is not None and fn is not None else None)

        fdr = _to_float(metrics.get("DE_FDR"))
        if fdr is None:
            fdr = _safe_ratio(fp, rejections)

        fnr = _to_float(metrics.get("DE_FNR"))
        if fnr is None:
            fnr = _safe_ratio(fn, tp + fn if tp is not None and fn is not None else None)

        accuracy = _to_float(metrics.get("DE_Accuracy"))
        if accuracy is None and total not in (None, 0):
            accuracy = _safe_ratio(tp + tn if None not in (tp, tn) else None, total)

        f1 = _to_float(metrics.get("DE_F1"))
        if f1 is None and precision is not None and recall is not None and (precision + recall) != 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        confusion_lines = [
            "| Pred \\ GT | GT: significant | GT: non-significant |",
            "| --- | --- | --- |",
            f"| Pred: significant | {_fmt_int(tp)} | {_fmt_int(fp)} |",
            f"| Pred: non-significant | {_fmt_int(fn)} | {_fmt_int(tn)} |",
        ]

        summary_lines = [
            "| Metric | Value |",
            "| --- | --- |",
            f"| TP (S) | {_fmt_int(tp)} |",
            f"| FP (V) | {_fmt_int(fp)} |",
            f"| FN (T) | {_fmt_int(fn)} |",
            f"| TN (U) | {_fmt_int(tn)} |",
            f"| m (total) | {_fmt_int(total)} |",
            f"| R (rejections) | {_fmt_int(rejections)} |",
            f"| m0 (true nulls) | {_fmt_int(true_nulls)} |",
            f"| Precision / PPV | {_fmt_float(precision)} |",
            f"| Recall / TPR / Power | {_fmt_float(recall)} |",
            f"| FDR (FDP) | {_fmt_float(fdr)} |",
            f"| FNR | {_fmt_float(fnr)} |",
            f"| Accuracy | {_fmt_float(accuracy)} |",
            f"| F1 | {_fmt_float(f1)} |",
        ]

        sections.append(
            "**Seed "
            + seed_label
            + " DE Confusion Matrix**\n"
            + "\n".join(confusion_lines)
            + "\n\n**Seed "
            + seed_label
            + " DE Metrics**\n"
            + "\n".join(summary_lines)
        )

    return "\n\n".join(sections)


def _ensure_int_list(value: Any, *, default: Optional[List[int]] = None) -> List[int]:
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    if isinstance(value, Number):
        return [int(value)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default or [])
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            return list(default or [])
        return [int(float(p)) for p in parts]
    return list(default or [])


def _to_optional_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        try:
            return int(float(text))
        except Exception:
            return default
    try:
        return int(value)
    except Exception:
        return default


def _resolve_activation_module(value: Any) -> nn.Module:
    if isinstance(value, nn.Module):
        return value
    name: Optional[str] = None
    if isinstance(value, dict):
        name = str(value.get("name") or value.get("type") or value.get("activation") or "").strip()
    elif value is not None:
        name = str(value).strip()

    if not name:
        return nn.GELU()

    name_low = name.lower()
    if name_low.startswith("nn."):
        name_low = name_low[3:]

    if name_low in {"none", "identity"}:
        return nn.Identity()

    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    factory = mapping.get(name_low)
    if factory is not None:
        module = factory() if callable(factory) else factory
        if isinstance(module, nn.Module):
            return module
    return nn.GELU()


def _resolve_model_mlp_kwargs(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    g_hidden = _ensure_int_list(model_cfg.get("g_mlp_hidden"), default=[384])
    p_hidden = _ensure_int_list(model_cfg.get("p_mlp_hidden"), default=[384])

    g_out = _to_optional_int(model_cfg.get("g_out_dim"), default=192)
    p_out = _to_optional_int(model_cfg.get("p_out_dim"), default=192)

    dropout_raw = model_cfg.get("gp_dropout", 0.1)
    try:
        gp_dropout = float(dropout_raw)
    except Exception:
        gp_dropout = 0.1
    gp_dropout = float(max(0.0, min(1.0, gp_dropout)))

    norm_raw = str(model_cfg.get("gp_norm", "layernorm") or "layernorm").strip().lower()
    gp_norm = norm_raw if norm_raw in {"none", "batchnorm", "layernorm"} else "none"

    activation = _resolve_activation_module(model_cfg.get("gp_activation", "gelu"))

    return {
        "g_mlp_hidden": g_hidden,
        "p_mlp_hidden": p_hidden,
        "g_out_dim": g_out,
        "p_out_dim": p_out,
        "gp_activation": activation,
        "gp_dropout": gp_dropout,
        "gp_norm": gp_norm,
    }



# -----------------------------
# 其他小工具
# -----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def json_dump(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

