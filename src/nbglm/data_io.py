# src/nbglm/data_io.py
# -*- coding: utf-8 -*-
"""
数据 I/O（Data I/O）与路径管理工具
=================================

本模块负责工程中的**数据读写与路径组织**，包括：
1) 读取/保存 AnnData（.h5ad）
2) 读取基因与扰动的嵌入矩阵（gene/pert embeddings, CSV）
3) 针对实验运行构建统一的输出目录结构（run_dir/ckpt/preds/metrics/logs）
4) 配置快照保存（config snapshot），便于复现（reproducibility）

设计要点（Design Notes）
-----------------------
- **解耦**：所有磁盘路径与 I/O 放在此模块，便于统一替换（例如改用 parquet/hdf5）。
- **容错**：当某些基因/扰动在嵌入表中缺失时，使用零向量（zero vector）占位并警告。
- **对齐**：嵌入按目标顺序输出，并提供 L2 归一化（row-wise L2 normalization）开关。

依赖（Dependencies）
-------------------
- anndata, scanpy（用于 .h5ad 读写）
- pandas（读 CSV）
- torch（输出为 torch.Tensor）
"""

from __future__ import annotations

import logging
import os
import json
import time
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import anndata as ad

try:
    import yaml  # 非必需；若缺失则使用 JSON 存配置快照
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


logger = logging.getLogger("nbglm.data_io")


# -----------------------------
# AnnData 读写
# -----------------------------
def read_h5ad(path: str) -> ad.AnnData:
    """
    读取 .h5ad 文件（AnnData 格式）。

    Parameters
    ----------
    path : str
        .h5ad 文件路径。

    Returns
    -------
    anndata.AnnData
        读取得到的 AnnData 对象。
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[read_h5ad] 文件不存在: {path}")
    return ad.read_h5ad(path)


def write_h5ad(adata: ad.AnnData, path: str) -> None:
    """
    保存 AnnData 到 .h5ad 文件。

    Parameters
    ----------
    adata : anndata.AnnData
        要写入的 AnnData 对象。
    path : str
        输出文件路径（若上级目录不存在将自动创建）。
    """
    path = str(path)
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    adata.write(path)


# -----------------------------
# 嵌入加载与对齐
# -----------------------------
def _row_l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    行向量 L2 归一化（Row-wise L2 normalization）。

    Parameters
    ----------
    x : torch.Tensor
        形状 [N, D] 的张量。
    eps : float
        数值稳定用的下限。

    Returns
    -------
    torch.Tensor
        L2 归一化后的张量（逐行）。
    """
    norms = torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(eps)
    return x / norms


def load_embeddings(
    gene_embedding_csv: str,
    pert_embedding_csv: str,
    gene_names: List[str],
    pert_names: List[str],
    l2_normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载并对齐基因与扰动的嵌入矩阵（embeddings）。

    输入 CSV 应该满足：
      - 行索引为实体名称（index 为 gene 名或 perturbation 名）
      - 列为数值型嵌入维度
    若目标列表（gene_names / pert_names）中某些名称在 CSV 中缺失，则用**零向量**填充。

    Parameters
    ----------
    gene_embedding_csv : str
        基因嵌入 CSV 文件路径，index=gene 名。
    pert_embedding_csv : str
        扰动嵌入 CSV 文件路径，index=pert 名。
    gene_names : List[str]
        目标基因顺序（将按此顺序对齐并输出）。
    pert_names : List[str]
        目标扰动顺序（将按此顺序对齐并输出）。
    l2_normalize : bool
        是否对输出矩阵逐行做 L2 归一化（row-wise）。

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (G_matrix, P_matrix)
        - G_matrix: [G, d_g]
        - P_matrix: [P, d_p]
    """
    if not os.path.exists(gene_embedding_csv):
        raise FileNotFoundError(f"[load_embeddings] 基因嵌入文件不存在: {gene_embedding_csv}")
    if not os.path.exists(pert_embedding_csv):
        raise FileNotFoundError(f"[load_embeddings] 扰动嵌入文件不存在: {pert_embedding_csv}")

    gene_df = pd.read_csv(gene_embedding_csv, index_col=0)
    pert_df = pd.read_csv(pert_embedding_csv, index_col=0)

    # 基因
    d_g = gene_df.shape[1]
    G = torch.zeros(len(gene_names), d_g, dtype=torch.float32)
    missing_genes = []
    for i, name in enumerate(gene_names):
        if name in gene_df.index:
            G[i] = torch.tensor(gene_df.loc[name].values, dtype=torch.float32)
        else:
            missing_genes.append(name)

    # 扰动
    d_p = pert_df.shape[1]
    P = torch.zeros(len(pert_names), d_p, dtype=torch.float32)
    missing_perts = []
    for i, name in enumerate(pert_names):
        if name in pert_df.index:
            P[i] = torch.tensor(pert_df.loc[name].values, dtype=torch.float32)
        else:
            missing_perts.append(name)

    if missing_genes:
        logger.warning(
            "load_embeddings: %d 个基因缺失嵌入，已用零向量填充。示例: %s",
            len(missing_genes),
            missing_genes[:5],
        )
    if missing_perts:
        logger.warning(
            "load_embeddings: %d 个扰动缺失嵌入，已用零向量填充。示例: %s",
            len(missing_perts),
            missing_perts[:5],
        )

    if l2_normalize:
        G = _row_l2_normalize(G)
        P = _row_l2_normalize(P)

    return G, P


# -----------------------------
# 运行目录与配置快照
# -----------------------------
def make_run_dirs(outputs_root: str, experiment_name: str, timestamp: Optional[str] = None) -> Dict[str, str]:
    """
    创建一次运行的输出目录树，并返回常用子目录路径。

    约定的目录结构为：
    outputs/{experiment_name}__{YYYYmmdd_HHMMSS}/
      ├─ ckpt/
      ├─ preds/
      ├─ metrics/
      └─ logs/

    Parameters
    ----------
    outputs_root : str
        总输出根目录（如 "./outputs"）。
    experiment_name : str
        实验名，用于组装 run_id。
    timestamp : Optional[str]
        指定时间戳；若为 None，将使用当前时间。

    Returns
    -------
    Dict[str, str]
        包含 'run_dir', 'ckpt_dir', 'pred_dir', 'metrics_dir', 'logs_dir' 五个键值。
    """
    outputs_root = str(outputs_root)
    Path(outputs_root).mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{experiment_name}__{timestamp}"

    run_dir = os.path.join(outputs_root, run_id)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    pred_dir = os.path.join(run_dir, "preds")
    metrics_dir = os.path.join(run_dir, "metrics")
    logs_dir = os.path.join(run_dir, "logs")

    for d in (run_dir, ckpt_dir, pred_dir, metrics_dir, logs_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "pred_dir": pred_dir,
        "metrics_dir": metrics_dir,
        "logs_dir": logs_dir,
    }


def save_config_snapshot(cfg: dict, out_dir: str, filename_yaml: str = "config.yaml") -> None:
    """
    保存配置快照，优先 YAML；若 PyYAML 不可用则保存 JSON。

    Parameters
    ----------
    cfg : dict
        配置字典（或可转为字典的对象）。
    out_dir : str
        输出目录（通常为 run_dir）。
    filename_yaml : str
        文件名（默认 "config.yaml"）。若不支持 YAML 将用 "config.json"。
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 尝试将非基本类型（如 OmegaConf）转为普通 dict
    try:
        from omegaconf import OmegaConf  # 可选
        if not isinstance(cfg, dict):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass

    if _HAS_YAML:
        out_path = os.path.join(out_dir, filename_yaml)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    else:
        out_path = os.path.join(out_dir, "config.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    logger.info("已保存配置快照 -> %s", out_path)


# -----------------------------
# 名称与索引工具
# -----------------------------
def make_index_with_control_first(
    names: Iterable[str],
    control_name: Optional[str] = None
) -> Tuple[List[str], Dict[str, int]]:
    """
    根据名称列表创建索引（index map），并可选择将 control 名称置于首位。

    Parameters
    ----------
    names : Iterable[str]
        名称（基因或扰动）。
    control_name : Optional[str]
        若提供，则将该名称（如 "non-targeting"）置于首位。

    Returns
    -------
    Tuple[List[str], Dict[str, int]]
        (ordered_names, name_to_id)
    """
    names = sorted(set(names))
    if control_name and control_name in names:
        names.remove(control_name)
        names.insert(0, control_name)
    name_to_id = {n: i for i, n in enumerate(names)}
    return names, name_to_id
