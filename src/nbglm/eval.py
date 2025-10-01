# src/nbglm/eval.py
# -*- coding: utf-8 -*-
"""
评估指标（Evaluation Metrics）与统一评估入口
==========================================

提供三个指标：
- MAE（Mean Absolute Error, 平均绝对误差）
- PDS（Perturbation Distinguishability Score, 扰动区分分数）
- DES（Differential Expression Score, 差异表达分数）

评估流程（Overview）
-------------------
1) 读入预测 AnnData（pred_adata）与真实 AnnData（true_adata）。
2) （如需）做标准化与 log1p（Normalization + log1p）。
3) 计算基于**扰动均值表达**（mean profiles）的 MAE 与 PDS。
4) 使用 Scanpy 的 `rank_genes_groups` 计算 DE 基因，对比真实 DE（可缓存）得到 DES。
5) 汇总并保存 `metrics.json`（可在 `pipelines.py` 调用）。

数学（MAE 与 PDS）
------------------
- 令 $\\bar{x}^{\\text{pred}}_g(p)$ 为预测中扰动 $p$ 的基因 $g$ 的**平均表达**；
  $\\bar{x}^{\\text{true}}_g(p)$ 为真实数据对应的平均表达。
  则
  $$
  \\mathrm{MAE}
  = \\frac{1}{|\\mathcal{P}|}\\sum_{p\\in\\mathcal{P}}\\left(
    \\frac{1}{G}\\sum_{g=1}^G\\big|\\bar{x}^{\\text{pred}}_g(p)-\\bar{x}^{\\text{true}}_g(p)\\big|
  \\right).
  $$

- PDS 直观目标是：对每个扰动 $p$，以**L1 距离（cityblock）**在预测与真实的扰动均值空间中检索最相近的真实扰动，
  并看**真值扰动**的排名是否靠前。我们使用
  $$
  \\mathrm{PDS} = \\frac{1}{|\\mathcal{P}|}\\sum_{p\\in\\mathcal{P}} \\left( 1 - \\frac{\\operatorname{rank}(p)-1}{|\\mathcal{P}|} \\right) .
  $$

依赖（Dependencies）
-------------------
- anndata, numpy, pandas, scanpy (sc), scipy, joblib, tqdm
- 本模块只做**评估**，不涉及训练或采样。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import logging
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import json_dump


logger = logging.getLogger("nbglm.eval")


# -----------------------------
# 标准化辅助
# -----------------------------
def _normalize_if_needed(adata: ad.AnnData, label: str) -> ad.AnnData:
    """按照旧脚本逻辑始终执行 total-count 归一化与 log1p。"""
    adata = adata.copy()
    logger.info(f"检测到 '{label}' 数据为原始计数。正在执行标准化和 log1p 转换...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


# -----------------------------
# 均值表达（mean profiles）
# -----------------------------
def _mean_profiles(
    adata: ad.AnnData,
    pert_col: str,
    control_name: str,
    pert_list: List[str],
    genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """与旧版评估脚本保持一致的扰动均值表达计算。"""
    if genes is None:
        genes = list(adata.var_names)
    profiles: Dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctrl = adata[adata.obs[pert_col] == control_name]
        profiles[control_name] = np.asarray(ctrl.X.mean(axis=0)).flatten()
        for gene in pert_list:
            sub = adata[adata.obs[pert_col] == gene]
            profiles[gene] = np.asarray(sub.X.mean(axis=0)).flatten()
    return pd.DataFrame(profiles, index=genes).T


# -----------------------------
# 指标：MAE / PDS
# -----------------------------
# def mae_score(pred_profiles: pd.DataFrame, true_profiles: pd.DataFrame, pert_list: List[str]) -> float:
#     vals = []
#     for p in pert_list:
#         if p in pred_profiles.index and p in true_profiles.index:
#             vals.append(np.mean(np.abs(pred_profiles.loc[p].values - true_profiles.loc[p].values)))
#     return float(np.mean(vals)) if vals else 0.0
def mae_score(pred_profiles: pd.DataFrame, true_profiles: pd.DataFrame, pert_list: List[str]) -> float:
    """完全按照旧脚本的 MAE 实现。"""
    mae_scores = [
        np.mean(np.abs(pred_profiles.loc[g] - true_profiles.loc[g]))
        for g in pert_list
        if g in pred_profiles.index and g in true_profiles.index
    ]
    return float(np.mean(mae_scores)) if mae_scores else 0.0


def pds_score(pred_profiles: pd.DataFrame, true_profiles: pd.DataFrame, pert_list: List[str]) -> float:
    if not pert_list:
        return 0.0
    pred_pert = pred_profiles.loc[pert_list]
    true_pert = true_profiles.loc[pert_list]
    dist_matrix = cdist(pred_pert.values, true_pert.values, metric="cityblock")
    for i, p_gene in enumerate(pert_list):
        correction_vector = np.abs(pred_pert.at[p_gene, p_gene] - true_pert[p_gene].values)
        dist_matrix[i, :] -= correction_vector
    dist_df = pd.DataFrame(dist_matrix, index=pert_list, columns=pert_list)
    ranks = [
        np.where(dist_df.loc[p].sort_values().index == p)[0][0] + 1
        for p in pert_list
    ]
    N = len(pert_list)
    scores = [1.0 - (rank - 1) / N for rank in ranks]
    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# DES（差异表达）
# -----------------------------
def _de_genes(adata_slice: ad.AnnData, pert_gene: str, pert_col: str, control_name: str) -> pd.DataFrame:
    """
    对指定扰动（与 control）计算 DE 基因，使用 `scanpy.tl.rank_genes_groups(..., method="wilcoxon")`。

    Returns
    -------
    pd.DataFrame
        包含列 ['names', 'pvals_adj', 'logfoldchanges']，已过滤 pvals_adj 无效项。
    """
    adata_local = adata_slice.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(
            adata_local,
            groupby=pert_col,
            groups=[pert_gene],
            reference=control_name,
            method="wilcoxon",
            use_raw=False,
            n_genes=adata_local.shape[1],
        )
        res = adata_local.uns["rank_genes_groups"]
        de_df = pd.DataFrame({
            "names": res["names"][pert_gene],
            "pvals_adj": res["pvals_adj"][pert_gene],
            "logfoldchanges": res["logfoldchanges"][pert_gene],
        })
    de_df = de_df.replace([np.inf, -np.inf], np.nan).dropna()
    return de_df[de_df["pvals_adj"] < 0.05]


def _legacy_cache_path(cache_dir: Optional[str]) -> str:
    if cache_dir:
        if os.path.isdir(cache_dir):
            return os.path.join(cache_dir, "true_de_cache.pkl")
        return cache_dir
    env_path = os.environ.get("NBGLM_TRUE_DE_CACHE")
    if env_path:
        return env_path
    return "/home/wzc26/work/vcc/NB/true_de_cache.pkl"


def _get_ground_truth_de_genes_cached(
    true_adata: ad.AnnData,
    pert_list: List[str],
    pert_col: str,
    control_name: str,
    n_jobs: int,
    cache_dir: Optional[str],
) -> Dict[str, pd.DataFrame]:
    cache_path = _legacy_cache_path(cache_dir)
    if cache_path and os.path.exists(cache_path):
        logger.info(f"发现缓存文件，正在加载: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"未找到缓存。正在为 {len(pert_list)} 个扰动基因计算真实DE基因 (此过程仅需一次)...")
    tasks = [
        (
            gene,
            true_adata[(true_adata.obs[pert_col] == gene) | (true_adata.obs[pert_col] == control_name)],
        )
        for gene in pert_list
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(lambda name, sl: (name, _de_genes(sl, name, pert_col, control_name)))(name, sl)
        for name, sl in tqdm(tasks, desc="计算真实DE基因")
    )
    true_de_map = dict(results)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        logger.info(f"正在保存DE结果到缓存: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(true_de_map, f)
    return true_de_map


def des_score(
    pred_adata: ad.AnnData,
    true_adata: ad.AnnData,
    pert_col: str,
    control_name: str,
    n_jobs: int,
    cache_dir: Optional[str] = None
) -> float:
    """
    DES 计算流程：
    1) 为**真实数据**缓存每个扰动的 DE 基因集合（p_adj<0.05）。
    2) 对**预测数据**，对每个扰动计算 DE 集合，并与真实集合做**交集占比**。
    3) 求各扰动的平均值。

    为提升稳定性，当真实 DE 数量为 0 时，返回 1.0（视为无可区分差异）。

    Parameters
    ----------
    n_jobs : int
        并行核数（<=0 则退化为 1；"auto" 已在外部解析为合理整数）。
    """
    if n_jobs <= 0:
        n_jobs = 1

    pert_list = sorted([p for p in set(true_adata.obs[pert_col]) if p != control_name])
    true_de_map = _get_ground_truth_de_genes_cached(
        true_adata=true_adata,
        pert_list=pert_list,
        pert_col=pert_col,
        control_name=control_name,
        n_jobs=n_jobs,
        cache_dir=cache_dir,
    )

    logger.info(f"正在使用 {n_jobs} 个CPU核心并行计算 DES...")

    def _compare_des_for_one_gene(pert_gene: str) -> float:
        true_de_df = true_de_map.get(pert_gene, pd.DataFrame(columns=["names", "pvals_adj", "logfoldchanges"]))
        true_de_genes = set(true_de_df["names"])
        n_k_true = len(true_de_genes)

        if n_k_true == 0:
            return 1.0

        pred_slice = pred_adata[(pred_adata.obs[pert_col] == pert_gene) | (pred_adata.obs[pert_col] == control_name)]
        full_pred_de_df = _de_genes(pred_slice, pert_gene, pert_col, control_name)

        if len(full_pred_de_df) > n_k_true:
            sorted_pred = full_pred_de_df.reindex(
                full_pred_de_df.logfoldchanges.abs().sort_values(ascending=False).index
            )
            final_pred_genes = set(sorted_pred.head(n_k_true)["names"])
        else:
            final_pred_genes = set(full_pred_de_df["names"])

        intersection_size = len(final_pred_genes.intersection(true_de_genes))
        return intersection_size / n_k_true

    scores = Parallel(n_jobs=n_jobs)(
            delayed(_compare_des_for_one_gene)(gene) for gene in pert_list
        )

    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# 统一评估入口
# -----------------------------
def evaluate(
    pred_adata_or_path: Union[str, ad.AnnData],
    true_adata_or_path: str,
    pert_col: str,
    control_name: str,
    metrics: List[str],
    control_adata_path: str = None,
    run_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
    n_jobs: Union[int, str] = "auto",
    normalize: bool = True,
    save_json: bool = True
) -> Dict[str, float]:
    """
    统一评估入口。

    Parameters
    ----------
    pred_adata_or_path : Union[str, anndata.AnnData]
        预测结果（AnnData 或 .h5ad 路径）。
    true_adata_or_path : str
        真实数据 .h5ad 路径。
    pert_col : str
        扰动列名（如 "target_gene"）。
    control_name : str
        控制组名称（如 "non-targeting"）。
    metrics : List[str]
        需要计算的指标子集（["MAE","PDS","DES"] 的任意子集）。
    run_dir : Optional[str]
        运行目录（用于 DES 的真值缓存与结果保存）。
    n_jobs : Union[int, str]
        DES 的并行核数（"auto" => 80% CPU）。
    cache_path : Optional[str]
        预计算的真实 DE 缓存文件或目录；若为目录则自动附加文件名。
    normalize : bool
        是否对 pred/true 执行 normalize_total + log1p。
    save_json : bool
        是否把指标落盘为 JSON（metrics/metrics.json）。

    Returns
    -------
    Dict[str, float]
        指标字典。
    """
    # 载入
    pred_adata = ad.read_h5ad(pred_adata_or_path) if isinstance(pred_adata_or_path, str) else pred_adata_or_path
    true_adata = ad.read_h5ad(true_adata_or_path)

    # 对齐基因顺序
    common_genes = pred_adata.var_names.intersection(true_adata.var_names)
    pred_adata = pred_adata[:, common_genes].copy()
    true_adata = true_adata[:, common_genes].copy()

    # 归一化
    if normalize:
        pred_adata = _normalize_if_needed(pred_adata, "预测")
        true_adata = _normalize_if_needed(true_adata, "真实")

    # 预备
    pert_list = sorted([p for p in set(true_adata.obs[pert_col]) if p != control_name])
    pred_profiles = _mean_profiles(pred_adata, pert_col, control_name, pert_list, list(common_genes))
    true_profiles = _mean_profiles(true_adata, pert_col, control_name, pert_list, list(common_genes))

    # auto n_jobs
    if isinstance(n_jobs, str) and n_jobs.lower() == "auto":
        try:
            import multiprocessing as mp
            n_jobs = max(1, int(mp.cpu_count() * 0.8))
        except Exception:
            n_jobs = 1

    out: Dict[str, float] = {}

    # MAE
    if "MAE" in metrics:
        out["MAE"] = mae_score(pred_profiles, true_profiles, pert_list)
        logger.info(f"[eval] MAE: {out['MAE']:.6f}")

    # PDS
    if "PDS" in metrics:
        out["PDS"] = pds_score(pred_profiles, true_profiles, pert_list)
        logger.info(f"[eval] PDS: {out['PDS']:.6f}")

    # DES
    if "DES" in metrics:
        out["DES"] = des_score(
            pred_adata=pred_adata,
            true_adata=true_adata,
            pert_col=pert_col,
            control_name=control_name,
            n_jobs=int(n_jobs),
            cache_dir=cache_path if cache_path is not None else run_dir,
        )
        logger.info(f"[eval] DES: {out['DES']:.6f}")

    # 综合得分
    if all(k in out for k in ("DES", "PDS", "MAE")):
        des_baseline = 0.106
        pds_baseline = 0.516
        mae_baseline = 0.027

        
        overall = (out["DES"] - des_baseline) / (1 - des_baseline) + \
                  (out["PDS"] - pds_baseline) / (1 - pds_baseline) + \
                  np.clip((mae_baseline - out["MAE"]) / mae_baseline, 0, 1)
        overall *= 100 / 3
        out["Overall"] = overall
        logger.info(f"[eval] 综合得分 (基于 DES, PDS and MAE): {out['Overall']:.6f}")

        overall_1 = (out["DES"] - des_baseline) / (1 - des_baseline) + (out["PDS"] - pds_baseline) / (1 - pds_baseline)
        overall_1 *= 100 / 3
        out["Overall_wo_MAE"] = overall_1
        logger.info(f"[eval] 综合得分 (仅基于 DES and PDS): {out['Overall_wo_MAE']:.6f}")

    # 保存
    if save_json and run_dir is not None:
        metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        json_dump(os.path.join(metrics_dir, "metrics.json"), out)
        logger.info(f"[eval] 已保存指标到 {os.path.join(metrics_dir, 'metrics.json')}")

    return out
