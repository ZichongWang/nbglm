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
import os
import json
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import json_dump


# -----------------------------
# 标准化辅助
# -----------------------------
def _normalize_if_needed(adata: ad.AnnData, label: str, verbose: bool = True) -> ad.AnnData:
    """
    如需则进行 total-count normalize + log1p。
    这里采取“总是做一次”的保守策略，以避免输入是原始计数时的量纲不匹配。

    Parameters
    ----------
    adata : anndata.AnnData
    label : str
        日志标签（"预测"/"真实"）。
    verbose : bool

    Returns
    -------
    anndata.AnnData
        归一化后的数据（原地修改的 copy）。
    """
    adata = adata.copy()
    if verbose:
        print(f"[eval] 对 '{label}' 数据执行 normalize_total + log1p ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.normalize_total(adata, target_sum=5e4)
        sc.pp.log1p(adata)
    return adata


# -----------------------------
# 均值表达（mean profiles）
# -----------------------------
def _mean_profiles(adata: ad.AnnData, pert_col: str, control_name: str, genes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    计算每个扰动（以及 control）的**基因均值表达**，[num_perts, G] 的 DataFrame。

    Returns
    -------
    pd.DataFrame
        index: perturbation name；columns: gene symbols（与输入 genes 对齐）
    """
    if genes is None:
        genes = list(adata.var_names)
    profiles = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # control
        ctrl = adata[adata.obs[pert_col] == control_name]
        profiles[control_name] = np.asarray(ctrl.X.mean(axis=0)).reshape(-1)
        # perts
        for p in sorted(set(adata.obs[pert_col]) - {control_name}):
            sub = adata[adata.obs[pert_col] == p]
            profiles[p] = np.asarray(sub.X.mean(axis=0)).reshape(-1)
    df = pd.DataFrame(profiles, index=genes).T
    return df


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
    vals = []
    for p in pert_list:
        if p in pred_profiles.index and p in true_profiles.index:
            # 关键：按列名对齐
            a = pred_profiles.loc[p]
            b = true_profiles.loc[p].reindex(a.index)  # 以 a 的列顺序对齐
            vals.append(float(np.abs(a - b).mean()))
    return float(np.mean(vals)) if vals else 0.0


def pds_score(pred_profiles: pd.DataFrame, true_profiles: pd.DataFrame, pert_list: List[str]) -> float:
    """
    使用 L1 距离矩阵并计算“真值扰动的检索名次”得分。
    """
    if not pert_list:
        return 0.0
    pred_pert = pred_profiles.loc[pert_list]
    true_pert = true_profiles.loc[pert_list]
    dist = cdist(pred_pert.values, true_pert.values, metric="cityblock")
    for i, p_gene in enumerate(pert_list):
        if p_gene in pred_pert.columns and p_gene in true_pert.columns:
            correction_vector = np.abs(pred_pert[p_gene].loc[p_gene] - true_pert[p_gene].values)
            # shape: (n_perts, ); 减到 dist 的第 i 行
            dist[i, :] -= correction_vector
    dist_df = pd.DataFrame(dist, index=pert_list, columns=pert_list)
    scores = []
    for p in pert_list:
        rank = np.where(dist_df.loc[p].sort_values().index == p)[0][0] + 1  # 1-based
        N = len(pert_list)
        scores.append(1.0 - (rank - 1) / N)
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
    return de_df


def _true_de_cache_path(run_dir: str) -> str:
    return os.path.join(run_dir, "metrics", "true_de_cache.pkl")


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
    # --- 1) 真值 DE 缓存 ---
    cache_path = None if cache_dir is None else _true_de_cache_path(cache_dir)
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            import pickle
            true_de_map = pickle.load(f)
        print(f"[eval] 载入真实 DE 缓存: {cache_path}")
    else:
        print(f"[eval] 计算真实 DE（{len(pert_list)} 个扰动） ...")
        tasks = []
        for p in pert_list:
            sl_true = true_adata[(true_adata.obs[pert_col] == p) | (true_adata.obs[pert_col] == control_name)]
            tasks.append((p, sl_true))
        def _worker(name, adata_slice):
            return name, _de_genes(adata_slice, name, pert_col, control_name)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_worker)(name, slice_) for name, slice_ in tqdm(tasks, desc="True DE")
        )
        true_de_map = {k: v for k, v in results}
        if cache_path:
            try:
                import pickle
                with open(cache_path, "wb") as f:
                    pickle.dump(true_de_map, f)
                print(f"[eval] 已缓存真实 DE 至: {cache_path}")
            except Exception as e:
                print(f"[eval][警告] 缓存真实 DE 失败：{e}")

    # --- 2) 预测 DE 并对比 ---
    print(f"[eval] 计算预测 DES ...")
    def _one(p):
        true_de_df = true_de_map.get(p, pd.DataFrame(columns=["names", "pvals_adj", "logfoldchanges"]))
        true_set = set(true_de_df.loc[true_de_df["pvals_adj"] < 0.05, "names"])
        n_k_true = len(true_set)
        if n_k_true == 0:
            return 1.0
        sl_pred = pred_adata[(pred_adata.obs[pert_col] == p) | (pred_adata.obs[pert_col] == control_name)]
        pred_de_df = _de_genes(sl_pred, p, pert_col, control_name)
        # 截断到与真实 DE 数量一致（按 |logFC| 排序）
        pred_de_df = pred_de_df.sort_values(by="logfoldchanges", key=lambda s: s.abs(), ascending=False)
        pred_set = set(pred_de_df.head(n_k_true)["names"])
        inter = len(pred_set & true_set)
        return inter / (n_k_true if n_k_true > 0 else 1.0)

    scores = Parallel(n_jobs=n_jobs)(delayed(_one)(p) for p in tqdm(pert_list, desc="DES"))
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
        pred_adata = _normalize_if_needed(pred_adata, "预测", verbose=True)
        true_adata = _normalize_if_needed(true_adata, "真实", verbose=True)

    # 预备
    pert_list = sorted([p for p in set(true_adata.obs[pert_col]) if p != control_name])
    pred_profiles = _mean_profiles(pred_adata, pert_col, control_name, list(common_genes))
    true_profiles = _mean_profiles(true_adata, pert_col, control_name, list(common_genes))

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
        print(f"[eval] MAE: {out['MAE']:.6f}")

    # PDS
    if "PDS" in metrics:
        out["PDS"] = pds_score(pred_profiles, true_profiles, pert_list)
        print(f"[eval] PDS: {out['PDS']:.6f}")

    # DES
    if "DES" in metrics:
        out["DES"] = des_score(
            pred_adata=pred_adata,
            true_adata=true_adata,
            pert_col=pert_col,
            control_name=control_name,
            n_jobs=int(n_jobs),
            cache_dir=run_dir,
        )
        print(f"[eval] DES: {out['DES']:.6f}")

    # 综合得分
    if all(k in out for k in ("DES", "PDS", "MAE")):
        des_baseline = 0.106
        pds_baseline = 0.516
        mae_baseline = 0.027
        overall = (out["DES"] - des_baseline) / (1 - des_baseline) + (out["PDS"] - pds_baseline) / (1 - pds_baseline) + ((mae_baseline - out["MAE"]) / mae_baseline).clip(0, 1)
        overall *= 100 / 3
        out["Overall"] = overall
        print(f"[eval] 综合得分 (基于 DES, PDS and MAE): {out['Overall']:.6f}")

        overall_1 = (out["DES"] - des_baseline) / (1 - des_baseline) + (out["PDS"] - pds_baseline) / (1 - pds_baseline)
        overall_1 *= 100 / 3
        out["Overall_wo_MAE"] = overall_1
        print(f"[eval] 综合得分 (仅基于 DES and PDS): {out['Overall_wo_MAE']:.6f}")

    # 保存
    if save_json and run_dir is not None:
        metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        json_dump(os.path.join(metrics_dir, "metrics.json"), out)
        print(f"[eval] 已保存指标到 {os.path.join(metrics_dir, 'metrics.json')}")

    return out
