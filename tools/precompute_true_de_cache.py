# tools/precompute_true_de_cache.py
# -*- coding: utf-8 -*-
"""预计算真实数据集所有扰动的 DE 基因缓存。

该脚本复用 legacy 评估逻辑：对真实 AnnData 中的每个扰动与 control
执行 `scanpy.tl.rank_genes_groups`（Wilcoxon），并保存 p 值校正 < 0.05 的基因。
生成的缓存可被工程版评估（src/nbglm/eval.py）直接读取，避免重复计算。
"""

from __future__ import annotations

import argparse
import os
import pickle
import warnings
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from tqdm import tqdm


def _de_genes(adata_slice: ad.AnnData, pert: str, pert_col: str, control_name: str) -> pd.DataFrame:
    """与 eval.py 中保持一致的 DE 计算流程。"""
    local = adata_slice.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(
            local,
            groupby=pert_col,
            groups=[pert],
            reference=control_name,
            method="wilcoxon",
            use_raw=False,
            n_genes=local.shape[1],
        )
        res = local.uns["rank_genes_groups"]
        df = pd.DataFrame(
            {
                "names": res["names"][pert],
                "pvals_adj": res["pvals_adj"][pert],
                "logfoldchanges": res["logfoldchanges"][pert],
            }
        )
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df[df["pvals_adj"] < 0.05]


def main() -> None:
    parser = argparse.ArgumentParser(description="预计算真实数据的 DE 缓存")
    parser.add_argument("--h5ad", required=True, help="真实数据集（.h5ad）路径")
    parser.add_argument("--pert-col", default="target_gene", help="扰动列名，与评估接口一致")
    parser.add_argument("--control", default="non-targeting", help="control 名称")
    parser.add_argument(
        "--output",
        default="true_de_cache_all.pkl",
        help="缓存输出路径（默认与 legacy 评估保持一致）",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="并行核数（<=0 使用 80% CPU）")
    args = parser.parse_args()

    print(f"读取真实数据: {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    pert_col = args.pert_col
    control_name = args.control
    pert_list: List[str] = sorted(p for p in set(adata.obs[pert_col]) if p != control_name)
    if not pert_list:
        raise RuntimeError("数据集中未找到任何非 control 的扰动记录。")

    n_jobs = args.n_jobs if args.n_jobs > 0 else max(1, int(os.cpu_count() * 0.8))
    print(f"将使用 {n_jobs} 个并行 worker，为 {len(pert_list)} 个扰动计算 DE …")

    tasks = [
        (
            pert,
            adata[(adata.obs[pert_col] == pert) | (adata.obs[pert_col] == control_name)],
        )
        for pert in pert_list
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(lambda name, slice_: (name, _de_genes(slice_, name, pert_col, control_name)))(name, slice_)
        for name, slice_ in tqdm(tasks, desc="计算真实DE基因")
    )
    true_de_map: Dict[str, pd.DataFrame] = dict(results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(true_de_map, f)
    print(f"已保存缓存到: {args.output}")


if __name__ == "__main__":
    main()
