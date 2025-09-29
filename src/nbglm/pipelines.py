# src/nbglm/pipelines.py
# -*- coding: utf-8 -*-
"""
训练 / 采样 / 评估 的流水线（Pipelines）
======================================

此模块把各个功能模块（data_io / dataset / model / eval）进行**编排**，提供简洁的上层接口：
- run_train(cfg, run_dirs)             -> {"model":..., "meta":..., "ckpt_path":...}
- run_sample(cfg, run_dirs, ...)       -> {"pred_adata_path": 或 "pred_adata": AnnData}
- run_evaluate(cfg, run_dirs, ...)     -> {"metrics": {...}}
- 组合模式：run_train_sample_eval / run_train_sample / run_sample_eval / run_sample / run_evaluate_only

设计要点（Design Notes）
-----------------------
- **内存直连**：由 cfg.pipeline.persist_intermediate 控制；false 时，采样产物不落盘，直接传递给评估函数。
- **幂等产物**：若持久化，则统一落在 `run_dirs` 指定的结构下（ckpt/preds/metrics）。
- **容错**：该层只做编排，不做重度逻辑；细节在下层模块中实现。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union, Any, List
import os
import json
import time

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import DataLoader

from . import data_io
from . import dataset as dset
from . import model as mdl
from . import eval as ev
from .utils import set_seed, get_device, get_logger, ensure_dir


# -----------------------------
# 内部小工具
# -----------------------------
def _choose_train_h5ad(cfg: dict) -> str:
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    if data_cfg.get("use_split", True):
        return paths.get("train_split_h5ad", paths.get("train_h5ad"))
    return paths.get("train_h5ad")


def _load_training_artifacts(cfg: dict) -> Tuple[ad.AnnData, ad.AnnData, str]:
    """
    读取训练期 h5ad，并分出 control / pert 两个子集。
    返回：(adata_all, adata_pert, pert_col_name)
    """
    pert_col = cfg["data"]["pert_name_col"]
    control_name = cfg["data"]["control_name"]
    h5ad_path = _choose_train_h5ad(cfg)
    adata_all = data_io.read_h5ad(h5ad_path)
    if pert_col not in adata_all.obs.columns:
        raise KeyError(f"[pipelines] 训练数据缺少列 '{pert_col}'")
    adata_pert = adata_all[adata_all.obs[pert_col] != control_name].copy()
    return adata_all, adata_pert, pert_col


def _prepare_embeddings_and_maps(cfg: dict, adata_all: ad.AnnData, df_val: pd.DataFrame) -> Dict[str, Any]:
    """
    构造有序的 gene / pert 名称列表，加载并对齐 embeddings，返回必要的映射。
    """
    pert_col = cfg["data"]["pert_name_col"]
    control_name = cfg["data"]["control_name"]

    gene_names = adata_all.var_names.tolist()
    train_perts = set(adata_all.obs[pert_col].unique())
    val_perts = set(df_val[pert_col].tolist()) if pert_col in df_val.columns else set(df_val["target_gene"].tolist())
    all_perts = sorted(train_perts | val_perts)

    # 控制组置前
    perts_ordered, pert2id = data_io.make_index_with_control_first(all_perts, control_name)

    # 加载嵌入并 L2 normalize
    G, P = data_io.load_embeddings(
        cfg["paths"]["gene_embedding_csv"],
        cfg["paths"]["pert_embedding_csv"],
        gene_names,
        perts_ordered,
        l2_normalize=True,
    )
    return {
        "gene_names": gene_names,
        "perts_ordered": perts_ordered,
        "pert2id": pert2id,
        "G": G,
        "P": P,
    }


def _build_training_dataloader(cfg: dict, adata_all: ad.AnnData, adata_pert: ad.AnnData, pert2id: Dict[str, int]) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    根据 cfg.train.fit_mode 选择 WholeCell / PseudoBulk 的 DataLoader。
    并返回训练需要的 meta（ref_depth / sf_ctrl 等）。
    """
    pert_col = cfg["data"]["pert_name_col"]
    control_name = cfg["data"]["control_name"]

    # tensors
    X_pert = dset.to_tensor(adata_pert.X)          # [N, G] CPU
    X_ctrl = dset.to_tensor(adata_all[adata_all.obs[pert_col] == control_name].X)  # [N_ctrl, G]

    # size factor
    use_sf = bool(cfg.get("size_factor", {}).get("use_sf", True))
    sf_ctrl, ref_depth = dset.compute_size_factors(X_ctrl)
    sf_pert, _ = dset.compute_size_factors(X_pert, ref_depth)

    # mu_control & theta
    if use_sf:
        X_ctrl_norm = X_ctrl / sf_ctrl.unsqueeze(1)
        mu_control = X_ctrl_norm.mean(dim=0)
        theta_vec = dset.estimate_theta_per_gene(X_ctrl_norm)
    else:
        mu_control = X_ctrl.mean(dim=0)
        theta_vec = dset.estimate_theta_per_gene(X_ctrl)

    # phase
    use_cycle = bool(cfg.get("model", {}).get("use_cycle", False))
    if use_cycle:
        phase_col = cfg["data"].get("phase_column", "phase")
        if phase_col not in adata_pert.obs.columns:
            raise KeyError(f"[pipelines] use_cycle=True 但训练数据缺少列 '{phase_col}'")
        phase_ids_train = dset.phases_to_ids(adata_pert.obs[phase_col].tolist())
    else:
        phase_ids_train = None

    # pert ids
    pert_ids_train = torch.tensor([pert2id[p] for p in adata_pert.obs[pert_col].tolist()], dtype=torch.long)

    # dataloader
    bs = int(cfg["train"].get("batch_size", 2048))
    fit_mode = cfg["train"].get("fit_mode", "concise").lower()

    if fit_mode == "whole":
        ds = dset.WholeCellDataset(
            X_tensor=X_pert,
            pert_ids=pert_ids_train,
            sf=(sf_pert if use_sf else None),
            use_sf=use_sf,
            use_cycle=use_cycle,
            phase_ids=phase_ids_train,
        )
        loader = DataLoader(ds, batch_size=max(1, bs // 2), shuffle=True, drop_last=False)
    elif fit_mode == "concise":
        Y_avg, unique_perts_eff, phase_ids_eff, log_s_eff = dset.build_pseudobulk(
            X_pert_train=X_pert,
            pert_ids_train=pert_ids_train,
            ref_depth=ref_depth,
            use_sf=use_sf,
            use_cycle=use_cycle,
            phase_ids_train=phase_ids_train,
            batch_size=bs * 2,
        )
        ds = dset.PseudoBulkDataset(
            Y_avg=Y_avg,
            pert_ids_eff=unique_perts_eff,
            use_cycle=use_cycle,
            phase_ids_eff=phase_ids_eff,
            log_s_eff=log_s_eff,
        )
        loader = DataLoader(ds, batch_size=max(1, bs // 2), shuffle=True, drop_last=False)
    else:
        raise ValueError(f"[pipelines] 未知 fit_mode: {fit_mode}")

    meta = {
        "use_sf": use_sf,
        "use_cycle": use_cycle,
        "ref_depth": ref_depth,     # scalar tensor
        "sf_ctrl": sf_ctrl,         # [N_ctrl]
        "mu_control": mu_control,   # [G]
        "theta_vec": theta_vec,     # [G]
    }
    return loader, meta


def _instantiate_model(cfg: dict, G: torch.Tensor, P: torch.Tensor, mu_control: torch.Tensor, theta_vec: torch.Tensor) -> mdl.LowRankNB_GLM:
    use_cycle = bool(cfg.get("model", {}).get("use_cycle", False))
    net = mdl.LowRankNB_GLM(
        gene_emb=G,
        pert_emb=P,
        mu_control=mu_control,
        theta_per_gene=theta_vec,
        use_cycle=use_cycle,
    )
    return net


# -----------------------------
# 训练
# -----------------------------
def run_train(cfg: dict, run_dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    训练一次模型；若 cfg.pipeline.persist_intermediate=True，将保存 ckpt 到 ckpt_dir。

    Returns
    -------
    Dict[str, Any]
        { "model": model_instance, "meta": {...}, "ckpt_path": Optional[str] }
    """
    set_seed(int(cfg["experiment"].get("seed", 2025)))

    # 读取训练数据与验证列表
    adata_all, adata_pert, pert_col = _load_training_artifacts(cfg)
    df_val = pd.read_csv(cfg["paths"]["val_list_csv"])

    # 嵌入与映射
    emb = _prepare_embeddings_and_maps(cfg, adata_all, df_val)
    G, P = emb["G"], emb["P"]

    # dataloader + 统计量
    loader, meta = _build_training_dataloader(cfg, adata_all, adata_pert, emb["pert2id"])

    # 模型与训练
    model = _instantiate_model(cfg, G, P, meta["mu_control"], meta["theta_vec"])
    loss_name = cfg["model"]["losses"]["primary"]
    model.fit(
        dataloader=loader,
        loss_type=loss_name,
        learning_rate=float(cfg["train"].get("lr", 5e-4)),
        n_epochs=int(cfg["train"].get("epochs", 100)),
        l1_lambda=float(cfg["model"]["regularization"].get("l1", 1e-4)),
        l2_lambda=float(cfg["model"]["regularization"].get("l2", 5e-3)),
        progress=True,
    )

    # 可选：保存 ckpt（包含必要常量，保证采样期无需再读 embeddings）
    ckpt_path = None
    if bool(cfg["pipeline"].get("persist_intermediate", True)):
        ckpt_path = os.path.join(run_dirs["ckpt_dir"], "model.pt")
        os.makedirs(run_dirs["ckpt_dir"], exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "G": G.cpu(),
            "P": P.cpu(),
            "mu_control": meta["mu_control"].cpu(),
            "theta_vec": meta["theta_vec"].cpu(),
            "use_cycle": bool(cfg["model"].get("use_cycle", False)),
            "gene_names": emb["gene_names"],
            "pert_names": emb["perts_ordered"],
            "cfg": cfg,
        }, ckpt_path)
        print(f"[pipelines] 已保存模型到: {ckpt_path}")

    # 返回对象与元信息
    ret = {
        "model": model,
        "meta": {
            **meta,
            **emb,
            "pert_col": pert_col,
            "control_name": cfg["data"]["control_name"],
            "train_h5ad_path": _choose_train_h5ad(cfg),
        },
        "ckpt_path": ckpt_path,
    }
    return ret


# -----------------------------
# 采样
# -----------------------------
def _load_model_from_ckpt(ckpt_path: str) -> mdl.LowRankNB_GLM:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = mdl.LowRankNB_GLM(
        gene_emb=ckpt["G"],
        pert_emb=ckpt["P"],
        mu_control=ckpt["mu_control"],
        theta_per_gene=ckpt["theta_vec"],
        use_cycle=bool(ckpt.get("use_cycle", False)),
    )
    net.load_state_dict(ckpt["state_dict"])
    return net


def run_sample(
    cfg: dict,
    run_dirs: Dict[str, str],
    model: Optional[mdl.LowRankNB_GLM] = None,
    meta: Optional[Dict[str, Any]] = None,
    ckpt_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    进行一次采样推理（prediction + sampling）。
    - 若提供 `model` 与 `meta`，将直接使用内存对象；
    - 否则将从 `ckpt_path` 载入。

    Returns
    -------
    Dict[str, Any]
        - 若 persist_intermediate=True:
            { "pred_adata_path": str }
          否则：
            { "pred_adata": anndata.AnnData }
    """
    # 加载模型 & 元信息
    if model is None or meta is None:
        if not ckpt_path:
            # 允许从 cfg.pipeline.pretrained_ckpt 提供
            ckpt_path = cfg.get("pipeline", {}).get("pretrained_ckpt", None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError("[pipelines] 缺少 ckpt_path 或文件不存在；请设置 pipeline.pretrained_ckpt")
        model = _load_model_from_ckpt(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        meta = {
            "gene_names": ckpt["gene_names"],
            "perts_ordered": ckpt["pert_names"],
            "pert2id": {n: i for i, n in enumerate(ckpt["pert_names"])},
            "use_cycle": bool(ckpt.get("use_cycle", False)),
            "control_name": cfg["data"]["control_name"],
            "pert_col": cfg["data"]["pert_name_col"],
            "train_h5ad_path": _choose_train_h5ad(cfg),
        }

    # 读取验证列表 & 训练数据（用于 sf 与 phase 统计）
    df_val = pd.read_csv(cfg["paths"]["val_list_csv"])
    adata_all = data_io.read_h5ad(meta["train_h5ad_path"])
    pert_col = meta["pert_col"]
    control_name = meta["control_name"]
    adata_ctrl = adata_all[adata_all.obs[pert_col] == control_name].copy()
    adata_pert = adata_all[adata_all.obs[pert_col] != control_name].copy()

    # 构造测试 pert id 列表
    if pert_col in df_val.columns:
        val_names = df_val[pert_col].tolist()
        n_cells = df_val["n_cells"].tolist()
    else:
        # 兼容列名 "target_gene"
        val_names = df_val["target_gene"].tolist()
        n_cells = df_val["n_cells"].tolist()

    pert_ids_test_list: List[int] = []
    for name, n in zip(val_names, n_cells):
        if name not in meta["pert2id"]:
            raise KeyError(f"[pipelines] 验证集扰动 '{name}' 不在训练映射中！")
        pert_ids_test_list.extend([meta["pert2id"][name]] * int(n))
    pert_ids_test = torch.tensor(pert_ids_test_list, dtype=torch.long)

    # size factor：根据 control 分布 + df_val 的 median_umi 构造
    use_sf = bool(cfg.get("size_factor", {}).get("use_sf", True))
    if use_sf:
        X_ctrl = dset.to_tensor(adata_ctrl.X)
        sf_ctrl, ref_depth = dset.compute_size_factors(X_ctrl)  # 重新估计（与训练一致）
        sf_test = dset.build_validation_size_factors(df_val, sf_ctrl, ref_depth, seed=int(cfg["experiment"].get("seed", 2025)))
    else:
        sf_test = None

    # phase：按策略生成
    use_cycle = bool(cfg.get("model", {}).get("use_cycle", False))
    if use_cycle:
        phase_strategy = cfg["sampling"].get("phase_strategy", "global")
        phase_col = cfg["data"].get("phase_column", "phase")
        if phase_col in adata_all.obs.columns:
            global_probs = dset.compute_global_phase_probs(adata_all.obs[phase_col].tolist())
        else:
            global_probs = np.array([0.7, 0.15, 0.15], dtype=float)
        per_pert_probs = dset.compute_per_pert_phase_probs(adata_pert, pert_col) if phase_strategy == "control" else None
        phase_ids_list = dset.sample_validation_phases(df_val, phase_strategy, global_probs, per_pert_probs, seed=int(cfg["experiment"].get("seed", 2025)))
        phase_ids_test = torch.tensor(phase_ids_list, dtype=torch.long)
    else:
        phase_ids_test = None

    # 采样参数
    sp = cfg.get("sampling", {})
    sampled_counts = model.predict_and_sample(
        pert_ids_test=pert_ids_test,
        batch_size=int(sp.get("batch_size", 4096)),
        use_sf=use_sf,
        sf_test=sf_test,
        use_gamma=bool(sp.get("gamma_heterogeneity", {}).get("enable", True)),
        gamma_r0=float(sp.get("gamma_heterogeneity", {}).get("r0", 50.0)),
        sampler=str(sp.get("sampler", "poisson")),
        use_cycle=use_cycle,
        phase_ids_test=phase_ids_test,
    )
    sampled_counts = sampled_counts.cpu().numpy().astype(np.float32)

    # 组装 AnnData：与训练 var 对齐，并**附带 control 细胞**以便评估（与历史评估兼容）
    obs_pred = pd.DataFrame({pert_col: [meta["perts_ordered"][i] for i in pert_ids_test_list]})
    if use_cycle and phase_ids_test is not None:
        inv_map = {0: "G1", 1: "S", 2: "G2M"}
        obs_pred["phase"] = [inv_map.get(int(x), "G1") for x in phase_ids_test.tolist()]
    # 预测集
    ad_pred = ad.AnnData(X=sampled_counts, obs=obs_pred, var=pd.DataFrame(index=meta["gene_names"]).copy())
    # control 原样附带（只用于评估一致性；如不需要可删掉这一 concat）
    ad_ctrl = adata_ctrl[:, meta["gene_names"]].copy()
    ad_final = ad.concat([ad_ctrl, ad_pred], join="outer", index_unique=None)
    ad_final.X = ad_final.X.astype(np.float32)

    # 落盘 or 内存直连
    if bool(cfg["pipeline"].get("persist_intermediate", True)):
        out_path = os.path.join(run_dirs["pred_dir"], "pred.h5ad")
        ad_final.write(out_path)
        print(f"[pipelines] 已保存预测到: {out_path}")
        return {"pred_adata_path": out_path}
    else:
        return {"pred_adata": ad_final}


# -----------------------------
# 评估
# -----------------------------
def run_evaluate(cfg: dict, run_dirs: Dict[str, str], pred_adata_or_path: Union[str, ad.AnnData]) -> Dict[str, Any]:
    """
    调用统一评估入口，返回并保存 metrics。
    """
    ev_cfg = cfg.get("evaluate", {})
    if not bool(ev_cfg.get("enable", True)):
        print("[pipelines] 评估被禁用（evaluate.enable=false）。")
        return {"metrics": {}}

    metrics = list(ev_cfg.get("metrics", ["MAE", "PDS", "DES"]))
    true_h5ad = cfg["paths"]["test_h5ad"]
    pert_col = cfg["data"]["pert_name_col"]
    control_name = cfg["data"]["control_name"]
    n_jobs = ev_cfg.get("n_jobs", "auto")

    res = ev.evaluate(
        pred_adata_or_path=pred_adata_or_path,
        true_adata_or_path=true_h5ad,
        pert_col=pert_col,
        control_name=control_name,
        metrics=metrics,
        run_dir=run_dirs["run_dir"],
        n_jobs=n_jobs,
        normalize=True,
        save_json=True,
    )
    return {"metrics": res}


# -----------------------------
# 组合模式
# -----------------------------
def run_train_sample_eval(cfg: dict, run_dirs: Dict[str, str]) -> Dict[str, Any]:
    t = run_train(cfg, run_dirs)
    s = run_sample(cfg, run_dirs, model=t["model"], meta=t["meta"])
    p = s["pred_adata_path"] if "pred_adata_path" in s else s["pred_adata"]
    e = run_evaluate(cfg, run_dirs, pred_adata_or_path=p)
    return {"ckpt_path": t["ckpt_path"], **s, **e}


def run_train_sample(cfg: dict, run_dirs: Dict[str, str]) -> Dict[str, Any]:
    t = run_train(cfg, run_dirs)
    s = run_sample(cfg, run_dirs, model=t["model"], meta=t["meta"])
    return {"ckpt_path": t["ckpt_path"], **s}


def run_sample_eval(cfg: dict, run_dirs: Dict[str, str]) -> Dict[str, Any]:
    s = run_sample(cfg, run_dirs, model=None, meta=None, ckpt_path=cfg.get("pipeline", {}).get("pretrained_ckpt"))
    p = s["pred_adata_path"] if "pred_adata_path" in s else s["pred_adata"]
    e = run_evaluate(cfg, run_dirs, pred_adata_or_path=p)
    return {**s, **e}


def run_sample_only(cfg: dict, run_dirs: Dict[str, str]) -> Dict[str, Any]:
    return run_sample(cfg, run_dirs, model=None, meta=None, ckpt_path=cfg.get("pipeline", {}).get("pretrained_ckpt"))


def run_evaluate_only(cfg: dict, run_dirs: Dict[str, str]) -> Dict[str, Any]:
    # 支持从 cfg.paths.pred_h5ad 读取预测文件
    pred_path = cfg.get("paths", {}).get("pred_h5ad", None)
    if not pred_path or not os.path.exists(pred_path):
        raise FileNotFoundError("[pipelines] evaluate_only 模式需要提供 paths.pred_h5ad")
    return run_evaluate(cfg, run_dirs, pred_adata_or_path=pred_path)
