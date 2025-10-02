# src/nbglm/dataset.py
# -*- coding: utf-8 -*-
"""
数据集与统计预处理（Datasets & Preprocessing Utilities）
=====================================================

本模块提供：
- 将稀疏/稠密矩阵转为 torch.Tensor 的便捷函数
- size factor（文库大小因子, *size factor*）估计与验证集重采样构造
- 负二项分布（Negative Binomial, NB）离散度参数 theta 的矩估计（Method of Moments, MoM）
- 细胞周期（cell cycle, phase）相关工具：标签映射、全局/分扰动相对频率、验证集 phase 采样策略
- 训练数据集（WholeCellDataset）与伪批（pseudo-bulk）数据集（PseudoBulkDataset）
- 在 CPU 上构建伪批（pseudobulk）的高效函数（避免 GPU OOM）

数学记号（Mathematical Notation）
--------------------------------
- 令 $X \\in \\mathbb{N}^{N \\times G}$ 为计数矩阵（cells × genes）。
- 文库深度（library size）$L_i = \\sum_{g=1}^G X_{ig}$。
- size factor 定义为
  $$
  s_i \\;=\\; \\frac{L_i}{\\operatorname{median}(L)}.
  $$
  在 GLM 中以偏置项（offset, $\\log s_i$）加入：
  $$
  \\log \\mu_{ig}^{(\\mathrm{obs})} = \\log \\mu_{ig}^{(\\mathrm{ref})} + \\log s_i.
  $$
- NB 的 MoM 离散度估计：
  $$
  \\theta_g = \\frac{\\mu_g^2}{\\max(\\operatorname{Var}_g - \\mu_g, \\varepsilon)}.
  $$

依赖（Dependencies）
-------------------
- torch, numpy, pandas, anndata, tqdm
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import anndata as ad


# -----------------------------
# 常量与基础工具
# -----------------------------
PHASE2ID: Dict[str, int] = {"G1": 0, "S": 1, "G2M": 2}


def to_tensor(X) -> torch.Tensor:
    """
    将稀疏/稠密矩阵转换为 float32 的 torch.Tensor（在 CPU 上）。

    Parameters
    ----------
    X : scipy.sparse.spmatrix 或 numpy.ndarray 或 array-like
        计数矩阵或任意矩阵。

    Returns
    -------
    torch.Tensor
        dtype=float32, device=CPU
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    return torch.tensor(X, dtype=torch.float32)


# -----------------------------
# size factor（文库大小因子）
# -----------------------------
def compute_size_factors(
    X: torch.Tensor,
    ref_depth: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 size factor（s_i）与参照深度（reference depth）。

    定义（Definition）
    -----------------
    设第 i 个细胞的文库深度（UMI 总数）为 $L_i = \\sum_g X_{ig}$。
    参照深度 $L_\\mathrm{ref} = \\operatorname{median}(L)$。
    则 size factor：
    $$
    s_i = \\frac{L_i}{L_\\mathrm{ref}}.
    $$

    在 GLM 中（作为 offset）使用 $\\log s_i$，使模型输出的参考尺度均值
    $\\mu^{(\\mathrm{ref})}$ 映射回观测尺度 $\\mu^{(\\mathrm{obs})}$。

    Parameters
    ----------
    X : torch.Tensor
        [N, G] 计数矩阵（CPU）。
    ref_depth : Optional[torch.Tensor]
        可选的外部参照深度；若为 None，则使用当前 X 的 median(L)。
    eps : float
        数值下限，用于避免除零。

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (sf, ref_depth)
        - sf: [N] 的 size factor
        - ref_depth: 标量张量（median library size）
    """
    lib = X.sum(dim=1)
    if ref_depth is None:
        ref_depth = torch.median(lib)
    sf = (lib / (ref_depth + eps)).clamp(min=eps)
    return sf, ref_depth


def build_validation_size_factors(
    df_val: pd.DataFrame,
    sf_ctrl: torch.Tensor,
    ref_depth: torch.Tensor,
    seed: int = 2025
) -> torch.Tensor:
    """
    基于 control 的 sf 分布与验证集 `median_umi_per_cell` 约束，构造验证集 cells 的 size factor。

    思路（Idea）
    -----------
    - 从 control sf 分布中**重采样**（resample）出与每个扰动所需细胞数相同的样本；
    - 将该批样本的**中位数**缩放至 `target_med_sf = median_umi_per_cell / ref_depth`。

    Parameters
    ----------
    df_val : pd.DataFrame
        必含列：["target_gene", "n_cells", "median_umi_per_cell"]，行顺序决定输出顺序。
    sf_ctrl : torch.Tensor
        来自 control 细胞的 size factor（[N_ctrl]）。
    ref_depth : torch.Tensor
        参照深度（scalar tensor）。
    seed : int
        重采样随机种子。

    Returns
    -------
    torch.Tensor
        验证集 cells 的 size factor，长度 = sum(n_cells)。
    """
    rng = np.random.default_rng(seed)
    sfc = sf_ctrl.detach().cpu().numpy()
    out = []
    for _, row in df_val.iterrows():
        n = int(row["n_cells"])
        med_umi = float(row["median_umi_per_cell"])
        target_med_sf = med_umi / float(ref_depth.item())
        idx = rng.integers(0, len(sfc), size=n)
        samp = sfc[idx]
        cur_med = np.median(samp) if np.median(samp) > 0 else 1.0
        scaled = samp * (target_med_sf / cur_med)
        out.append(torch.tensor(scaled, dtype=torch.float32))
    return torch.cat(out, dim=0)


# -----------------------------
# NB 离散度参数 theta 的 MoM 估计
# -----------------------------
def estimate_theta_per_gene(X_ctrl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    对每个基因估计 NB 离散度参数 θ（theta），采用矩估计（Method of Moments, MoM）。

    公式（Formula）
    --------------
    对第 g 个基因，
    $$
    \\mu_g = \\operatorname{mean}(X_{\\cdot g}),\\quad
    \\operatorname{Var}_g = \\operatorname{var}(X_{\\cdot g})
    $$
    在 NB 模型中 $\\operatorname{Var}(Y) = \\mu + \\mu^2/\\theta$，
    故
    $$
    \\theta_g \\approx \\frac{\\mu_g^2}{\\max(\\operatorname{Var}_g - \\mu_g, \\varepsilon)}.
    $$

    当 $\\operatorname{Var}_g \\le \\mu_g$ 时，视为接近 Poisson（无过度离散），给出较大 θ。

    Parameters
    ----------
    X_ctrl : torch.Tensor
        控制组的计数或标准化计数（可先除以 size factor 后再估计），形状 [N_ctrl, G]。
    eps : float
        数值下限；避免除零与负方差导致的异常。

    Returns
    -------
    torch.Tensor
        [G] 的 θ 向量（clamp 到合理范围）。
    """
    mu_g = X_ctrl.mean(dim=0)
    var_g = X_ctrl.var(dim=0, unbiased=False)
    denominator = var_g - mu_g
    theta_g = torch.full_like(mu_g, 1e6)  # 大值 ≈ Poisson
    over_mask = denominator > eps
    theta_g[over_mask] = (mu_g[over_mask] * mu_g[over_mask]) / denominator[over_mask]
    theta_g = theta_g.clamp(min=1e-6, max=1e12)
    return theta_g


# -----------------------------
# Phase（细胞周期）工具
# -----------------------------
def phases_to_ids(phases: List[str]) -> torch.LongTensor:
    """
    将字符串 phase（"G1"/"S"/"G2M"）映射为 ID（0/1/2）。

    Returns
    -------
    torch.LongTensor
        [N]，取值 ∈ {0,1,2}，未知按 G1 处理（0）。
    """
    return torch.tensor([PHASE2ID.get(p, 0) for p in phases], dtype=torch.long)


def compute_global_phase_probs(phases: List[str]) -> np.ndarray:
    """
    计算全局 phase 频率分布（probabilities）。

    Returns
    -------
    np.ndarray
        形如 [p(G1), p(S), p(G2M)]。
    """
    counts = np.zeros(3, dtype=float)
    for p in phases:
        if p in PHASE2ID:
            counts[PHASE2ID[p]] += 1
    total = counts.sum()
    if total <= 0:
        return np.array([0.7, 0.15, 0.15], dtype=float)
    return counts / total


def compute_per_pert_phase_probs(adata_pert: ad.AnnData, pert_name_col: str) -> Dict[str, np.ndarray]:
    """
    计算**按扰动**的 phase 概率，用于 `phase_strategy="control"`。

    Parameters
    ----------
    adata_pert : anndata.AnnData
        含有训练集的扰动细胞。
    pert_name_col : str
        扰动列名（如 "target_gene"）。

    Returns
    -------
    Dict[str, np.ndarray]
        {pert_name: [pG1, pS, pG2M]}
    """
    df = adata_pert.obs[[pert_name_col, "phase"]].copy()
    out: Dict[str, np.ndarray] = {}
    for name, sub in df.groupby(pert_name_col):
        out[name] = compute_global_phase_probs(sub["phase"].tolist())
    return out


def sample_validation_phases(
    df_val: pd.DataFrame,
    phase_strategy: str,
    global_probs: np.ndarray,
    per_pert_probs: Optional[Dict[str, np.ndarray]],
    seed: int = 2025
) -> List[int]:
    """
    为验证集**按策略**生成 phase 序列（与 df_val 展开顺序一致）。

    策略（Strategies）
    ------------------
    - "ignore" / "fixed_G1": 全为 G1（基线）。
    - "fixed_S": 全为 S。
    - "fixed_G2M": 全为 G2M。
    - "global": 按全局比例采样。
    - "control": 若该扰动有训练期统计，则按其分布采样，否则退化为 global。

    Returns
    -------
    List[int]
        phase 的 ID 列表（0/1/2）。
    """
    rng = np.random.default_rng(seed)
    out: List[int] = []
    for _, row in df_val.iterrows():
        pert = str(row["target_gene"])
        n = int(row["n_cells"])
        if phase_strategy in ("ignore", "fixed_G1"):
            out.extend([PHASE2ID["G1"]] * n)
        elif phase_strategy == "fixed_S":
            out.extend([PHASE2ID["S"]] * n)
        elif phase_strategy == "fixed_G2M":
            out.extend([PHASE2ID["G2M"]] * n)
        elif phase_strategy == "global":
            out.extend(rng.choice([0, 1, 2], size=n, p=global_probs).tolist())
        elif phase_strategy == "control":
            probs = global_probs
            if per_pert_probs is not None and pert in per_pert_probs:
                probs = per_pert_probs[pert]
            out.extend(rng.choice([0, 1, 2], size=n, p=probs).tolist())
        else:
            # 未知策略 ⇒ global
            out.extend(rng.choice([0, 1, 2], size=n, p=global_probs).tolist())
    return out


# -----------------------------
# Dataset 定义
# -----------------------------
class WholeCellDataset(Dataset):
    """
    基于**单细胞级**训练的 Dataset。

    约定（Conventions）
    -------------------
    - 本数据集**不在内部改变**原始计数 `Y`；size factor 的影响通过 `log_s`（offset）提供给模型。
    - 若 `use_cycle=True`，则需提供 `phase_ids`。

    __getitem__ 返回字段（fields）
    -----------------------------
    - 'y'    : FloatTensor [G]
    - 'pert' : LongTensor  []
    - 'log_s': FloatTensor []
    - 'phase': LongTensor  []（可选）
    """
    def __init__(
        self,
        X_tensor: torch.Tensor,                 # [N, G] CPU
        pert_ids: torch.LongTensor,             # [N]
        sf: Optional[torch.Tensor] = None,      # [N] 若 use_sf=False 可为 None
        use_sf: bool = True,
        use_cycle: bool = False,
        phase_ids: Optional[torch.LongTensor] = None  # [N] ∈ {0,1,2}
    ):
        assert X_tensor.device.type == "cpu", "WholeCellDataset 仅支持 CPU 张量作为输入。"
        assert pert_ids.device.type == "cpu"
        self.Y = X_tensor

        self.use_sf = use_sf
        if use_sf:
            assert sf is not None and sf.device.type == "cpu"
            self.log_s = torch.log(sf.clamp_min(1e-12))
        else:
            self.log_s = torch.zeros(X_tensor.size(0), dtype=torch.float32)

        self.pert_ids = pert_ids
        self.use_cycle = use_cycle
        if use_cycle:
            assert phase_ids is not None and phase_ids.device.type == "cpu"
        self.phase_ids = phase_ids

    def __len__(self) -> int:
        return self.Y.shape[0]

    def __getitem__(self, idx: int):
        item = {
            "y": self.Y[idx],
            "pert": self.pert_ids[idx],
            "log_s": self.log_s[idx],
        }
        if self.use_cycle:
            item["phase"] = self.phase_ids[idx]
        return item


class PseudoBulkDataset(Dataset):
    """
    基于**伪批（pseudo-bulk）**聚合的 Dataset。

    模式（两种）
    -----------
    - 无周期：每个样本对应一个扰动的**平均表达** [G]。
    - 有周期：每个样本对应 (扰动 × phase) 的**平均表达** [G]。

    __getitem__ 返回字段（fields）
    -----------------------------
    - 'y'    : FloatTensor [G]
    - 'pert' : LongTensor  []
    - 'log_s': FloatTensor []（若未使用 size factor 则为 0）
    - 'phase': LongTensor  []（仅在 use_cycle=True 时）
    """
    def __init__(
        self,
        Y_avg: torch.Tensor,                         # [K, G] 或 [B_eff, G]
        pert_ids_eff: torch.LongTensor,              # 与 Y_avg 对齐
        use_cycle: bool = False,
        phase_ids_eff: Optional[torch.LongTensor] = None,
        log_s_eff: Optional[torch.Tensor] = None
    ):
        assert Y_avg.device.type == "cpu"
        assert pert_ids_eff.device.type == "cpu"
        self.Y = Y_avg
        self.pert_ids_eff = pert_ids_eff
        self.use_cycle = use_cycle
        if use_cycle:
            assert phase_ids_eff is not None and phase_ids_eff.device.type == "cpu"
        self.phase_ids_eff = phase_ids_eff
        self.log_s = log_s_eff if log_s_eff is not None else torch.zeros(Y_avg.size(0), dtype=torch.float32)

    def __len__(self) -> int:
        return self.Y.shape[0]

    def __getitem__(self, idx: int):
        item = {"y": self.Y[idx], "pert": self.pert_ids_eff[idx], "log_s": self.log_s[idx]}
        if self.use_cycle:
            item["phase"] = self.phase_ids_eff[idx]
        return item


# -----------------------------
# 伪批（pseudo-bulk）构建
# -----------------------------
@torch.no_grad()
def build_pseudobulk(
    X_pert_train: torch.Tensor,            # [N, G] CPU
    pert_ids_train: torch.LongTensor,      # [N]    CPU（映射到 0..K-1 的扰动索引）
    ref_depth: torch.Tensor,               # scalar tensor
    use_sf: bool,
    use_cycle: bool,
    phase_ids_train: Optional[torch.LongTensor],  # [N] ∈ {0,1,2}，当 use_cycle=True 必须提供
    batch_size: int = 4096
) -> Tuple[torch.Tensor, torch.LongTensor, Optional[torch.LongTensor], Optional[torch.Tensor]]:
    """
    在 CPU 上构建伪批（避免 GPU OOM）。将每个扰动（或扰动×phase）的细胞表达**平均**到一条样本。

    - 无周期时：
      返回 (Y_avg[K,G], unique_perts[K], None, log_s[K])
    - 有周期时：
      返回 (Y_avg_flat[B_eff,G], pert_ids_eff[B_eff], phase_ids_eff[B_eff], log_s[B_eff])

    size factor 的处理（作为 offset）：
    ----------------------------------
    对第 k 个聚合单元（扰动或扰动×phase），计算其平均文库深度
    $\\bar{L}_k = \\frac{1}{n_k} \\sum_{i \\in \\mathcal{I}_k} L_i$，
    并令
    $$
      s_k = \\frac{\\bar{L}_k}{L_\\mathrm{ref}},\\quad
      \\log s_k = \\log(\\max(s_k, 10^{-12})).
    $$
    下游模型 forward 可将 `log_s` 当作 offset 直接加到 $\\log \\mu$ 上。

    Returns
    -------
    Tuple[torch.Tensor, torch.LongTensor, Optional[torch.LongTensor], Optional[torch.Tensor]]
        (Y_avg, pert_ids_eff, phase_ids_eff, log_s_eff)
    """
    assert X_pert_train.device.type == "cpu"
    assert pert_ids_train.device.type == "cpu"
    if use_cycle:
        assert phase_ids_train is not None and phase_ids_train.device.type == "cpu"

    N, G = X_pert_train.shape
    lib_all = X_pert_train.sum(dim=1)

    unique_perts, inverse_indices = torch.unique(pert_ids_train, return_inverse=True)  # [K], [N]
    K = unique_perts.numel()

    if not use_cycle:
        Y_sum = torch.zeros(K, G, dtype=torch.float32)
        counts = torch.zeros(K, dtype=torch.long)
        lib_sum = torch.zeros(K, dtype=torch.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X_pert_train[start:end]
            invb = inverse_indices[start:end]
            libb = lib_all[start:end].to(torch.float32)

            Y_sum.index_add_(0, invb, Xb)
            lib_sum.index_add_(0, invb, libb)
            counts.index_add_(0, invb, torch.ones_like(invb, dtype=torch.long))

        counts = counts.clamp_min(1)
        Y_avg = Y_sum / counts.unsqueeze(1)

        if use_sf:
            mean_lib = lib_sum / counts.to(torch.float32)
            s_eff = (mean_lib / ref_depth).clamp_min(1e-12)
            log_s_eff = torch.log(s_eff)
        else:
            log_s_eff = None

        return Y_avg, unique_perts, None, log_s_eff

    else:
        # 维度 [K, 3, G] 分别聚合 G1/S/G2M
        Y_sum = torch.zeros(K, 3, G, dtype=torch.float32)
        counts = torch.zeros(K, 3, dtype=torch.long)
        lib_sum = torch.zeros(K, 3, dtype=torch.float32)

        for start in tqdm(range(0, N, batch_size), desc="构建 pseudo-bulk by phase (CPU)"):
            end = min(start + batch_size, N)
            Xb = X_pert_train[start:end]
            invb = inverse_indices[start:end]
            phb = phase_ids_train[start:end]
            libb = lib_all[start:end].to(torch.float32)

            for ph in (0, 1, 2):
                mask = (phb == ph)
                if mask.any():
                    idxp = invb[mask]
                    Xsub = Xb[mask]
                    lsub = libb[mask]
                    Y_sum_phase = Y_sum[:, ph, :]
                    Y_sum_phase.index_add_(0, idxp, Xsub)
                    lib_sum[:, ph].index_add_(0, idxp, lsub)
                    counts[:, ph].index_add_(0, idxp, torch.ones_like(idxp, dtype=torch.long))

        counts = counts.clamp_min(1)
        Y_avg = Y_sum / counts.unsqueeze(2)  # [K, 3, G]

        # 摊平成有效条目
        mask_flat = (counts > 0).reshape(-1)              # [K*3]
        Y_flat = Y_avg.reshape(K * 3, G)
        sel_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)

        ks = torch.arange(K, dtype=torch.long).unsqueeze(1).repeat(1, 3).reshape(-1)  # [K*3]
        phs = torch.tensor([0, 1, 2], dtype=torch.long).repeat(K)                     # [K*3]

        pert_ids_eff = ks.index_select(0, sel_idx)   # 相对 unique_perts 的索引
        phase_ids_eff = phs.index_select(0, sel_idx)
        Y_avg_flat = Y_flat.index_select(0, sel_idx)

        if use_sf:
            lib_flat = lib_sum.reshape(K * 3)
            cnt_flat = counts.reshape(K * 3).to(torch.float32)
            mean_lib_flat = (lib_flat / cnt_flat.clamp_min(1.0)).index_select(0, sel_idx)
            s_eff = (mean_lib_flat / ref_depth).clamp_min(1e-12)
            log_s_eff = torch.log(s_eff)
        else:
            log_s_eff = None

        return Y_avg_flat, unique_perts[pert_ids_eff], phase_ids_eff, log_s_eff
