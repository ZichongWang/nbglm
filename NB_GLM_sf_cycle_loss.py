###############################
'''
Comparing to v1, this version use flexible theta per gene
加上细胞周期，用了更多种类的loss
Modifications:
- CPU-friendly fit_concise (build pseudo-bulk on CPU; only KxG on GPU)
- --use_sf toggle for size factor (train + sampling)
- Cell cycle covariate (--use_cycle) with G1 baseline; S & G2M learned per-gene effects
- --phase_strategy {ignore, global, control, fixed_G1, fixed_S, fixed_G2M}
- Extended losses: POIS_DEV, NB_DEV, MSE_LOG1P, MSE_ANS
- NB sampler uses logits parameterization (stable & mean-correct)

基线（只用 size factor、不用周期，MSE）：
python NB_GLM_cycle.py --fit concise --use_sf --loss MSE

用周期 + Poisson deviance：
python NB_GLM_cycle.py --fit concise --use_sf --use_cycle --loss POIS_DEV --phase_strategy global

用周期 + log1p-MSE（预测固定为 S 期）：
python NB_GLM_cycle.py --fit concise --use_sf --use_cycle --loss MSE_LOG1P --phase_strategy fixed_S

不用 size factor、不用周期 + NB deviance：
python NB_GLM_cycle.py --fit concise --loss NB_DEV
'''
###############################

import os # 指定使用的GPU设备编号
import subprocess
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import time
import warnings
from torch.utils.data import Dataset, TensorDataset, DataLoader
# 忽略anndata可能发出的性能警告
warnings.filterwarnings('ignore', category=FutureWarning, module='anndata')

# ------------------ 辅助工具 ------------------
PHASE2ID = {"G1": 0, "S": 1, "G2M": 2}

def to_tensor(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return torch.tensor(X, dtype=torch.float32)

def compute_size_factors(X: torch.Tensor, ref_depth: torch.Tensor | None = None, eps: float = 1e-8):
    lib = X.sum(dim=1)
    if ref_depth is None:
        ref_depth = torch.median(lib)
    sf = (lib / (ref_depth + eps)).clamp(min=eps)
    return sf, ref_depth

def build_validation_size_factors(df_val: pd.DataFrame,
                                  sf_ctrl: torch.Tensor,
                                  ref_depth: torch.Tensor,
                                  seed: int = 2025) -> torch.Tensor:
    """按 df_val 顺序生成测试细胞的 size factor；每个扰动从 control sf 重采样并缩放到 median_umi/ref_depth 的中位数。"""
    import numpy as np
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

def estimate_theta_per_gene(X_ctrl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    最简单 MoM：theta = mu^2 / (var - mu)；若 var<=mu 视为近似 Poisson，置大值。
    注意：若 --use_sf，则 main() 会先把 X_ctrl 做了 size-factor 标准化，再丢到此函数。
    """
    print("开始为每个基因估计离散度参数 theta...")
    mu_g = X_ctrl.mean(dim=0)
    var_g = X_ctrl.var(dim=0, unbiased=False)
    denominator = var_g - mu_g
    theta_g = torch.full_like(mu_g, 1e6)  # 大值近似 Poisson
    overdispersed_mask = denominator > eps
    theta_g[overdispersed_mask] = (
        torch.square(mu_g[overdispersed_mask]) / denominator[overdispersed_mask]
    )
    theta_g = theta_g.clamp(min=1e-6, max=1e12)
    print("Theta 估计完成！")
    return theta_g

# --------- cell cycle 工具：训练/预测阶段的 phase 处理 ----------
def phases_to_ids(phases: list[str]) -> torch.LongTensor:
    return torch.tensor([PHASE2ID.get(p, 0) for p in phases], dtype=torch.long)

def compute_global_phase_probs(phases: list[str]) -> np.ndarray:
    """返回 [p(G1), p(S), p(G2M)]"""
    counts = np.zeros(3, dtype=float)
    for p in phases:
        if p in PHASE2ID:
            counts[PHASE2ID[p]] += 1
    total = counts.sum()
    if total <= 0:  # fallback: assume mostly G1
        return np.array([0.7, 0.15, 0.15], dtype=float)
    return counts / total

def compute_per_pert_phase_probs(adata_pert, pert_name_col: str) -> dict[str, np.ndarray]:
    """按扰动统计 phase 概率，返回 {pert_name: [pG1,pS,pG2M]}"""
    df = adata_pert.obs[[pert_name_col, "phase"]].copy()
    out = {}
    for name, sub in df.groupby(pert_name_col):
        out[name] = compute_global_phase_probs(sub["phase"].tolist())
    return out

def sample_validation_phases(df_val: pd.DataFrame,
                             phase_strategy: str,
                             global_probs: np.ndarray,
                             per_pert_probs: dict[str, np.ndarray] | None,
                             seed: int = 2025) -> list[int]:
    """
    根据策略为验证集每个待生成细胞采样/指定 phase，返回 phase_id 列表（与 df_val 展开顺序一致）。
    策略：
      - ignore: 全设为 G1（基线）
      - global: 按全局比例采样
      - control: 按训练集中对应扰动的比例采样，若未知则退化为 global
      - fixed_G1 / fixed_S / fixed_G2M
    """
    rng = np.random.default_rng(seed)
    out = []
    for _, row in df_val.iterrows():
        pert = str(row["target_gene"])
        n = int(row["n_cells"])
        if phase_strategy == "ignore" or phase_strategy == "fixed_G1":
            out.extend([PHASE2ID["G1"]]*n)
        elif phase_strategy == "fixed_S":
            out.extend([PHASE2ID["S"]]*n)
        elif phase_strategy == "fixed_G2M":
            out.extend([PHASE2ID["G2M"]]*n)
        elif phase_strategy == "global":
            out.extend(rng.choice([0,1,2], size=n, p=global_probs).tolist())
        elif phase_strategy == "control":
            probs = global_probs
            if per_pert_probs is not None and pert in per_pert_probs:
                probs = per_pert_probs[pert]
            out.extend(rng.choice([0,1,2], size=n, p=probs).tolist())
        else:
            # 默认退化为 global
            out.extend(rng.choice([0,1,2], size=n, p=global_probs).tolist())
    return out

# ===================================================================
# 新增：Dataset 与 Pseudo-bulk 构建（放在 fit 外部）
# ===================================================================

class WholeCellDataset(Dataset):
    """
    """
    def __init__(self,
                 X_tensor: torch.Tensor,            # [N, G] CPU
                 pert_ids: torch.LongTensor,        # [N]    CPU
                 sf: torch.Tensor | None = None,    # [N]    CPU
                 use_sf: bool = True,
                 use_cycle: bool = False,
                 phase_ids: torch.LongTensor | None = None):
        assert X_tensor.device.type == "cpu"
        assert pert_ids.device.type == "cpu"
        self.Y = X_tensor  # 原始计数，不在 Dataset 内做 sf 归一

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

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        item = {
            'y': self.Y[idx],
            'pert': self.pert_ids[idx]
        }
        item['log_s'] = self.log_s[idx]
        if self.use_cycle:
            item['phase'] = self.phase_ids[idx]
        return item


class PseudoBulkDataset(Dataset):
    """
    聚合后的 pseudo-bulk 数据集：
      - 不使用周期：每个样本是一个扰动的平均 [G]
      - 使用周期：每个样本是 (扰动×phase) 的平均 [G]
    __getitem__ 返回一个 dict：{'y': [G], 'pert': [], 'phase': [] (可选)}
    """
    def __init__(self,
                 Y_avg: torch.Tensor,                       # [K, G] 或 [B_eff, G] CPU
                 pert_ids_eff: torch.LongTensor,            # [K] 或 [B_eff]
                 use_cycle: bool = False,
                 phase_ids_eff: torch.LongTensor | None = None,
                 log_s_eff: torch.Tensor | None = None):  # [K] 或 [B_eff]，若不使用 size factor 则传 None
        assert Y_avg.device.type == "cpu"
        assert pert_ids_eff.device.type == "cpu"
        self.Y = Y_avg
        self.pert_ids_eff = pert_ids_eff
        self.use_cycle = use_cycle
        if use_cycle:
            assert phase_ids_eff is not None and phase_ids_eff.device.type == "cpu"
        self.phase_ids_eff = phase_ids_eff
        self.log_s = log_s_eff if log_s_eff is not None else torch.zeros(Y_avg.size(0), dtype=torch.float32)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        item = {
            'y': self.Y[idx],
            'pert': self.pert_ids_eff[idx]
        }
        if self.use_cycle:
            item['phase'] = self.phase_ids_eff[idx]
        item['log_s'] = self.log_s[idx]
        return item


@torch.no_grad()
def build_pseudobulk(X_pert_train: torch.Tensor,          # [N, G] CPU
                     pert_ids_train: torch.LongTensor,    # [N]    CPU
                     sf_pert: torch.Tensor | None,        # [N]    CPU，若不使用 size factor 则传 None。 仅为接口兼容，实际未使用
                     ref_depth,
                     use_sf: bool,
                     use_cycle: bool,
                     phase_ids_train: torch.LongTensor | None,  # [N] CPU，当 use_cycle=True 必须提供
                     batch_size: int = 4096):
    """
    在 CPU 上构建 pseudo-bulk（原先 fit_concise 内的逻辑移至此处）：
      - 若 use_cycle=False：返回 (Y_avg[K,G], unique_perts[K], None)
      - 若 use_cycle=True ：返回 (Y_avg_flat[B_eff,G], pert_ids_eff[B_eff], phase_ids_eff[B_eff])
    """
    lib_all = X_pert_train.sum(dim=1)  # [N] 每个细胞的 UMI（原始库大小）
    assert X_pert_train.device.type == "cpu"
    assert pert_ids_train.device.type == "cpu"
    if use_cycle:
        assert phase_ids_train is not None and phase_ids_train.device.type == "cpu"

    N, G = X_pert_train.shape
    unique_perts, inverse_indices = torch.unique(pert_ids_train, return_inverse=True)  # [K], [N]
    K = unique_perts.numel()

    if not use_cycle:
        Y_sum = torch.zeros(K, G, dtype=torch.float32)    # CPU
        counts = torch.zeros(K, dtype=torch.long)         # CPU
        lib_sum = torch.zeros(K, dtype=torch.float32)  # CPU
        for start in tqdm(range(0, N, batch_size), desc="构建 pseudo-bulk (CPU)"):
            end = min(start + batch_size, N)
            Xb = X_pert_train[start:end]                  # [B,G]
            invb = inverse_indices[start:end]             # [B]
            libb = lib_all[start:end].to(torch.float32)

            Y_sum.index_add_(0, invb, Xb)
            lib_sum.index_add_(0, invb, libb)
            counts.index_add_(0, invb, torch.ones_like(invb, dtype=torch.long))
        counts = counts.clamp_min(1)
        Y_avg = Y_sum / counts.unsqueeze(1)               # [K,G]
        if use_sf:
            mean_lib = lib_sum / counts.clamp_min(1).to(torch.float32)     # [K]
            s_eff = (mean_lib / ref_depth).clamp_min(1e-12)                # [K]
            log_s_eff = torch.log(s_eff)
        else:
            log_s_eff = None

        return Y_avg, unique_perts, None, log_s_eff
    else:
        Y_sum = torch.zeros(K, 3, G, dtype=torch.float32) # CPU
        lib_sum = torch.zeros(K, 3, dtype=torch.float32)  # CPU
        counts = torch.zeros(K, 3, dtype=torch.long)      # CPU
        for start in tqdm(range(0, N, batch_size), desc="构建 pseudo-bulk by phase (CPU)"):
            end = min(start + batch_size, N)
            Xb  = X_pert_train[start:end]                 # [B,G]
            invb = inverse_indices[start:end]             # [B]
            phb  = phase_ids_train[start:end]             # [B]
            libb = lib_all[start:end].to(torch.float32)

            for phase_id in (0, 1, 2):
                mask = (phb == phase_id)
                if mask.any():
                    idxp = invb[mask]
                    Xsub = Xb[mask]
                    lsub = libb[mask]
                    Y_sum_phase = Y_sum[:, phase_id, :]   # [K,G]
                    Y_sum_phase.index_add_(0, idxp, Xsub)
                    lib_sum[:, phase_id].index_add_(0, idxp, lsub)
                    counts_phase = counts[:, phase_id]    # [K]
                    counts_phase.index_add_(0, idxp, torch.ones_like(idxp, dtype=torch.long))
        counts = counts.clamp_min(1)
        Y_avg = Y_sum / counts.unsqueeze(2)               # [K,3,G]

        # 保留非空并摊平
        mask_flat = (counts > 0).reshape(-1)              # [K*3]
        Y_flat    = Y_avg.reshape(K * 3, G)               # [K*3,G]
        sel_idx   = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        lib_flat = lib_sum.reshape(K*3)                     # [K*3]
        Y_avg_flat = Y_flat.index_select(0, sel_idx)      # [B_eff,G]

        ks  = torch.arange(K, dtype=torch.long).unsqueeze(1).repeat(1, 3).reshape(-1)
        phs = torch.tensor([0,1,2], dtype=torch.long).repeat(K)
        pert_ids_eff  = ks.index_select(0, sel_idx)       # [B_eff]（相对 unique_perts 的索引）
        phase_ids_eff = phs.index_select(0, sel_idx)      # [B_eff]
        if use_sf:
            mean_lib_flat = (lib_flat / counts.reshape(-1).clamp_min(1).to(torch.float32)).index_select(0, sel_idx)  # [B_eff]
            s_eff_flat = (mean_lib_flat / ref_depth).clamp_min(1e-12)
            log_s_eff = torch.log(s_eff_flat)                        # [B_eff]
        else:
            log_s_eff = None
        return Y_avg_flat, unique_perts[pert_ids_eff], phase_ids_eff, log_s_eff

# ===================================================================
# 1. 模型定义: LowRankNB_GLM（加入 cell cycle 固定效应）
# ===================================================================
class LowRankNB_GLM(nn.Module):
    def __init__(self, 
                 gene_emb: torch.Tensor, 
                 pert_emb: torch.Tensor,
                 mu_control: torch.Tensor,
                 theta_per_gene: torch.Tensor,
                 use_cycle: bool = False):
        """
        初始化模型。
        - 若 use_cycle=True：在 log-mean 中加入周期固定效应（G1 基线；学习 S & G2M 的 per-gene 系数）
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模型将在 {self.device} 上运行。")

        # --- 固定的先验和参数 ---
        self.G = gene_emb.to(self.device)
        self.P = pert_emb.to(self.device)
        self.mu_control = (mu_control + 1e-8).to(self.device)
        self.theta = theta_per_gene.to(self.device)
        self.use_cycle = use_cycle

        # --- 需要学习的模型参数（与原始结构一致） ---
        gene_emb_dim = self.G.shape[1]
        pert_emb_dim = self.P.shape[1]
        n_genes = self.G.shape[0]
        self.K = nn.Parameter(torch.empty(gene_emb_dim, pert_emb_dim, device=self.device))
        self.bias = nn.Parameter(torch.empty(n_genes, device=self.device))
        self.delta_log_mu_scaler = nn.Parameter(torch.tensor(5.0, device=self.device))

        # 周期固定效应：两个通道（S, G2M），G1 为基线
        if self.use_cycle:
            self.beta_cycle = nn.Parameter(torch.zeros(n_genes, 2, device=self.device))
        else:
            self.beta_cycle = None

        nn.init.xavier_uniform_(self.K)
        nn.init.zeros_(self.bias)
        
    def forward(self, pert_ids: torch.LongTensor, phase_ids: torch.LongTensor | None = None, offset_log_s: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向：输出参考深度下的均值 mu_ref。
        log mu_ref = log mu_control + Delta_pert + [cycle effects]
        - phase_ids: LongTensor in {0:G1, 1:S, 2:G2M}; 若 None 或 use_cycle=False，则不加周期项（等价 G1）。
        """
        P_selected = self.P[pert_ids]                                   # [B, d_p]
        raw_output = (self.G @ self.K @ P_selected.T).T + self.bias.unsqueeze(0)  # [B, G]
        activated_output = torch.tanh(raw_output)
        delta_log_mu = self.delta_log_mu_scaler * activated_output       # [B, G]

        # 周期固定效应（G1 基线；S、G2M 两个系数）
        if self.use_cycle and phase_ids is not None:
            # one-hot (S,G2M)，G1 -> (0,0), S -> (1,0), G2M -> (0,1)
            is_S   = (phase_ids == 1).float().unsqueeze(1)               # [B,1]
            is_G2M = (phase_ids == 2).float().unsqueeze(1)               # [B,1]
            # [B,G] = is_S * beta[:,0]^T + is_G2M * beta[:,1]^T
            cycle_term = is_S @ self.beta_cycle[:,0].unsqueeze(0) + is_G2M @ self.beta_cycle[:,1].unsqueeze(0)
            log_mu = torch.log(self.mu_control.unsqueeze(0)) + delta_log_mu + cycle_term
        else:
            log_mu = torch.log(self.mu_control.unsqueeze(0)) + delta_log_mu
        if offset_log_s is not None:
            log_mu = log_mu + offset_log_s.unsqueeze(1) # [B,G]

        mu_pred = torch.exp(log_mu)
        return torch.clamp(mu_pred, min=1e-10, max=1e7)

    # --------- 多种损失（deviance / 变换刻度 MSE） ----------
    @staticmethod
    def loss_pois_dev(mu_pred, y_true, eps=1e-8):
        # 2 * [y*log(y/mu) - (y-mu)], with y*log(y/mu)=0 when y=0
        y = y_true
        mu = mu_pred.clamp_min(eps)
        term = torch.zeros_like(mu)
        mask = y > 0
        term[mask] = y[mask] * (torch.log(y[mask] + eps) - torch.log(mu[mask]))
        dev = 2.0 * (term - (y - mu))
        return dev.mean()

    def loss_nb_dev(self, mu_pred, y_true, eps=1e-8):
        # 2 * [ y*log(y/mu) - (y+theta)*log((y+theta)/(mu+theta)) ]
        y = y_true
        mu = mu_pred.clamp_min(eps)
        theta = self.theta.unsqueeze(0).expand_as(mu).clamp_min(eps)
        term1 = torch.zeros_like(mu)
        mask = y > 0
        term1[mask] = y[mask] * (torch.log(y[mask] + eps) - torch.log(mu[mask]))
        term2 = (y + theta) * (torch.log(y + theta + eps) - torch.log(mu + theta))
        dev = 2.0 * (term1 - term2)
        return dev.mean()

    @staticmethod
    def loss_mse(mu_pred, y_true):
        return nn.MSELoss()(mu_pred, y_true)

    @staticmethod
    def loss_mse_log1p(mu_pred, y_true):
        return nn.MSELoss()(torch.log1p(mu_pred), torch.log1p(y_true))

    @staticmethod
    def loss_mse_anscombe(mu_pred, y_true):
        # Anscombe transform: z = 2*sqrt(y + 3/8)
        z_true = 2.0 * torch.sqrt(y_true + 0.375)
        z_pred = 2.0 * torch.sqrt(mu_pred + 0.375)
        return nn.MSELoss()(z_pred, z_true)


    # ========= 统一的训练接口：fit(dataloader, ...) =========
    def fit(self,
            dataloader: DataLoader,
            loss_type: str = "MSE",
            learning_rate: float = 5e-4,
            n_epochs: int = 100,
            l1_lambda: float = 0.001,
            l2_lambda: float = 0.01):
        """
        统一训练循环：dataloader 每个 batch 提供
          - batch['pert']: LongTensor [B]
          - batch['y']   : FloatTensor [B,G]（目标均值，已在外部完成 sf 或 pseudo-bulk）
          - （可选）batch['phase']: LongTensor [B]，当 use_cycle=True
        """
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

        def _loss(mu_pred, y_true):
            lt = loss_type.upper()
            if lt == "MSE":        return self.loss_mse(mu_pred, y_true)
            if lt == "NB":         return self.negative_binomial_nll_loss(mu_pred, y_true)
            if lt == "POIS_DEV":   return self.loss_pois_dev(mu_pred, y_true)
            if lt == "NB_DEV":     return self.loss_nb_dev(mu_pred, y_true)
            if lt == "MSE_LOG1P":  return self.loss_mse_log1p(mu_pred, y_true)
            if lt == "MSE_ANS":    return self.loss_mse_anscombe(mu_pred, y_true)
            raise ValueError(f"Unknown loss_type: {loss_type}")

        pbar = tqdm(range(n_epochs), desc="训练中 (GPU, unified)")
        for epoch in pbar:
            epoch_loss = 0.0
            for batch in dataloader:
                pert = batch['pert'].to(self.device)
                y_true = batch['y'].to(self.device)
                log_s = batch['log_s'].to(self.device)
                phase = batch.get('phase', None)
                if phase is not None:
                    phase = phase.to(self.device)

                optimizer.zero_grad()
                mu_pred = self.forward(pert, phase, offset_log_s=log_s)  # [B,G]
                loss = _loss(mu_pred, y_true) + l1_lambda * self.K.abs().sum() + l2_lambda * (self.K ** 2).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * y_true.size(0)

            avg_loss = epoch_loss / max(1, len(dataloader.dataset))
            pbar.set_postfix(loss=f"{avg_loss:.4f}")


    def negative_binomial_nll_loss(self, mu, y_true):
        """NB 对数似然（允许 y_true 为实数，使用 Γ 的连续延拓）。"""
        eps = 1e-8
        theta_broadcast = self.theta.unsqueeze(0)
        log_theta_mu_eps = torch.log(theta_broadcast + mu + eps)
        ll = (
            torch.lgamma(theta_broadcast + y_true + eps)
            - torch.lgamma(theta_broadcast + eps)
            - torch.lgamma(y_true + 1.0)
            + theta_broadcast * torch.log(theta_broadcast + eps)
            - theta_broadcast * log_theta_mu_eps
            + y_true * torch.log(mu + eps)
            - y_true * log_theta_mu_eps
        )
        return -ll.mean()
    
    def mse_loss(self, mu, y_true):
        return nn.MSELoss()(mu, y_true)

    @torch.no_grad()
    def predict_and_sample(self, 
        pert_ids_test: torch.LongTensor,
        batch_size: int = 1024,
        use_sf: bool = True,
        sf_test: torch.Tensor | None = None,        # [N_test] 若 use_sf=False 可传 None
        use_gamma: bool = True,
        gamma_r0: float = 50.0,
        sampler: str = "poisson",
        use_cycle: bool = False,
        phase_ids_test: torch.LongTensor | None = None  # [N_test] in {0,1,2}，若 use_cycle=True 必须提供
    ) -> torch.Tensor:
        """
        预测并采样：
          mu_ref = forward(pert_ids, [phase_ids])
          mu_obs = (use_sf ? sf_test : 1) * mu_ref
          若 use_gamma: mu_obs = (Gamma(r0,r0) per cell) * mu_obs
          采样：poisson 或 nb（logits参数化；mean=mu_obs）
        """
        print(f"开始分批预测并采样，批大小为 {batch_size}...")
        self.eval() 
        all_sampled_counts = []
        pbar = tqdm(range(0, len(pert_ids_test), batch_size), desc="预测中")
        with torch.no_grad():
            for i in pbar:
                sl = slice(i, i + batch_size)
                batch_pert_ids = pert_ids_test[sl].to(self.device)
                if use_cycle:
                    assert phase_ids_test is not None
                    batch_phase_ids = phase_ids_test[sl].to(self.device)
                else:
                    batch_phase_ids = None
                if use_sf:
                    sfb = sf_test[sl].to(self.device, non_blocking=True)
                    log_sfb = torch.log(sfb.clamp_min(1e-12))
                else:
                    log_sfb = None
                mu_obs = self.forward(batch_pert_ids, batch_phase_ids, offset_log_s=log_sfb)  # [B,G]

                if use_gamma:
                    r0 = torch.tensor(gamma_r0, device=self.device)
                    L = torch.distributions.Gamma(concentration=r0, rate=r0).sample((mu_obs.size(0),)).unsqueeze(1)
                    mu_obs = (mu_obs * L).clamp(min=1e-12)

                if sampler == "poisson":
                    sampled_counts_batch = torch.distributions.Poisson(mu_obs).sample()
                elif sampler == "nb":
                    theta_b = self.theta.unsqueeze(0)
                    logits = torch.log(theta_b) - torch.log(mu_obs)         # logit(p)=log(theta)-log(mu)
                    logits = logits.clamp(min=-40.0, max=40.0)
                    nb_dist = torch.distributions.NegativeBinomial(total_count=theta_b, logits=logits)
                    sampled_counts_batch = nb_dist.sample()
                else:
                    raise ValueError("sampler must be 'poisson' or 'nb'")

                all_sampled_counts.append(sampled_counts_batch.cpu())
        print("所有批次预测完成，正在拼接结果...")
        final_sampled_counts = torch.cat(all_sampled_counts, dim=0)
        print("采样完成！")
        return final_sampled_counts

# ===================================================================
# 2. 主执行逻辑
# ===================================================================
def main():
    start = time.time()
    ap = argparse.ArgumentParser(description="Estimate NB_GLM (concise, size factor & cell cycle & heterogeneity)")
    ap.add_argument("--task", choices=['test','real'],default='test', help="If test, run local evaluation after writing h5ad; if real, only write h5ad.")
    ap.add_argument("--cheat", action="store_true", help="Use train data to test.")
    # Loss 扩展：MSE（默认）、NB、POIS_DEV、NB_DEV、MSE_LOG1P、MSE_ANS
    ap.add_argument("--loss", choices=["MSE","NB","POIS_DEV","NB_DEV","MSE_LOG1P","MSE_ANS"], default="MSE",
                    help="Loss for concise training.")
    ap.add_argument("--fit", choices=['concise','whole'], default='concise', help="Concise will use pseudo-bulk; whole uses all cells directly.")
    ap.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    ap.add_argument("--sample_n", type=int, default=1, help="How many h5ad files to generate and evaluate.")
    ap.add_argument("--seed", type=int, default=2025, help="Base random seed.")

    # 是否使用 size factor（训练与采样）
    ap.add_argument("--use_sf", action="store_true", help="Use size factor in training (concise) and sampling.")
    # 细胞间异质性
    ap.add_argument("--use_gamma", action="store_true", help="Use shared Gamma(r0,r0) cell-level heterogeneity in sampling.")
    ap.add_argument("--gamma_r0", type=float, default=50.0, help="Gamma shape=rate r0; larger => less heterogeneity.")
    # 采样分布
    ap.add_argument("--sampler", choices=["poisson","nb"], default="poisson", help="Sampling distribution at prediction.")

    # 新增：是否使用 cell cycle 协变量（训练与采样）
    ap.add_argument("--use_cycle", action="store_true", help="Use cell cycle (phase) fixed effects (G1 baseline; S/G2M learned).")
    # 预测阶段如何指定 phase
    ap.add_argument("--phase_strategy", choices=["ignore","global","control","fixed_G1","fixed_S","fixed_G2M"], default="global",
                    help="How to assign phases to generated cells when --use_cycle.")
    ap.add_argument("--lambda_l1", type=float, default=0.0001, help="L1 lambda for regularization.")
    ap.add_argument("--lambda_l2", type=float, default=0.005, help="L2 lambda for regularization.")

    args = ap.parse_args()

    # Arc
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) 
    if args.task == 'real':
        TRAIN_DATA_PATH = "../vcc_data/adata_pp.h5ad"
        GENE_EMBEDDING_PATH = '../vcc_data/PCA_gene_embedding_512D.csv'
        PERT_EMBEDDING_PATH = "../vcc_data/perturbation_embedding_P_512D_cpu.csv"
        VALIDATION_LIST_PATH = "../vcc_data/pert_counts_Validation.csv"
        OUTPUT_PATH = "NBGLM_0924_size_factor_seed114514.h5ad"
        PERT_NAME = "target_gene"
    else:
        if args.cheat:
            TRAIN_DATA_PATH = "../vcc_data/adata_pp.h5ad"
        else:
            TRAIN_DATA_PATH = "../vcc_data/Official_Data_Split/train.h5ad"
        GENE_EMBEDDING_PATH = '../vcc_data/PCA_gene_embedding_512D.csv'
        PERT_EMBEDDING_PATH = "../vcc_data/perturbation_embedding_P_512D_cpu.csv"
        VALIDATION_LIST_PATH = "../vcc_data/Official_Data_Split/test_pert_info.csv"
        OUTPUT_PATH = "nbglm_concise_cycle_out_test.h5ad"
        PERT_NAME = "target_gene"

    CONTROL_PERT_NAME = "non-targeting"
    LEARNING_RATE = 5e-4
    N_EPOCHS = 100
    PREDICTION_BATCH_SIZE = 4096
    LOSS = args.loss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1. 加载所有数据和元信息 (使用原始计数) ---
    print(f"--- 1. 加载所有数据和元信息 ---")
    start_time = time.time()
    adata = sc.read_h5ad(TRAIN_DATA_PATH)
    print("数据加载完成，使用原始计数。")
    gene_names = adata.var.index.tolist()
    
    train_perts = set(adata.obs[PERT_NAME].unique())
    df_val = pd.read_csv(VALIDATION_LIST_PATH)
    val_perts = set(df_val['target_gene'].tolist())
    all_perts_set = train_perts.union(val_perts)
    all_perts_list = sorted(list(all_perts_set))
    if CONTROL_PERT_NAME in all_perts_list:
        all_perts_list.remove(CONTROL_PERT_NAME)
        all_perts_list.insert(0, CONTROL_PERT_NAME)
    pert_to_id = {name: i for i, name in enumerate(all_perts_list)}
    
    # --- 2. 加载和准备全局嵌入矩阵 G 和 P ---
    print(f"\n--- 2. 加载和准备全局嵌入矩阵 G 和 P ---")
    gene_embeddings = pd.read_csv(GENE_EMBEDDING_PATH, index_col=0)
    gene_dict = {gene: torch.tensor(row.values, dtype=torch.float32) for gene, row in gene_embeddings.iterrows()}
    pert_embeddings = pd.read_csv(PERT_EMBEDDING_PATH, index_col=0)
    pert_dict = {gene: torch.tensor(row.values, dtype=torch.float32) for gene, row in pert_embeddings.iterrows()}
    gene_emb_dim = next(iter(gene_dict.values())).shape[0]
    pert_emb_dim = next(iter(pert_dict.values())).shape[0]
    G_matrix = torch.zeros(len(gene_names), gene_emb_dim)
    P_matrix = torch.zeros(len(all_perts_list), pert_emb_dim)
    for i, name in enumerate(gene_names):
        if name in gene_dict: G_matrix[i] = gene_dict[name]
    for i, name in enumerate(all_perts_list):
        if name in pert_dict: P_matrix[i] = pert_dict[name]
        
    G_matrix = torch.nn.functional.normalize(G_matrix, p=2, dim=1)
    P_matrix = torch.nn.functional.normalize(P_matrix, p=2, dim=1)

    # --- 3. 准备训练数据张量 (CPU；避免OOM) ---
    print(f"\n--- 3. 准备训练数据张量 ---")
    adata_ctrl = adata[adata.obs[PERT_NAME] == CONTROL_PERT_NAME].copy()
    adata_pert = adata[adata.obs[PERT_NAME] != CONTROL_PERT_NAME].copy()

    X_pert_train = to_tensor(adata_pert.X)   # CPU
    X_ctrl_all = to_tensor(adata_ctrl.X)     # CPU

    # size factor（以 control 的中位深度为参考）；若不使用 sf，也仍计算以支持采样阶段可选使用
    sf_ctrl, ref_depth = compute_size_factors(X_ctrl_all)
    sf_pert, _ = compute_size_factors(X_pert_train, ref_depth)

    # mu_control & theta：若使用 sf，则先标准化到参考深度再估计；否则直接估计
    if args.use_sf:
        X_ctrl_norm = X_ctrl_all / sf_ctrl.unsqueeze(1)
        mu_control = X_ctrl_norm.mean(dim=0)
        theta_vector = estimate_theta_per_gene(X_ctrl_norm)
    else:
        mu_control = X_ctrl_all.mean(dim=0)
        theta_vector = estimate_theta_per_gene(X_ctrl_all)

    # 训练用的扰动 id
    pert_names_train = adata_pert.obs[PERT_NAME].tolist()
    pert_ids_train = torch.tensor([pert_to_id[p] for p in pert_names_train], dtype=torch.long)

    # 若启用 cell cycle：把 adata_pert.obs['phase'] 映射为 {0:G1,1:S,2:G2M}
    if args.use_cycle:
        if "phase" not in adata_pert.obs.columns:
            raise RuntimeError("use_cycle=True 但 adata.obs['phase'] 不存在。")
        phase_ids_train = phases_to_ids(adata_pert.obs["phase"].tolist())
    else:
        phase_ids_train = None

    
    # --- 4. 实例化并训练（使用统一的 dataloader 接口） ---
    print(f"\n--- 4. 实例化并训练 NB-GLM 模型 ---")
    model = LowRankNB_GLM(
        gene_emb=G_matrix,
        pert_emb=P_matrix,
        mu_control=mu_control,
        theta_per_gene=theta_vector,
        use_cycle=args.use_cycle
    )

    # 根据 args.fit 选择数据构建方式
    if args.fit == 'whole':
        # per-cell 数据集（已在外部做 size factor 归一）
        whole_ds = WholeCellDataset(
            X_tensor=X_pert_train,
            pert_ids=pert_ids_train,
            sf=(sf_pert if args.use_sf else None),
            use_sf=args.use_sf,
            use_cycle=args.use_cycle,
            phase_ids=phase_ids_train
        )
        train_loader = DataLoader(whole_ds, batch_size=max(1, PREDICTION_BATCH_SIZE // 4), shuffle=True, drop_last=False)
    elif args.fit == 'concise':
        # concise：先在外部构建 pseudo-bulk，再用 DataLoader
        Y_avg, unique_perts_eff, phase_ids_eff, log_s_eff = build_pseudobulk(
            X_pert_train=X_pert_train,
            pert_ids_train=pert_ids_train,
            sf_pert=(sf_pert if args.use_sf else None),
            ref_depth=ref_depth,
            use_sf=args.use_sf,
            use_cycle=args.use_cycle,
            phase_ids_train=phase_ids_train,
            batch_size=PREDICTION_BATCH_SIZE
        )
        if args.use_cycle:
            pb_ds = PseudoBulkDataset(
                Y_avg=Y_avg,                         # [B_eff,G]
                pert_ids_eff=unique_perts_eff,       # [B_eff]
                use_cycle=True,
                phase_ids_eff=phase_ids_eff,          # [B_eff]
                log_s_eff=log_s_eff
            )
        else:
            pb_ds = PseudoBulkDataset(
                Y_avg=Y_avg,                         # [K,G]
                pert_ids_eff=unique_perts_eff,       # [K]
                use_cycle=False,
                phase_ids_eff=None,
                log_s_eff=log_s_eff
            )
        train_loader = DataLoader(pb_ds, batch_size=max(1, PREDICTION_BATCH_SIZE // 4), shuffle=True, drop_last=False)
    else:
        raise ValueError("fit must be 'whole' or 'concise'")

    # 统一训练入口
    model.fit(
        dataloader=train_loader,
        loss_type=LOSS,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        l1_lambda=args.lambda_l1,
        l2_lambda=args.lambda_l2
    )
    
    # --- 5. 准备预测任务 ---
    print(f"\n--- 5. 准备预测任务 ---")
    val_pert_names = df_val['target_gene'].tolist()
    n_cells_per_val_pert = df_val['n_cells'].tolist()
    pert_ids_test_list = []
    for name, n_cells in zip(val_pert_names, n_cells_per_val_pert):
        pert_ids_test_list.extend([pert_to_id[name]] * int(n_cells))
    pert_ids_test = torch.tensor(pert_ids_test_list, dtype=torch.long)

    # 若使用 sf：根据 csv median_umi_per_cell + control sf 分布构造 sf_test；否则为 None
    if args.use_sf:
        sf_test = build_validation_size_factors(df_val, sf_ctrl, ref_depth, seed=args.seed)
        assert sf_test.numel() == len(pert_ids_test_list)
    else:
        sf_test = None

    # 若使用 cell cycle：根据策略生成 phase_ids_test
    if args.use_cycle:
        # 计算全局/每扰动的 phase 概率（基于训练集）
        global_probs = compute_global_phase_probs(adata.obs["phase"].tolist() if "phase" in adata.obs.columns else [])
        per_pert_probs = compute_per_pert_phase_probs(adata_pert, PERT_NAME) if args.phase_strategy == "control" else None
        phase_ids_list = sample_validation_phases(df_val, args.phase_strategy, global_probs, per_pert_probs, seed=args.seed)
        phase_ids_test = torch.tensor(phase_ids_list, dtype=torch.long)
        assert phase_ids_test.numel() == len(pert_ids_test_list)
    else:
        phase_ids_test = None
    
    # --- 6. 执行预测（包含可选的细胞间异质性 & cell cycle） ---
    print(f"\n--- 6. 执行预测 ---")
    sampled_counts = model.predict_and_sample(
        pert_ids_test=pert_ids_test, 
        batch_size=PREDICTION_BATCH_SIZE,
        use_sf=args.use_sf,
        sf_test=sf_test,
        use_gamma=args.use_gamma,
        gamma_r0=args.gamma_r0,
        sampler=args.sampler,
        use_cycle=args.use_cycle,
        phase_ids_test=phase_ids_test
    )
    final_predictions_tensor = sampled_counts.cpu()
    
    # --- 7. 组装并保存最终的 anndata 对象 ---
    print(f"\n--- 7. 组装并保存最终的 h5ad 文件 ---")
    adata_output_ctrl = adata[adata.obs[PERT_NAME] == CONTROL_PERT_NAME].copy()
    obs_pred = pd.DataFrame({PERT_NAME: [all_perts_list[i] for i in pert_ids_test_list]})
    if args.use_cycle:
        # 方便下游分析：把生成的 phase 也写到 obs（若选择 ignore/fixed_... 会是单一值）
        phase_str = {0:"G1", 1:"S", 2:"G2M"}
        obs_pred["phase"] = [phase_str.get(int(x), "G1") for x in (phase_ids_test.tolist() if phase_ids_test is not None else [0]*len(obs_pred))]
    adata_pred = ad.AnnData(X=final_predictions_tensor.numpy(), obs=obs_pred, var=adata.var.copy())
    
    final_adata = ad.concat([adata_output_ctrl, adata_pred], join='outer', index_unique=None)
    final_adata.X = final_adata.X.astype(np.float32)

    final_adata.write(OUTPUT_PATH)
    print(f"成功将最终结果保存至: {OUTPUT_PATH}")
    print(f"--- 总耗时: {time.time() - start_time:.2f} 秒 ---")
    if args.task == 'test':
        cmd = ["python", "eval.py", "--prediction", OUTPUT_PATH]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="") 
        process.wait()

if __name__ == '__main__':
    main()
