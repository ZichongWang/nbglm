# src/nbglm/model.py
# -*- coding: utf-8 -*-
"""
模型定义（Model Definition）与训练/采样接口
========================================

本模块提供低秩负二项广义线性模型（Low-Rank Negative Binomial GLM, *LowRankNB_GLM*），
以及训练循环、损失函数（deviance/MSE 等）与采样（Poisson 或 NB，logits 参数化）接口。

模型形式（Model Form）
---------------------
令：
- 基因嵌入矩阵 $G \\in \\mathbb{R}^{\\,n_g \\times d_g}$；
- 扰动嵌入矩阵 $P \\in \\mathbb{R}^{\\,n_p \\times d_p}$；
- 低秩核 $K \\in \\mathbb{R}^{\\,d_g \\times d_p}$（可学习参数）；
- 控制组均值（参考尺度）$\\mu^{\\mathrm{ctrl}} \\in \\mathbb{R}^{\\,n_g}$；
- 周期固定效应（S/G2M 两路）$\\beta \\in \\mathbb{R}^{\\,n_g\\times 2}$（可选）；
- size factor 偏置（offset）$\\log s_b$（样本维度）。

对于第 b 个样本、基因 g，定义参考尺度（reference-scale）的对数均值：
$$
\\log \\mu^{(\\mathrm{ref})}_{bg}
= \\log \\mu^{\\mathrm{ctrl}}_g
+ \\alpha_g\\, \\tanh\\!\\big(\\,(G K P_{b})_g\\,\\big)
+ \\beta_{g,S}\\,\\mathbf{1}[\\phi_b=\\mathrm{S}]
+ \\beta_{g,G2M}\\,\\mathbf{1}[\\phi_b=\\mathrm{G2M}]
+ \\log s_b,
$$
其中 $\\alpha_g$ 由 `delta_log_mu_scaler` 缩放控制，$\\phi_b \\in \\{\\mathrm{G1},\\mathrm{S},\\mathrm{G2M}\\}$。
观测尺度（observation scale）期望 $\\mu^{(\\mathrm{obs})} = \\exp(\\log\\mu^{(\\mathrm{ref})})$。

负二项分布采样（NB Sampling）
-----------------------------
采用 logits 参数化确保数值稳定性：
- total_count = $\\theta_g$（基因特异的离散度参数）
- logits = $\\log\\theta_g - \\log\\mu_{bg}^{(\\mathrm{obs})}$

在此参数化下，NB 的均值 $\\mathbb{E}[Y]=\\mu$ 精确成立（除数值截断外）。

正则化（Regularization）
------------------------
对核矩阵 K 应用 L1/L2 正则：
$$
\\mathcal{L}_{\\mathrm{reg}} = \\lambda_1 \\|K\\|_1 + \\lambda_2 \\|K\\|_F^2 .
$$

依赖（Dependencies）
-------------------
- torch（核心）
- torch.distributions（采样）
"""

from __future__ import annotations

from typing import Callable, Dict, Optional
import logging
import math
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# 损失函数集合（Registry）
# -----------------------------
def _loss_mse(mu_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(mu_pred, y_true)


def _loss_mse_log1p(mu_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(torch.log1p(mu_pred), torch.log1p(y_true))


def _loss_mse_anscombe(mu_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Anscombe 变换：
    z = 2 * sqrt(y + 3/8)，用于稳定方差（variance-stabilizing transform）。
    """
    z_true = 2.0 * torch.sqrt(y_true + 0.375)
    z_pred = 2.0 * torch.sqrt(mu_pred + 0.375)
    return nn.MSELoss()(z_pred, z_true)


def _loss_pois_dev(mu_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Poisson deviance（泊松偏差）：
    2 * [ y*log(y/mu) - (y - mu) ]，定义 0*log(0) = 0。
    """
    y = y_true
    mu = mu_pred.clamp_min(eps)
    term = torch.zeros_like(mu)
    mask = y > 0
    term[mask] = y[mask] * (torch.log(y[mask] + eps) - torch.log(mu[mask]))
    dev = 2.0 * (term - (y - mu))
    return dev.mean()


# 负二项 deviance 需要 theta
def _loss_nb_dev(mu_pred: torch.Tensor, y_true: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    NB deviance（负二项偏差）：
    2 * [ y*log(y/mu) - (y + theta)*log((y+theta)/(mu+theta)) ].
    """
    y = y_true
    mu = mu_pred.clamp_min(eps)
    th = theta.unsqueeze(0).expand_as(mu).clamp_min(eps)
    term1 = torch.zeros_like(mu)
    mask = y > 0
    term1[mask] = y[mask] * (torch.log(y[mask] + eps) - torch.log(mu[mask]))
    term2 = (y + th) * (torch.log(y + th + eps) - torch.log(mu + th))
    dev = 2.0 * (term1 - term2)
    return dev.mean()


LOSSES_NO_THETA: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "MSE": _loss_mse,
    "MSE_LOG1P": _loss_mse_log1p,
    "MSE_ANS": _loss_mse_anscombe,
    "POIS_DEV": _loss_pois_dev,  # 需要额外 eps，已内置缺省
}

# 需要 theta 的损失单独处理（例如 NB_DEV、NB_NLL）
LOSSES_WITH_THETA: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "NB_DEV": _loss_nb_dev,
}


# -----------------------------
# 模型定义
# -----------------------------
class LowRankNB_GLM(nn.Module):
    """
    低秩负二项 GLM（Low-Rank NB GLM）。

    Parameters
    ----------
    gene_emb : torch.Tensor
        基因嵌入矩阵 G，[n_g, d_g]。
    pert_emb : torch.Tensor
        扰动嵌入矩阵 P，[n_p, d_p]。
    mu_control : torch.Tensor
        控制组参考均值（reference mean）向量，[n_g]。
    theta_per_gene : torch.Tensor
        基因级离散度参数 θ_g，[n_g]。
    use_cycle : bool
        是否在 log-mean 中加入细胞周期（S/G2M）固定效应，G1 为基线。

    Attributes
    ----------
    K : nn.Parameter
        低秩核矩阵，[d_g, d_p]。
    bias : nn.Parameter
        基因级偏置项，[n_g]。
    delta_log_mu_scaler : nn.Parameter
        对 tanh 激活后的项统一缩放，标量。
    beta_cycle : Optional[nn.Parameter]
        周期项（S/G2M 两列）[n_g, 2]，仅当 use_cycle=True 时存在。
    device : torch.device
        模型所在设备（cuda/cpu），构造时自动选择。
    """
    def __init__(
        self,
        gene_emb: torch.Tensor,
        pert_emb: torch.Tensor,
        mu_control: torch.Tensor,
        theta_per_gene: torch.Tensor,
        use_cycle: bool = False
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 缓存常量到 device
        self.G = gene_emb.to(self.device)                # [n_g, d_g]
        self.P = pert_emb.to(self.device)                # [n_p, d_p]
        self.mu_control = (mu_control + 1e-8).to(self.device)  # 避免 log(0)
        self.theta = theta_per_gene.to(self.device)      # [n_g]
        self.use_cycle = use_cycle

        n_genes, d_g = self.G.shape
        _, d_p = self.P.shape

        # 可学习参数
        self.K = nn.Parameter(torch.empty(d_g, d_p, device=self.device))
        self.bias = nn.Parameter(torch.empty(n_genes, device=self.device))
        self.delta_log_mu_scaler = nn.Parameter(torch.tensor(5.0, device=self.device))
        if self.use_cycle:
            self.beta_cycle = nn.Parameter(torch.zeros(n_genes, 2, device=self.device))  # (S, G2M)
        else:
            self.beta_cycle = None

        # 初始化
        nn.init.xavier_uniform_(self.K)
        nn.init.zeros_(self.bias)

    # -------------------------
    # 前向：输出观测尺度的均值 μ（已加 offset）
    # -------------------------
    def forward(
        self,
        pert_ids: torch.LongTensor,                        # [B]
        phase_ids: Optional[torch.LongTensor] = None,      # [B] in {0,1,2}
        offset_log_s: Optional[torch.Tensor] = None        # [B]
    ) -> torch.Tensor:
        """
        前向传播（Forward）。输出**观测尺度**的均值 `mu_pred`（非对数），形状 [B, n_g]。

        公式（Formula）
        --------------
        设
        $$
        \\text{raw}_b = (G K P_{b}) + \\text{bias},\\quad
        \\Delta_b = \\alpha \\cdot \\tanh(\\text{raw}_b),
        $$
        若 use_cycle=True，则
        $$
        \\text{cycle}_b = \\beta_{\\cdot,S}\\,\\mathbf{1}[\\phi_b=S] + \\beta_{\\cdot,G2M}\\,\\mathbf{1}[\\phi_b=G2M].
        $$
        则
        $$
        \\log \\mu_b = \\log \\mu^{\\mathrm{ctrl}} + \\Delta_b + \\text{cycle}_b + \\log s_b,
        \\quad \\mu_b = \\exp(\\log \\mu_b).
        $$

        Returns
        -------
        torch.Tensor
            [B, n_g] 的观测尺度均值（已 clamp 到 [1e-10, 1e7]）。
        """
        # 选择扰动嵌入并计算低秩项
        P_sel = self.P[pert_ids]  # [B, d_p]
        raw = (self.G @ self.K @ P_sel.T).T + self.bias.unsqueeze(0)  # [B, n_g]
        delta_log_mu = self.delta_log_mu_scaler * torch.tanh(raw)     # [B, n_g]

        # 周期项
        if self.use_cycle and phase_ids is not None:
            is_S = (phase_ids == 1).float().unsqueeze(1)   # [B, 1]
            is_G2M = (phase_ids == 2).float().unsqueeze(1)
            cycle_term = is_S @ self.beta_cycle[:, 0].unsqueeze(0) + is_G2M @ self.beta_cycle[:, 1].unsqueeze(0)
            log_mu = torch.log(self.mu_control.unsqueeze(0)) + delta_log_mu + cycle_term
        else:
            log_mu = torch.log(self.mu_control.unsqueeze(0)) + delta_log_mu

        # offset（size factor）
        if offset_log_s is not None:
            log_mu = log_mu + offset_log_s.unsqueeze(1)

        mu_pred = torch.exp(log_mu)
        return torch.clamp(mu_pred, min=1e-10, max=1e7)

    # -------------------------
    # 训练接口
    # -------------------------
    def fit(
        self,
        dataloader,
        loss_type: str = "MSE",
        learning_rate: float = 5e-4,
        n_epochs: int = 100,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 5e-3,
        progress: bool = True
    ) -> None:
        """
        统一训练循环（Unified training loop）。

        dataloader 要求每个 batch 至少包含：
          - 'pert' : LongTensor [B]
          - 'y'    : FloatTensor [B, n_g]
          - 'log_s': FloatTensor [B]
          - （可选）'phase': LongTensor [B]（当 use_cycle=True）

        Parameters
        ----------
        loss_type : str
            取值之一：["MSE","NB","POIS_DEV","NB_DEV","MSE_LOG1P","MSE_ANS"]。
            - "NB": 代表使用 NB 对数似然（negative binomial NLL）。
            - "NB_DEV": 使用 NB 偏差（需要 theta）。
        """
        loss_type = loss_type.upper()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        logger.info(f"[LowRankNB_GLM.fit] Start training with loss: {loss_type}. Device: {self.device}")

        def loss_nb_nll(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            负二项 NLL（允许 y 为实数，使用 Γ 的连续延拓）。
            """
            eps = 1e-8
            theta = self.theta.unsqueeze(0)  # [1, n_g]
            log_theta_mu = torch.log(theta + mu + eps)
            ll = (
                torch.lgamma(theta + y + eps)
                - torch.lgamma(theta + eps)
                - torch.lgamma(y + 1.0)
                + theta * torch.log(theta + eps)
                - theta * log_theta_mu
                + y * torch.log(mu + eps)
                - y * log_theta_mu
            )
            return -ll.mean()

        # 选择损失
        if loss_type == "NB":
            loss_fn = loss_nb_nll
            needs_theta = False
        elif loss_type in LOSSES_NO_THETA:
            loss_fn = LOSSES_NO_THETA[loss_type]
            needs_theta = False
        elif loss_type in LOSSES_WITH_THETA:
            loss_fn_wt = LOSSES_WITH_THETA[loss_type]
            needs_theta = True
        else:
            raise ValueError(f"[LowRankNB_GLM.fit] Unknown loss_type: {loss_type}")

        loop = range(n_epochs)
        if progress:
            try:
                from tqdm import tqdm
                loop = tqdm(loop, desc=f"Training ({loss_type})")
            except Exception:
                pass

        for _ in loop:
            epoch_loss = 0.0
            total_n = 0
            for batch in dataloader:
                pert = batch["pert"].to(self.device)
                y_true = batch["y"].to(self.device)
                log_s = batch["log_s"].to(self.device)
                phase = batch.get("phase", None)
                if phase is not None:
                    phase = phase.to(self.device)

                optimizer.zero_grad()
                mu_pred = self.forward(pert, phase, offset_log_s=log_s)

                if needs_theta:
                    loss = loss_fn_wt(mu_pred, y_true, self.theta)
                else:
                    loss = loss_fn(mu_pred, y_true)

                # 正则项（对 K）
                reg = l1_lambda * self.K.abs().sum() + l2_lambda * (self.K ** 2).sum()
                loss = loss + reg

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                bs = y_true.size(0)
                epoch_loss += loss.item() * bs
                total_n += bs

            if progress and total_n > 0:
                avg_loss = epoch_loss / total_n
                try:
                    loop.set_postfix(loss=f"{avg_loss:.4f}")
                except Exception:
                    pass

    # -------------------------
    # 预测与采样接口
    # -------------------------
    @torch.no_grad()
    def predict_and_sample(
        self,
        pert_ids_test: torch.LongTensor,
        batch_size: int = 1024,
        use_sf: bool = True,
        sf_test: Optional[torch.Tensor] = None,      # [N_test]（若 use_sf=False 可为 None）
        use_gamma: bool = True,
        gamma_r0: float = 50.0,
        sampler: str = "poisson",
        use_cycle: bool = False,
        phase_ids_test: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        预测并采样（Predict & Sample）。

        步骤（Steps）
        ------------
        1) 调用 forward 得到参考尺度 μ_ref 并加 offset 得到观测尺度 μ_obs。
        2) 若启用 Gamma 异质性（*Gamma heterogeneity*），对每个 cell 采样 L ~ Gamma(r0, r0)，
           并令 μ_obs ← μ_obs × L（均值保持不变，增加方差）。
        3) 采样：
           - Poisson：Y ~ Pois(μ_obs)
           - NB（logits 参数化）：
             logits = log(θ) - log(μ_obs)，total_count = θ

        Returns
        -------
        torch.Tensor
            采样得到的计数矩阵 [N_test, n_g]（在 CPU 上）。
        """
        sampler = sampler.lower()
        assert sampler in ("poisson", "nb", "gp_mixture"), "sampler 必须为 'poisson' 或 'nb'"

        self.eval()
        logger.info(f"[Sample] Start prediction & sampling with sampler={sampler}, use_sf={use_sf}, use_gamma={use_gamma}, use_cycle={use_cycle}. Device: {self.device}")
        outs = []
        n = pert_ids_test.numel()
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            pids = pert_ids_test[sl].to(self.device)

            if use_cycle:
                assert phase_ids_test is not None, "[predict_and_sample] use_cycle=True 需要提供 phase_ids_test"
                ph = phase_ids_test[sl].to(self.device)
            else:
                ph = None

            if use_sf:
                assert sf_test is not None, "[predict_and_sample] use_sf=True 需要提供 sf_test"
                sfb = sf_test[sl].to(self.device)
                log_sfb = torch.log(sfb.clamp_min(1e-12))
            else:
                log_sfb = None

            mu_obs = self.forward(pids, ph, offset_log_s=log_sfb)  # [B, n_g]

            # Gamma 异质性（每 cell 一个缩放因子，均值=1，方差=1/r0）
            if use_gamma:
                r0 = torch.tensor(gamma_r0, device=self.device)
                L = torch.distributions.Gamma(concentration=r0, rate=r0).sample((mu_obs.size(0),)).unsqueeze(1)
                mu_obs = (mu_obs * L).clamp_min(1e-12)

            if sampler == "poisson":
                sampled = torch.distributions.Poisson(mu_obs).sample()
            elif sampler == "nb":
                probs_batch = (mu_obs / (mu_obs + self.theta.unsqueeze(0)))
                nb_dist = torch.distributions.NegativeBinomial(total_count=self.theta.unsqueeze(0), probs=probs_batch)
                sampled = nb_dist.sample()
            elif sampler == "gp_mixture":
                theta_b = self.theta.unsqueeze(0)
                mu = mu_obs
                rate = (theta_b / mu)
                lam = torch.distributions.Gamma(concentration=theta_b, rate=rate).sample()
                pois = torch.distributions.Poisson(lam)
                sampled = pois.sample()
            else:
                raise ValueError(f"[predict_and_sample] 未知 sampler: {sampler}")

            outs.append(sampled.cpu())

        return torch.cat(outs, dim=0)
logger = logging.getLogger("nbglm.model")
