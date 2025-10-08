# nbglm 低秩负二项广义线性模型（Low-Rank NB‑GLM）

本文结合仓库实现，系统介绍 nbglm 的建模假设与训练/采样流程。实现对应：`src/nbglm/model.py`, `src/nbglm/dataset.py`, `src/nbglm/pipelines.py`。

## 记号与数据
- 计数矩阵：细胞×基因 `X ∈ N^{N×G}`，文库深度 `L_i = \sum_g X_{ig}`。
- 基因嵌入 `G ∈ R^{G×d_g}` 与扰动嵌入 `P ∈ R^{P×d_p}`（CSV 读入并行归一化）。
- 低秩核参数 `K ∈ R^{d_g×d_p}`（可学习），基因偏置 `b ∈ R^{G}`，缩放系数 `α`（标量）。
- 控制组参考均值 `μ^{ctrl} ∈ R^{G}`（由 control 细胞均值得到）。
- 可选固定效应：细胞周期 `β ∈ R^{G×2}`（S/G2M 两列；G1 为基线）。
- size factor（offset）：`s_i = L_i / median(L)`，在 GLM 中以 `log s_i` 加入。

## 前向（参考尺度 → 观测尺度）
对第 `b` 个样本、基因 `g`（扰动 id 为 `p(b)`，phase 为 `φ_b`），先计算低秩交互：
$$
\text{raw}_b = (G K P_{p(b)}) + b,\qquad
\Delta_b = α\,\tanh(\text{raw}_b).
$$
若启用周期：
$$
\text{cycle}_{b,g} = β_{g,S}\,\mathbf{1}[\phi_b=S] + β_{g,G2M}\,\mathbf{1}[\phi_b=G2M].
$$
则参考尺度对数均值与观测尺度期望为：
$$
\log \mu_b = \log \mu^{ctrl} + \Delta_b + \text{cycle}_b + \log s_b,\quad
\mu_b = \exp(\log \mu_b).
$$
实现细节：`tanh` 保证效应幅度有界，`α=delta_log_mu_scaler` 提供全局缩放；`b` 为作用于效应项的基因级偏置。

## 观测模型：负二项（NB）
为刻画过度离散，以基因特异离散度 `θ_g` 建模：
$$
Y_{bg} \sim \mathrm{NB}(\text{total\_count}=\theta_g,\; \text{probs}=\mu_{bg}/(\mu_{bg}+\theta_g)).
$$
其均值与方差为 `E[Y]=μ`，`Var[Y]=μ+μ^2/θ`。实现也支持 Poisson 与显式 Gamma–Poisson 混合采样。

## 损失函数与正则化
主损失由配置 `model.losses.primary` 指定：
- NB 对数似然（"NB"）：对 `Y|μ,θ` 的负对数似然。
- 偏差类：Poisson/NB deviance（"POIS_DEV"/"NB_DEV"）。
- 回归类：`MSE`、`MSE_LOG1P`、Anscombe 变换的 `MSE_ANS`。

对核矩阵 `K` 施加稀疏与权重衰减：
$$
\mathcal{L}_{reg} = \lambda_1\|K\|_1 + \lambda_2\|K\|_F^2.
$$
总目标 `\mathcal{L} = \mathcal{L}_{data} + \mathcal{L}_{reg}`。优化采用 AdamW，梯度裁剪 `max_norm=1.0`。

## 统计量估计与数据组织
- `μ^{ctrl}`：control 细胞均值。
- `θ_g`：对 control 使用矩估计（MoM）
  $$\theta_g \approx \mu_g^2 / \max(\mathrm{Var}_g-\mu_g,\varepsilon).$$
- size factor：`s_i = L_i/median(L)`，训练/采样均以 `log s` 作为 offset。
- 伪批（pseudo-bulk）：可将同扰动（或扰动×phase）的细胞在 CPU 上平均聚合，并用聚合单元的平均文库深度构造对应的 `log s`（形状与模型前向对齐）。

## 采样与异质性
推理时先得到 `μ`，可选乘以 `L ~ \mathrm{Gamma}(r_0,r_0)` 引入细胞级异质性（均值不变、方差增大），再按 Poisson/NB 抽样。提供三种 `sampler`：`poisson`、`nb`、`gp_mixture`（显式 Gamma–Poisson）。

## 形状与实现对齐（要点）
- `G:[G,d_g]`, `P:[P,d_p]`, `K:[d_g,d_p]`，前向产生 `μ_pred:[B,G]`。
- 周期标签编码 `G1/S/G2M → 0/1/2`；`β:[G,2]` 仅在使能周期时参与。
- 配置项映射：`model.regularization.{l1,l2}`、`size_factor.use_sf`、`sampling.*` 与训练/采样接口一一对应。

以上构成了 nbglm 的低秩表征（嵌入×核）+ 偏置/固定效应 + 负二项观测的完整闭环，可在保持参数高效的同时建模高维基因响应的非线性、过度离散与技术批次规模效应。

