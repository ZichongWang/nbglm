
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.special import digamma, polygamma
from numpy.linalg import lstsq

# ===============================================================
# sctransform-like normalization (Hafemeister & Satija, 2019)
# ===============================================================
#
# 约定：
# - X 形状为 (G, N)，G = 基因数，N = 细胞数
# - depth m_j = 每个细胞的 UMI 总数（size factor / sequencing depth）
# - 协变量使用 log10(depth)
# - NB2 参数化：Var(X_ij) = mu_ij + mu_ij^2 / theta_i
# - 三步流程：Poisson-GLM → theta MLE → 核回归正则化 → Pearson residuals
#


@dataclass
class SCTParams:
    """保存每个基因的参数（原始估计与正则化后）。"""
    beta0: np.ndarray
    beta1: np.ndarray
    theta: np.ndarray
    beta0_smooth: np.ndarray
    beta1_smooth: np.ndarray
    theta_smooth: np.ndarray
    gene_mean: np.ndarray


def compute_depth(X: np.ndarray) -> np.ndarray:
    """计算每个细胞的 depth（总 UMI counts）."""
    return X.sum(axis=0)


def geometric_mean_counts(x: np.ndarray, eps: float = 1.0) -> float:
    """几何平均：gm = exp(mean(log(x + eps))) - eps"""
    return float(np.exp(np.mean(np.log(x + eps))) - eps)


def _poisson_glm_irls_single_gene(y: np.ndarray, z: np.ndarray,
                                  max_iter: int = 25,
                                  tol: float = 1e-8) -> Tuple[float, float]:
    """单基因 Poisson-GLM (log link) 拟合：log(mu)=beta0+beta1*z."""
    N = y.shape[0]
    X = np.column_stack([np.ones(N), z])
    beta = lstsq(X, np.log1p(y), rcond=None)[0]  # 初值

    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-8, None)
        W = mu
        z_work = eta + (y - mu) / mu
        WX = X * W[:, None]
        XtWX = X.T @ WX
        XtWz = X.T @ (W * z_work)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = lstsq(XtWX, XtWz, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def _theta_mle_fixed_mu(x: np.ndarray, mu: np.ndarray,
                        theta_init: float = 10.0,
                        max_iter: int = 50,
                        tol: float = 1e-6) -> float:
    """在 mu 固定情况下对 theta 做极大似然估计（NB2，size 参数 theta>0）."""
    theta = float(max(theta_init, 1e-3))
    mu = np.clip(mu, 1e-12, None)
    for _ in range(max_iter):
        a = digamma(x + theta) - digamma(theta)
        b = np.log(theta) - np.log(theta + mu)
        c = (mu - x) / (theta + mu)
        grad = np.sum(a + b + c)

        trig = polygamma(1, x + theta) - polygamma(1, theta)
        d = 1.0 / theta - 1.0 / (theta + mu)
        e = -(mu - x) / (theta + mu) ** 2
        hess = np.sum(trig + d + e)

        if not np.isfinite(grad) or not np.isfinite(hess):
            break
        step = grad / max(hess, -1e12)
        theta_new = theta - step
        theta_new = float(np.clip(theta_new, 1e-3, 1e6))
        if abs(theta_new - theta) < tol * (1.0 + theta):
            theta = theta_new
            break
        theta = theta_new
    return float(theta)


def _nadaraya_watson_gaussian(x: np.ndarray, y: np.ndarray,
                              bandwidth: float) -> np.ndarray:
    """一维高斯核回归（Nadaraya–Watson）."""
    G = x.shape[0]
    Xdiff = x[:, None] - x[None, :]
    W = np.exp(-0.5 * (Xdiff / bandwidth) ** 2)
    denom = W.sum(axis=1) + 1e-12
    return (W @ y) / denom


def _silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman 经验带宽（近似 bw.SJ）."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return 0.1
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    a = min(std, iqr / 1.34) if (std > 0 and iqr > 0) else max(std, iqr)
    h = 0.9 * a * n ** (-1 / 5)
    return float(max(h, 1e-3))


def sctransform_like(
    X: np.ndarray,
    depth: Optional[np.ndarray] = None,
    eps_gm: float = 1.0,
    max_iter_glm: int = 25,
    max_iter_theta: int = 50,
    bw_factor: float = 3.0,
    clip_residuals: bool = True,
) -> Tuple[np.ndarray, SCTParams]:
    """执行 sctransform-like：Poisson-GLM → theta MLE → 核回归 → Pearson residuals."""
    X = np.asarray(X)
    assert X.ndim == 2, "X 必须为二维 (G, N)"
    G, N = X.shape
    if depth is None:
        depth = compute_depth(X)
    depth = np.asarray(depth, float)
    assert depth.shape == (N,), "depth 形状必须是 (N,)"

    z = np.log10(np.clip(depth, 1.0, None))

    beta0 = np.zeros(G, dtype=float)
    beta1 = np.zeros(G, dtype=float)
    mu = np.empty_like(X, dtype=float)
    for i in range(G):
        y = X[i, :].astype(float)
        if y.sum() <= 0:
            beta0[i], beta1[i] = -20.0, 0.0
            mu[i, :] = 1e-9
            continue
        b0, b1 = _poisson_glm_irls_single_gene(y, z, max_iter=max_iter_glm)
        beta0[i], beta1[i] = b0, b1
        mu[i, :] = np.exp(b0 + b1 * z)

    theta = np.zeros(G, dtype=float)
    for i in range(G):
        x_i = X[i, :].astype(float)
        mu_i = mu[i, :]
        if x_i.sum() <= 0:
            theta[i] = 1e6
            continue
        theta[i] = _theta_mle_fixed_mu(x_i, mu_i, theta_init=10.0, max_iter=max_iter_theta)

    gene_mean = np.array([geometric_mean_counts(X[i, :], eps=eps_gm) for i in range(G)], dtype=float)
    gx = np.log10(gene_mean + 1e-8)
    h = bw_factor * _silverman_bandwidth(gx)

    beta0_s = _nadaraya_watson_gaussian(gx, beta0, h)
    beta1_s = _nadaraya_watson_gaussian(gx, beta1, h)
    theta_log = np.log(theta + 1e-8)
    theta_s = np.exp(_nadaraya_watson_gaussian(gx, theta_log, h))

    mu_s = np.exp(beta0_s[:, None] + beta1_s[:, None] * z[None, :])
    var_s = mu_s + (mu_s ** 2) / theta_s[:, None]
    mu_s = np.clip(mu_s, 1e-12, None)
    var_s = np.clip(var_s, 1e-12, None)
    Z = (X - mu_s) / np.sqrt(var_s)

    if clip_residuals:
        Z = np.clip(Z, -np.sqrt(N), np.sqrt(N))

    params = SCTParams(
        beta0=beta0, beta1=beta1, theta=theta,
        beta0_smooth=beta0_s, beta1_smooth=beta1_s, theta_smooth=theta_s,
        gene_mean=gene_mean
    )
    return Z, params


# ==== Sampling utilities =====================================================

def _mu_var_from_params(params: SCTParams, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mu_ij and var_ij from *regularized* parameters for given depth."""
    beta0_s = params.beta0_smooth
    beta1_s = params.beta1_smooth
    theta_s = params.theta_smooth
    z = np.log10(np.clip(depth, 1.0, None))
    mu = np.exp(beta0_s[:, None] + beta1_s[:, None] * z[None, :])
    var = mu + (mu ** 2) / theta_s[:, None]
    return mu, var


def posterior_predictive_nb(params: SCTParams, depth: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Posterior predictive sampling under the fitted NB-GLM (independent across genes)."""
    if rng is None:
        rng = np.random.default_rng()
    mu, _ = _mu_var_from_params(params, depth)
    G, N = mu.shape
    X_new = np.empty((G, N), dtype=int)
    theta = params.theta_smooth.astype(float)
    for i in range(G):
        shape = theta[i]
        rate = theta[i] / np.clip(mu[i, :], 1e-12, None)
        lam = rng.gamma(shape=shape, scale=1.0 / rate, size=N)
        X_new[i, :] = rng.poisson(lam)
    return X_new


def posterior_predictive_nb_match_totals(params: SCTParams, depth: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sampling that *preserves each cell's total UMI exactly* via conditional multinomial."""
    if rng is None:
        rng = np.random.default_rng()
    mu, _ = _mu_var_from_params(params, depth)
    G, N = mu.shape
    X_new = np.empty((G, N), dtype=int)
    theta = params.theta_smooth.astype(float)
    depth_int = np.array(np.round(depth), dtype=int)
    for j in range(N):
        shape = theta  # (G,)
        rate = theta / np.clip(mu[:, j], 1e-12, None)  # (G,)
        lam = rng.gamma(shape=shape, scale=1.0 / rate, size=G)  # (G,)
        s = lam.sum()
        if s <= 0 or depth_int[j] <= 0:
            X_new[:, j] = 0
            continue
        p = lam / s
        p = p / p.sum()
        X_new[:, j] = rng.multinomial(n=int(depth_int[j]), pvals=p)
    return X_new


def invert_from_residuals(Z: np.ndarray, params: SCTParams, depth: np.ndarray, clip: bool = True) -> np.ndarray:
    """Approximate inversion from Pearson residuals: x_hat = z*sqrt(var) + mu."""
    mu, var = _mu_var_from_params(params, depth)
    X_hat = Z * np.sqrt(var) + mu
    X_hat = np.rint(X_hat).astype(int)
    if clip:
        X_hat = np.maximum(X_hat, 0)
    return X_hat


# -------------------------------
# 简单演示（toy example）
# -------------------------------
def _demo(seed: int = 7) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    N = 800
    G = 400
    log10m = rng.normal(4.0, 0.25, size=N)
    m = np.clip(10 ** log10m, 50.0, None)

    g_mean = rng.lognormal(mean=1.0, sigma=1.0, size=G)
    beta0_true = np.log(g_mean + 1.0) - 1.5
    beta1_true = 0.9 + 0.1 * np.tanh((np.log10(g_mean + 1e-8) - 0.5))
    theta_true = 10.0 + 40.0 * (1.0 + np.tanh(np.log(g_mean + 1e-8)))

    MU = np.exp(beta0_true[:, None] + beta1_true[:, None] * log10m[None, :])
    X = np.empty((G, N), dtype=int)
    for i in range(G):
        shape = theta_true[i]
        rate = theta_true[i] / np.clip(MU[i, :], 1e-8, None)
        lam = rng.gamma(shape=shape, scale=1.0 / rate, size=N)
        X[i, :] = rng.poisson(lam)

    Z, params = sctransform_like(X, depth=m, bw_factor=3.0)

    # Sampling tests
    X_nb = posterior_predictive_nb(params, depth=m, rng=rng)
    X_nb_fix = posterior_predictive_nb_match_totals(params, depth=m, rng=rng)

    corr_list = []
    var_list = []
    for i in range(G):
        zi = Z[i, :]
        if np.all(zi == 0):
            continue
        c = np.corrcoef(zi, log10m)[0, 1]
        corr_list.append(c)
        var_list.append(np.var(zi))
    corr_median = float(np.median(np.abs(corr_list)))
    var_median = float(np.median(var_list))

    # Check column sums
    depth_nb = X_nb.sum(axis=0)
    depth_fix = X_nb_fix.sum(axis=0)
    mae_depth_nb = float(np.mean(np.abs(depth_nb - m)))
    maxerr_fix = float(np.max(np.abs(depth_fix - np.round(m))))

    return {"|corr(Z, log10m)|_median": corr_median,
            "Var(Z)_median": var_median,
            "MAE(depth, NB)": mae_depth_nb,
            "maxerr(depth, NB_fix)": maxerr_fix}


if __name__ == "__main__":
    stats = _demo()
    print("Demo summary:", stats)
