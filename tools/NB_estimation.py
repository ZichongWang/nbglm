
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Sequence, List

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from numpy.linalg import lstsq
from scipy.special import digamma, polygamma

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("nb_fit_resample")

@dataclass
class SCTParams:
    beta0: np.ndarray
    beta1: np.ndarray
    theta: np.ndarray
    beta0_smooth: np.ndarray
    beta1_smooth: np.ndarray
    theta_smooth: np.ndarray
    gene_mean: np.ndarray

def _to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)

def _poisson_glm_irls_single_gene(y: np.ndarray, z: np.ndarray,
                                  max_iter: int = 25,
                                  tol: float = 1e-8) -> Tuple[float, float]:
    N = y.shape[0]
    X = np.column_stack([np.ones(N), z])
    beta = lstsq(X, np.log1p(y), rcond=None)[0]
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
        if not np.isfinite(grad) or not np.isfinite(hess) or abs(hess) < 1e-12:
            break
        step = grad / hess
        theta_new = theta - step
        theta_new = float(np.clip(theta_new, 1e-3, 1e9))
        if abs(theta_new - theta) < tol * (1.0 + theta):
            theta = theta_new
            break
        theta = theta_new
    return float(theta)

def _geometric_mean_counts(x: np.ndarray, eps: float = 1.0) -> float:
    return float(np.exp(np.mean(np.log(x + eps))) - eps)

def _silverman_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return 0.1
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    a = min(std, iqr / 1.34) if (std > 0 and iqr > 0) else max(std, iqr)
    h = 0.9 * a * n ** (-1 / 5)
    return float(max(h, 1e-3))

def _nw_gaussian_local(x: np.ndarray, y: np.ndarray, bandwidth: float, truncate: float = 3.0) -> np.ndarray:
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    G = len(x_sorted)
    h = bandwidth
    r = truncate * h
    out_sorted = np.empty(G, dtype=float)
    left = 0
    right = 0
    for i in range(G):
        xi = x_sorted[i]
        while right < G and (x_sorted[right] - xi) <= r:
            right += 1
        while left < i and (xi - x_sorted[left]) > r:
            left += 1
        sl = left
        sr = right
        xx = x_sorted[sl:sr]
        yy = y_sorted[sl:sr]
        w = np.exp(-0.5 * ((xx - xi) / h) ** 2)
        denom = w.sum() + 1e-12
        out_sorted[i] = float((w * yy).sum() / denom)
    out = np.empty(G, dtype=float)
    out[idx] = out_sorted
    return out

def sctransform_like(
    X_gc: np.ndarray,
    depth_c: np.ndarray,
    bw_factor: float = 3.0,
    clip_residuals: bool = True,
    max_iter_glm: int = 25,
    max_iter_theta: int = 50,
) -> Tuple[np.ndarray, SCTParams]:
    X_gc = np.asarray(X_gc)
    G, N = X_gc.shape
    z = np.log10(np.clip(depth_c, 1.0, None))

    beta0 = np.zeros(G, float)
    beta1 = np.zeros(G, float)
    mu_gc = np.empty((G, N), float)
    for i in range(G):
        y = X_gc[i, :].astype(float)
        if y.sum() <= 0:
            beta0[i], beta1[i] = -20.0, 0.0
            mu_gc[i, :] = 1e-9
            continue
        b0, b1 = _poisson_glm_irls_single_gene(y, z, max_iter=max_iter_glm)
        beta0[i], beta1[i] = b0, b1
        mu_gc[i, :] = np.exp(b0 + b1 * z)

    theta = np.zeros(G, float)
    for i in range(G):
        x_i = X_gc[i, :].astype(float)
        mu_i = mu_gc[i, :]
        if x_i.sum() <= 0:
            theta[i] = 1e9
            continue
        theta[i] = _theta_mle_fixed_mu(x_i, mu_i, theta_init=10.0, max_iter=max_iter_theta)

    gene_mean = np.array([_geometric_mean_counts(X_gc[i, :], eps=1.0) for i in range(G)], float)
    gx = np.log10(gene_mean + 1e-8)
    h = bw_factor * _silverman_bandwidth(gx)
    beta0_s = _nw_gaussian_local(gx, beta0, h)
    beta1_s = _nw_gaussian_local(gx, beta1, h)
    theta_s = np.exp(_nw_gaussian_local(gx, np.log(theta + 1e-8), h))

    mu_s = np.exp(beta0_s[:, None] + beta1_s[:, None] * z[None, :])
    var_s = mu_s + (mu_s ** 2) / theta_s[:, None]
    mu_s = np.clip(mu_s, 1e-12, None)
    var_s = np.clip(var_s, 1e-12, None)
    Z_gc = (X_gc - mu_s) / np.sqrt(var_s)
    if clip_residuals:
        Z_gc = np.clip(Z_gc, -np.sqrt(N), np.sqrt(N))

    params = SCTParams(
        beta0=beta0, beta1=beta1, theta=theta,
        beta0_smooth=beta0_s, beta1_smooth=beta1_s, theta_smooth=theta_s,
        gene_mean=gene_mean
    )
    return Z_gc, params

def _nb_sample_preserve_totals(params: SCTParams, depth_c: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    beta0_s = params.beta0_smooth
    beta1_s = params.beta1_smooth
    theta_s = params.theta_smooth
    z = np.log10(np.clip(depth_c, 1.0, None))

    G = beta0_s.shape[0]
    N = depth_c.shape[0]
    X_gc = np.empty((G, N), dtype=int)
    for j in range(N):
        mu_j = np.exp(beta0_s + beta1_s * z[j])
        rate = theta_s / np.clip(mu_j, 1e-12, None)
        lam = rng.gamma(shape=theta_s, scale=1.0 / rate, size=G)
        s = lam.sum()
        n = int(np.round(depth_c[j]))
        if s <= 0 or n <= 0:
            X_gc[:, j] = 0
            continue
        p = lam / s
        p = p / p.sum()
        X_gc[:, j] = rng.multinomial(n=n, pvals=p)
    return X_gc

def _normalize_label(x: str) -> str:
    if x is None:
        return ""
    return str(x).lower().replace(" ", "").replace("-", "").replace("_", "")

def run(
    in_h5ad: str,
    out_h5ad: str,
    out_csv: str,
    pert_col: str = "target_gene",
    depth_col: str = "total_counts",
    non_targeting_values: Sequence[str] = ("non-targeting",),
    counts_layer: Optional[str] = None,
    preserve_totals: bool = True,
    bw_factor: float = 3.0,
    store_layers: bool = True,
):
    logger.info(f"Reading AnnData: {in_h5ad}")
    adata = sc.read_h5ad(in_h5ad)

    assert pert_col in adata.obs, f"obs lacks {pert_col!r}"
    assert depth_col in adata.obs, f"obs lacks {depth_col!r}"

    if counts_layer is not None:
        assert counts_layer in adata.layers, f"layer {counts_layer!r} not found"
        counts_cg = _to_dense(adata.layers[counts_layer])
    elif "counts" in adata.layers:
        counts_cg = _to_dense(adata.layers["counts"])
    else:
        counts_cg = _to_dense(adata.X)
    N, G = counts_cg.shape
    logger.info(f"Data shape: cells={N}, genes={G}")

    perts = adata.obs[pert_col].astype(str).to_numpy()
    depth = adata.obs[depth_col].to_numpy().astype(float)

    unique_perts = np.unique(perts)
    nt_norm = {_normalize_label(v) for v in non_targeting_values}

    out_counts_cg = np.zeros_like(counts_cg, dtype=np.int32)
    rows: List[Dict] = []

    logger.info(f"Found {len(unique_perts)} perturbations.")
    for k, pert in enumerate(unique_perts, 1):
        mask = (perts == pert)
        n_cells = int(mask.sum())
        logger.info(f"[{k}/{len(unique_perts)}] Perturbation={pert!r} (cells={n_cells}) ...")

        X_sub_cg = counts_cg[mask, :]
        m_sub = depth[mask]

        X_sub_gc = X_sub_cg.T.copy()

        logger.info("  - Fitting NB-GLM + smoothing ...")
        Z_gc, params = sctransform_like(
            X_sub_gc,
            depth_c=m_sub,
            bw_factor=bw_factor,
            clip_residuals=True,
        )

        gene_names = adata.var_names.to_numpy()
        assert gene_names.shape[0] == G, "var_names length mismatch"
        df_params = pd.DataFrame({
            "perturbation": pert,
            "gene": gene_names,
            "beta0": params.beta0,
            "beta1": params.beta1,
            "theta": params.theta,
            "beta0_smooth": params.beta0_smooth,
            "beta1_smooth": params.beta1_smooth,
            "theta_smooth": params.theta_smooth,
            "gene_mean": params.gene_mean,
            "n_cells": n_cells,
        })
        rows.append(df_params)

        if _normalize_label(pert) in nt_norm:
            logger.info("  - Non-targeting: copying original counts (no sampling).")
            out_counts_cg[mask, :] = X_sub_cg
        else:
            logger.info(f"  - Sampling counts (preserve_totals={preserve_totals}) ...")
            if preserve_totals:
                X_new_gc = _nb_sample_preserve_totals(params, depth_c=m_sub)
            else:
                rng = np.random.default_rng()
                beta0_s = params.beta0_smooth
                beta1_s = params.beta1_smooth
                theta_s = params.theta_smooth
                z = np.log10(np.clip(m_sub, 1.0, None))
                G2, N2 = X_sub_gc.shape
                X_new_gc = np.empty((G2, N2), dtype=int)
                for i in range(G2):
                    mu = np.exp(beta0_s[i] + beta1_s[i] * z)
                    rate = theta_s[i] / np.clip(mu, 1e-12, None)
                    lam = rng.gamma(shape=theta_s[i], scale=1.0 / rate, size=N2)
                    X_new_gc[i, :] = rng.poisson(lam)
            out_counts_cg[mask, :] = X_new_gc.T

    logger.info("Assembling new AnnData ...")
    new_adata = ad.AnnData(
        X=out_counts_cg,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        dtype=out_counts_cg.dtype,
    )
    if store_layers:
        new_adata.layers["counts_nb"] = out_counts_cg.astype(np.int32)
        if "counts" in adata.layers:
            new_adata.layers["counts_orig"] = _to_dense(adata.layers["counts"]).astype(np.int32)
        else:
            new_adata.layers["counts_orig"] = _to_dense(adata.X).astype(np.int32)

    logger.info(f"Writing H5AD to: {out_h5ad}")
    new_adata.write_h5ad(out_h5ad, compression="gzip")

    logger.info(f"Writing parameters CSV to: {out_csv}")
    df_all = pd.concat(rows, axis=0, ignore_index=True)
    df_all.to_csv(out_csv, index=False)

    logger.info("Done.")

def main():
    p = argparse.ArgumentParser(description="Per-perturbation NB fitting & resampling (single-file).")
    p.add_argument("--in_h5ad", type=str, default="/home/wzc26/work/vcc/nbglm/data/Official_Data_Split/test.h5ad")
    p.add_argument("--out_h5ad", type=str, default="test_nb_per_pert.h5ad")
    p.add_argument("--out_csv", type=str, default="nb_params_per_pert.csv")
    p.add_argument("--pert_col", type=str, default="target_gene")
    p.add_argument("--depth_col", type=str, default="total_counts")
    p.add_argument("--non_targeting_values", type=str, nargs="*", default=["non-targeting"])
    p.add_argument("--counts_layer", type=str, default=None)
    p.add_argument("--preserve_totals", action="store_true", default=False)
    p.add_argument("--no-preserve_totals", dest="preserve_totals", action="store_false")
    p.add_argument("--bw_factor", type=float, default=3.0)
    p.add_argument("--no_layers", dest="store_layers", action="store_false", default=True)
    args = p.parse_args()

    run(
        in_h5ad=args.in_h5ad,
        out_h5ad=args.out_h5ad,
        out_csv=args.out_csv,
        pert_col=args.pert_col,
        depth_col=args.depth_col,
        non_targeting_values=args.non_targeting_values,
        counts_layer=args.counts_layer,
        preserve_totals=args.preserve_totals,
        bw_factor=args.bw_factor,
        store_layers=args.store_layers,
    )

if __name__ == "__main__":
    main()
