#!/usr/bin/env python3
"""VCC evaluation backend compatible with nbglm pipelines.

This module can be imported by the pipeline (``evaluate.backend = "vcc"``)
while still providing a CLI for ad-hoc usage. It reproduces the MAE, PDS, and
DES metrics using the VCC implementation based on ``pdex`` and Polars.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from pdex import parallel_differential_expression
from scipy.sparse import issparse
from sklearn.metrics import mean_absolute_error, pairwise_distances


logger = logging.getLogger("nbglm.vcc_eval")


# ---------------------------------------------------------------------------
# Normalization helpers (adapted from the standalone script)
# ---------------------------------------------------------------------------

def guess_is_lognorm(adata: ad.AnnData, n_cells: int | float = 5e2, epsilon: float = 1e-2) -> bool:
    """Heuristic to decide whether an AnnData matrix is already log-normalized."""
    n_cells = int(min(adata.shape[0], n_cells))
    if n_cells == 0:
        return False
    cell_mask = np.random.choice(adata.shape[0], n_cells, replace=False)
    cell_sums = adata.X[cell_mask].sum(axis=1)
    if issparse(cell_sums):  # type: ignore[attr-defined]
        cell_sums = cell_sums.A
    cell_sums = np.asarray(cell_sums).flatten()
    return bool(np.any(np.abs((cell_sums - np.round(cell_sums))) > epsilon))


def ensure_norm_log(adata: ad.AnnData, allow_discrete: bool, label: str) -> None:
    """Apply total-count normalization and log1p if the matrix looks discrete."""
    if guess_is_lognorm(adata):
        logger.info("%s already appears log-normalized; skipping conversion", label)
        return
    if allow_discrete:
        logger.info("%s contains integer counts; leaving data as-is (allow_discrete)", label)
        return
    logger.debug("Normalizing and log-transforming %s", label)
    sc.pp.normalize_total(adata=adata, inplace=True)
    sc.pp.log1p(adata)


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_real_cache(cache_dir: Path) -> dict[str, object]:
    """Load cached ground-truth artifacts from a cache directory."""
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in cache directory {cache_dir}")
    metadata = json.loads(metadata_path.read_text())

    bulk_path = cache_dir / metadata["bulk_file"]
    if not bulk_path.exists():
        raise FileNotFoundError(f"Cached bulk file not found: {bulk_path}")
    with np.load(bulk_path, allow_pickle=True) as npz:
        bulk_keys = np.asarray(npz["bulk_keys"], dtype=str)
        bulk_values = np.asarray(npz["bulk_values"], dtype=float)
        genes = np.asarray(npz["genes"], dtype=str)

    de_path = cache_dir / metadata["real_de_file"]
    if not de_path.exists():
        raise FileNotFoundError(f"Cached differential expression file not found: {de_path}")

    return {
        "metadata": metadata,
        "bulk_keys": bulk_keys,
        "bulk_values": bulk_values,
        "genes": genes,
        "real_de_path": de_path,
    }


# ---------------------------------------------------------------------------
# Perturbation pair utilities (trimmed down version of PerturbationAnndataPair)
# ---------------------------------------------------------------------------


class BulkArrays:
    """Container for pseudobulk arrays of a perturbation and its controls."""

    def __init__(self, key: str, pert_real: np.ndarray, pert_pred: np.ndarray, ctrl_real: np.ndarray, ctrl_pred: np.ndarray) -> None:
        self.key = key
        self.pert_real = pert_real
        self.pert_pred = pert_pred
        self.ctrl_real = ctrl_real
        self.ctrl_pred = ctrl_pred

    def perturbation_effect(self, which: str, abs_values: bool = False) -> np.ndarray:
        if which == "real":
            effect = self.pert_real - self.ctrl_real
        elif which == "pred":
            effect = self.pert_pred - self.ctrl_pred
        else:
            raise ValueError(f"Invalid option for `which`: {which}")
        return np.abs(effect) if abs_values else effect


class PerturbationPair:
    """Minimal view over AnnData pair for perturbation-level metrics."""

    def __init__(
        self,
        real: ad.AnnData,
        pred: ad.AnnData,
        pert_col: str,
        control_pert: str,
        cached_real_bulk: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        if real.shape[1] != pred.shape[1]:
            raise ValueError("Real and predicted AnnData must have the same number of genes")

        real_genes = np.asarray(real.var.index.values)
        pred_genes = np.asarray(pred.var.index.values)
        if not np.array_equal(real_genes, pred_genes):
            raise ValueError("Real and predicted AnnData must share identical gene order")

        if pert_col not in real.obs.columns or pert_col not in pred.obs.columns:
            raise ValueError(f"Perturbation column '{pert_col}' missing in one of the AnnData objects")

        self.real = real
        self.pred = pred
        self.pert_col = pert_col
        self.control_pert = control_pert
        self.genes = real_genes.astype(str)

        real_labels = real.obs[pert_col].to_numpy(str)
        pred_labels = pred.obs[pert_col].to_numpy(str)
        perts_real = np.unique(real_labels)
        perts_pred = np.unique(pred_labels)
        if not np.array_equal(perts_real, perts_pred):
            raise ValueError("Real and predicted AnnData list different perturbations")
        if control_pert not in perts_real:
            raise ValueError(f"Control perturbation '{control_pert}' not found in AnnData obs")

        perts = np.union1d(perts_real, perts_pred)
        self.perts = np.array([p for p in perts if p != control_pert])
        self._pert_mask_real = self._make_mask(real_labels)
        self._pert_mask_pred = self._make_mask(pred_labels)

        self._bulk_real_keys: Optional[np.ndarray] = None
        self._bulk_real_values: Optional[np.ndarray] = None
        self._bulk_pred_keys: Optional[np.ndarray] = None
        self._bulk_pred_values: Optional[np.ndarray] = None

        if cached_real_bulk is not None:
            keys, values = cached_real_bulk
            keys = np.asarray(keys, dtype=str)
            if control_pert not in keys:
                raise ValueError("Cached bulk is missing the control perturbation")
            expected = set(np.union1d(perts, np.array([control_pert])))
            if set(keys) != expected:
                raise ValueError("Cached bulk perturbations do not match AnnData")
            self._bulk_real_keys = keys
            self._bulk_real_values = np.asarray(values, dtype=float)

    @staticmethod
    def _make_mask(labels: np.ndarray) -> dict[str, np.ndarray]:
        unique, inverse = np.unique(labels, return_inverse=True)
        return {str(unique[i]): np.where(inverse == i)[0] for i in range(unique.size)}

    def _build_bulk(self, adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
        matrix = adata.X
        if issparse(matrix):
            logger.debug("Converting sparse matrix to dense for pseudobulk computation")
            matrix = matrix.toarray()  # type: ignore[attr-defined]
        frame = pl.DataFrame(matrix).with_columns(
            pl.Series("groupby_key", adata.obs[self.pert_col].to_numpy(str))
        )
        bulked = frame.group_by("groupby_key").mean().sort("groupby_key")
        keys = bulked["groupby_key"].to_numpy()
        values = bulked.drop("groupby_key").to_numpy()
        return keys.astype(str), values

    def _ensure_bulk(self) -> None:
        if self._bulk_real_keys is None or self._bulk_real_values is None:
            self._bulk_real_keys, self._bulk_real_values = self._build_bulk(self.real)
        if self._bulk_pred_keys is None or self._bulk_pred_values is None:
            self._bulk_pred_keys, self._bulk_pred_values = self._build_bulk(self.pred)
        assert self._bulk_real_keys is not None and self._bulk_pred_keys is not None
        if not np.array_equal(self._bulk_real_keys, self._bulk_pred_keys):
            raise ValueError("Mismatch between real and predicted pseudobulk perturbations")
        if self.control_pert not in self._bulk_real_keys:
            raise ValueError("Control perturbation missing in pseudobulk arrays")

    def iter_bulk_arrays(self) -> Iterator[BulkArrays]:
        self._ensure_bulk()
        assert self._bulk_real_keys is not None
        assert self._bulk_real_values is not None
        assert self._bulk_pred_values is not None
        ctrl_idx = int(np.flatnonzero(self._bulk_real_keys == self.control_pert)[0])
        ctrl_real = self._bulk_real_values[ctrl_idx]
        ctrl_pred = self._bulk_pred_values[ctrl_idx]
        for pert in self.perts:
            idx = int(np.flatnonzero(self._bulk_real_keys == pert)[0])
            yield BulkArrays(
                key=str(pert),
                pert_real=self._bulk_real_values[idx],
                pert_pred=self._bulk_pred_values[idx],
                ctrl_real=ctrl_real,
                ctrl_pred=ctrl_pred,
            )

    def ctrl_matrix(self, which: str) -> np.ndarray:
        matrix = self.real.X if which == "real" else self.pred.X
        mask_lookup = self._pert_mask_real if which == "real" else self._pert_mask_pred
        ctrl_idx = mask_lookup[self.control_pert]
        ctrl_matrix = matrix[ctrl_idx]
        if issparse(ctrl_matrix):
            ctrl_matrix = ctrl_matrix.toarray()  # type: ignore[attr-defined]
        return np.asarray(ctrl_matrix)


# ---------------------------------------------------------------------------
# Metric implementations (mirroring cell_eval.metrics behaviour)
# ---------------------------------------------------------------------------

def compute_mae(pair: PerturbationPair) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for bulk in pair.iter_bulk_arrays():
        value = mean_absolute_error(bulk.pert_pred, bulk.pert_real)
        scores[bulk.key] = float(value)
    return scores


def compute_discrimination_score_l1(pair: PerturbationPair) -> Dict[str, float]:
    if pair.perts.size == 0:
        return {}
    real_effects = []
    pred_effects = []
    for bulk in pair.iter_bulk_arrays():
        real_effects.append(bulk.perturbation_effect("real"))
        pred_effects.append(bulk.perturbation_effect("pred"))
    real_effects = np.vstack(real_effects)
    pred_effects = np.vstack(pred_effects)

    norm_ranks: Dict[str, float] = {}
    for idx, pert in enumerate(pair.perts):
        include_mask = np.flatnonzero(pair.genes != pert)
        if include_mask.size == 0:
            include_mask = np.ones(real_effects.shape[1], dtype=bool)
        distances = pairwise_distances(
            real_effects[:, include_mask],
            pred_effects[idx, include_mask].reshape(1, -1),
            metric="l1",
        ).flatten()
        sorted_indices = np.argsort(distances)
        pert_index = int(np.flatnonzero(pair.perts == pert)[0])
        rank = int(np.flatnonzero(sorted_indices == pert_index)[0])
        norm_rank = rank / pair.perts.size
        norm_ranks[str(pert)] = 1.0 - norm_rank
    return norm_ranks


# ---------------------------------------------------------------------------
# Differential expression helpers for overlap_at_N
# ---------------------------------------------------------------------------

def compute_pdex(
    adata: ad.AnnData,
    control: str,
    pert_col: str,
    de_method: str,
    num_workers: int,
    batch_size: int,
) -> pl.DataFrame:
    frame = parallel_differential_expression(
        adata=adata,
        reference=control,
        groupby_key=pert_col,
        metric=de_method,
        num_workers=num_workers,
        batch_size=batch_size,
        as_polars=True,
    )
    return frame  # already a Polars DataFrame


def sanitize_de_results(frame: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = ["fold_change", "p_value", "fdr", "log2_fold_change", "abs_log2_fold_change"]
    categorical_cols = ["target", "feature"]

    if "log2_fold_change" not in frame.columns:
        frame = frame.with_columns(
            pl.col("fold_change").log(base=2).alias("log2_fold_change").fill_nan(0.0)
        )
    if "abs_log2_fold_change" not in frame.columns:
        frame = frame.with_columns(pl.col("log2_fold_change").abs().alias("abs_log2_fold_change"))

    frame = frame.with_columns([pl.col(c).cast(pl.Float32) for c in numeric_cols if c in frame.columns])
    frame = frame.with_columns([pl.col(c).cast(pl.Categorical) for c in categorical_cols if c in frame.columns])
    keep = [c for c in numeric_cols + categorical_cols if c in frame.columns]
    return frame.select(keep)


def build_rank_matrix(
    frame: pl.DataFrame,
    perts: np.ndarray,
    fdr_threshold: float | None = None,
) -> pl.DataFrame:
    threshold = 0.05 if fdr_threshold is None else fdr_threshold
    descending = True  # abs_log2_fold_change is default and sorted descending
    filtered = frame.filter(pl.col("fdr") < threshold)
    if filtered.height == 0:
        return pl.DataFrame()
    rank_df = (
        filtered.with_columns(
            rank=pl.struct("abs_log2_fold_change")
            .rank("ordinal", descending=descending)
            .over("target")
            - 1
        )
        .pivot(index="rank", on="target", values="feature")
        .sort("rank")
    )
    missing = {str(p) for p in perts} - set(rank_df.columns)
    if missing:
        rank_df = rank_df.with_columns([pl.lit(None).alias(col) for col in sorted(missing)])
    return rank_df


def compute_overlap_at_n(
    real_df: pl.DataFrame,
    pred_df: pl.DataFrame,
    perts: np.ndarray,
    k: int | None = None,
    metric: str = "overlap",
    fdr_threshold: float | None = None,
) -> Dict[str, float]:
    real_rank = build_rank_matrix(real_df, perts, fdr_threshold=fdr_threshold)
    pred_rank = build_rank_matrix(pred_df, perts, fdr_threshold=fdr_threshold)

    if real_rank.height == 0 or pred_rank.height == 0:
        return {str(pert): 0.0 for pert in perts}

    overlaps: Dict[str, float] = {}
    for pert in perts:
        pert = str(pert)
        if pert not in real_rank.columns or pert not in pred_rank.columns:
            overlaps[pert] = 0.0
            continue
        real_genes = real_rank[pert].drop_nulls().to_numpy()
        pred_genes = pred_rank[pert].drop_nulls().to_numpy()
        if metric == "overlap":
            k_eff = real_genes.size if k is None else min(k, real_genes.size)
        elif metric == "precision":
            k_eff = pred_genes.size if k is None else min(k, pred_genes.size)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        if k_eff == 0:
            overlaps[pert] = 0.0
            continue
        real_subset = real_genes[:k_eff]
        pred_subset = pred_genes[:k_eff]
        overlaps[pert] = np.intersect1d(real_subset, pred_subset).size / k_eff
    return overlaps


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _json_dump(path: Path, obj: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def _ensure_anndata(obj: Union[str, Path, ad.AnnData]) -> ad.AnnData:
    if isinstance(obj, ad.AnnData):
        return obj.copy()
    return ad.read_h5ad(str(obj))


def _resolve_workers(n_jobs: Union[int, str]) -> int:
    if isinstance(n_jobs, str) and n_jobs.lower() == "auto":
        try:
            cpu_total = mp.cpu_count() or 1
        except Exception:  # pragma: no cover - platform specific
            cpu_total = 1
        limit = max(1, int(cpu_total * 0.9))
        return limit
    try:
        jobs = int(n_jobs)
    except Exception:
        jobs = -1
    if jobs <= 0:
        try:
            cpu_total = mp.cpu_count() or 1
        except Exception:  # pragma: no cover - platform specific
            cpu_total = 1
        jobs = cpu_total
    return max(1, jobs)


def _prepare_cache(
    cache_path: Optional[Union[str, Path]],
    pert_col: str,
    control_name: str,
    required_perts: np.ndarray,
    genes: np.ndarray,
) -> Optional[dict[str, object]]:
    if cache_path is None:
        return None
    cache_dir = Path(cache_path)
    if not cache_dir.exists() or not cache_dir.is_dir():
        logger.warning("true_de_cache path '%s' is not a directory; ignoring for vcc backend", cache_dir)
        return None
    cache = load_real_cache(cache_dir)
    metadata = cache["metadata"]  # type: ignore[index]
    if metadata.get("pert_col") != pert_col:
        logger.warning(
            "Cache pert_col mismatch (cache=%s, expected=%s); ignoring cache",
            metadata.get("pert_col"),
            pert_col,
        )
        return None
    if metadata.get("control") != control_name:
        logger.warning(
            "Cache control mismatch (cache=%s, expected=%s); ignoring cache",
            metadata.get("control"),
            control_name,
        )
        return None

    cached_genes = np.asarray(cache["genes"], dtype=str)  # type: ignore[index]
    if not np.array_equal(cached_genes, genes):
        logger.warning("Gene order in cache does not match evaluation data; ignoring cache")
        return None

    bulk_keys = np.asarray(cache["bulk_keys"], dtype=str)  # type: ignore[index]
    bulk_values = np.asarray(cache["bulk_values"], dtype=float)  # type: ignore[index]
    key_to_idx = {k: i for i, k in enumerate(bulk_keys)}
    required_all = [str(p) for p in required_perts]
    missing = [p for p in required_all if p not in key_to_idx]
    if missing:
        logger.warning("Cached bulk missing required perturbations: %s; ignoring cache", ", ".join(missing))
        return None
    order_idx = [key_to_idx[p] for p in required_all]
    cached_bulk = (bulk_keys[order_idx], bulk_values[order_idx])

    cache_info = {
        "bulk": cached_bulk,
        "real_de_path": cache["real_de_path"],  # type: ignore[index]
    }
    return cache_info


# ---------------------------------------------------------------------------
# Public evaluate function (pipeline entry point)
# ---------------------------------------------------------------------------

def evaluate(
    pred_adata_or_path: Union[str, Path, ad.AnnData],
    true_adata_or_path: Union[str, Path, ad.AnnData],
    pert_col: str,
    control_name: str,
    metrics: List[str],
    control_adata_path: str | None = None,  # unused but kept for API parity
    run_dir: Optional[str] = None,
    cache_path: Optional[Union[str, Path]] = None,
    n_jobs: Union[int, str] = "auto",
    normalize: bool = True,
    save_json: bool = True,
    de_method: str = "wilcoxon",
    batch_size: int = 1024,
) -> Dict[str, float]:
    """Evaluate predictions using the VCC metric implementation."""

    del control_adata_path  # API parity with legacy backend
    pl.enable_string_cache()

    real = _ensure_anndata(true_adata_or_path)
    pred = _ensure_anndata(pred_adata_or_path)

    common_genes = real.var_names.intersection(pred.var_names)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between real and predicted AnnData")
    real = real[:, common_genes].copy()
    pred = pred[:, real.var_names].copy()

    allow_discrete = not normalize
    ensure_norm_log(real, allow_discrete=allow_discrete, label="Real AnnData")
    ensure_norm_log(pred, allow_discrete=allow_discrete, label="Predicted AnnData")

    pred_labels = pred.obs[pert_col].to_numpy(str)
    pred_perts = np.unique(pred_labels)
    if control_name not in pred_perts:
        raise ValueError("Control perturbation missing from predicted AnnData")

    real_labels_full = real.obs[pert_col].to_numpy(str)
    real_perts_full = set(np.unique(real_labels_full))
    if control_name not in real_perts_full:
        raise ValueError("Control perturbation missing from real AnnData")
    missing_in_real = sorted(set(pred_perts) - real_perts_full)
    if missing_in_real:
        raise ValueError(
            "Real AnnData is missing perturbations present in prediction: "
            + ", ".join(missing_in_real)
        )

    required_all = np.union1d(pred_perts, np.array([control_name]))
    real_mask = np.isin(real_labels_full, required_all)
    if real_mask.sum() == 0:
        raise ValueError("No overlapping perturbations between real and predicted AnnData")
    real = real[real_mask].copy()

    cache_info = _prepare_cache(cache_path, pert_col, control_name, required_all, real.var_names.to_numpy(str))
    cached_bulk = cache_info["bulk"] if cache_info is not None else None  # type: ignore[index]

    pair = PerturbationPair(
        real=real,
        pred=pred,
        pert_col=pert_col,
        control_pert=control_name,
        cached_real_bulk=cached_bulk,  # type: ignore[arg-type]
    )

    results: Dict[str, float] = {}
    wanted = {m.upper() for m in metrics}
    
    if "MAE" in wanted:
        mae_scores = compute_mae(pair)
        value = float(np.mean(list(mae_scores.values()))) if mae_scores else 0.0
        results["MAE"] = value
        logger.info("[vcc_eval] MAE: %.6f", value)

    if "PDS" in wanted:
        discrimination_scores = compute_discrimination_score_l1(pair)
        value = float(np.mean(list(discrimination_scores.values()))) if discrimination_scores else 0.0
        results["PDS"] = value
        logger.info("[vcc_eval] PDS: %.6f", value)

    # Always compute DE frames (for CSV export and potential DES computation)
    workers = _resolve_workers(n_jobs)
    with pl.StringCache():
        if cache_info is not None:
            real_de_path = cache_info["real_de_path"]  # type: ignore[index]
            real_de = sanitize_de_results(pl.read_parquet(real_de_path))  # type: ignore[arg-type]
        else:
            real_de = sanitize_de_results(
                compute_pdex(
                    real,
                    control=control_name,
                    pert_col=pert_col,
                    de_method=de_method,
                    num_workers=workers,
                    batch_size=batch_size,
                )
            )
        # For DES we include control; for CSV we will filter to perts only later
        required_targets_all = {str(p) for p in pair.perts}
        required_targets_all.add(control_name)
        real_de = real_de.filter(pl.col("target").is_in(list(required_targets_all)))

        pred_de = sanitize_de_results(
            compute_pdex(
                pred,
                control=control_name,
                pert_col=pert_col,
                de_method=de_method,
                num_workers=workers,
                batch_size=batch_size,
            )
        )
        pred_de = pred_de.filter(pl.col("target").is_in(list(required_targets_all)))



    if "DES" in wanted:
        logger.info("[vcc_eval] Computing DES with %d workers", workers)
        # Use precomputed and filtered DE frames
        overlap_scores = compute_overlap_at_n(real_de, pred_de, pair.perts, k=None, metric="overlap", fdr_threshold=None)
        value = float(np.mean(list(overlap_scores.values()))) if overlap_scores else 0.0
        results["DES"] = value
        logger.info("[vcc_eval] DES: %.6f", value)

    if all(k in results for k in ("DES", "PDS", "MAE")):
        # des_baseline = 0.106
        # pds_baseline = 0.516
        # mae_baseline = 0.027

        des_baseline = 0.0761
        pds_baseline = 0.52
        mae_baseline = 0.0269

        des_scaled = float(np.clip((results["DES"] - des_baseline) / (1 - des_baseline), 0, 1))
        pds_scaled = float(np.clip((results["PDS"] - pds_baseline) / (1 - pds_baseline), 0, 1))
        mae_scaled = float(np.clip((mae_baseline - results["MAE"]) / mae_baseline, 0, 1))
        overall = 100.0 * (des_scaled + pds_scaled + mae_scaled) / 3.0
        results["Overall"] = overall
        logger.info("[vcc_eval] Overall score: %.6f", overall)

        overall_wo_mae = (des_scaled + pds_scaled) * (100.0 / 3.0)
        results["Overall_wo_MAE"] = overall_wo_mae
        logger.info("[vcc_eval] Overall score (DES+PDS): %.6f", overall_wo_mae)

    # Export full DE table (per perturbation x gene) for both ground truth and prediction
    if run_dir is not None:
        try:
            # Filter export to perturbations only (exclude control)
            export_targets = [str(p) for p in pair.perts]
            real_export = (
                real_de
                .filter(pl.col("target").is_in(export_targets))
                .select(["feature", "target", "p_value", "fdr", "log2_fold_change"])
                .rename({
                    "feature": "gene",
                    "target": "pert",
                    "p_value": "gt_pvals",
                    "fdr": "gt_fdr",
                    "log2_fold_change": "gt_logfoldchanges",
                })
            )
            pred_export = (
                pred_de
                .filter(pl.col("target").is_in(export_targets))
                .select(["feature", "target", "p_value", "fdr", "log2_fold_change"])
                .rename({
                    "feature": "gene",
                    "target": "pert",
                    "p_value": "prediction_pvals",
                    "fdr": "prediction_fdr",
                    "log2_fold_change": "prediction_logfoldchanges",
                })
            )

            merged = real_export.join(pred_export, on=["gene", "pert"], how="outer")
            merged = merged.with_columns([
                (pl.col("gt_fdr") < 0.05).alias("gt_significant"),
                (pl.col("prediction_fdr") < 0.05).alias("pred_significant"),
            ])
            merged = merged.with_columns([
                pl.col("gt_significant").fill_null(False),
                pl.col("pred_significant").fill_null(False),
            ])
            merged = merged.select([
                "gene",
                "pert",
                "gt_pvals",
                "gt_fdr",
                "gt_logfoldchanges",
                "prediction_pvals",
                "prediction_fdr",
                "prediction_logfoldchanges",
                "gt_significant",
                "pred_significant",
            ])

            merged_pd = merged.to_pandas()
            merged_pd["gt_significant"] = merged_pd["gt_significant"].astype(bool)
            merged_pd["pred_significant"] = merged_pd["pred_significant"].astype(bool)

            def _safe_div(a: int, b: int) -> float:
                return float(a) / float(b) if b else 0.0

            tp = int(np.logical_and(merged_pd["pred_significant"], merged_pd["gt_significant"]).sum())
            fp = int(np.logical_and(merged_pd["pred_significant"], ~merged_pd["gt_significant"]).sum())
            fn = int(np.logical_and(~merged_pd["pred_significant"], merged_pd["gt_significant"]).sum())
            tn = int(np.logical_and(~merged_pd["pred_significant"], ~merged_pd["gt_significant"]).sum())

            total = tp + fp + fn + tn
            rejections = tp + fp
            true_nulls = tn + fp

            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            fdr = _safe_div(fp, tp + fp)
            fnr = _safe_div(fn, tp + fn)
            accuracy = _safe_div(tp + tn, total)
            f1 = 0.0
            if (tp + fp) and (tp + fn):
                precision_safe = precision
                recall_safe = recall
                denom = precision_safe + recall_safe
                if denom:
                    f1 = 2 * (precision_safe * recall_safe) / denom

            overall_payload = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "m": total,
                "R": rejections,
                "m0": true_nulls,
                "Precision": precision,
                "Recall": recall,
                "FDR": fdr,
                "FNR": fnr,
                "Accuracy": accuracy,
                "F1": f1,
            }

            results.update({
                "DE_PPV": precision,
                "DE_TPR": recall,
                "DE_FDR": fdr,
                "DE_FNR": fnr,
                "DE_Accuracy": accuracy,
                "DE_F1": f1,
                "DE_TP": tp,
                "DE_FP": fp,
                "DE_FN": fn,
                "DE_TN": tn,
                "DE_m": total,
                "DE_R": rejections,
                "DE_m0": true_nulls,
            })

            records: list[dict[str, float | str | int]] = []
            for pert, subset in merged_pd.groupby("pert", sort=False):
                if pd.isna(pert):
                    continue
                subset_tp = int(np.logical_and(subset["pred_significant"], subset["gt_significant"]).sum())
                subset_fp = int(np.logical_and(subset["pred_significant"], ~subset["gt_significant"]).sum())
                subset_fn = int(np.logical_and(~subset["pred_significant"], subset["gt_significant"]).sum())
                subset_tn = int(np.logical_and(~subset["pred_significant"], ~subset["gt_significant"]).sum())
                subset_total = subset_tp + subset_fp + subset_fn + subset_tn
                subset_rejections = subset_tp + subset_fp
                subset_precision = _safe_div(subset_tp, subset_rejections)
                subset_recall = _safe_div(subset_tp, subset_tp + subset_fn)
                subset_fdr = _safe_div(subset_fp, subset_rejections)
                subset_fnr = _safe_div(subset_fn, subset_tp + subset_fn)
                subset_accuracy = _safe_div(subset_tp + subset_tn, subset_total)
                subset_f1 = 0.0
                if subset_rejections and (subset_tp + subset_fn):
                    denom = subset_precision + subset_recall
                    if denom:
                        subset_f1 = 2 * (subset_precision * subset_recall) / denom
                records.append({
                    "pert": str(pert),
                    "TP": subset_tp,
                    "FP": subset_fp,
                    "FN": subset_fn,
                    "TN": subset_tn,
                    "m": subset_total,
                    "R": subset_rejections,
                    "m0": subset_tn + subset_fp,
                    "PPV": subset_precision,
                    "TPR": subset_recall,
                    "FDR": subset_fdr,
                    "FNR": subset_fnr,
                    "Accuracy": subset_accuracy,
                    "F1": subset_f1,
                })

            if run_dir is not None:
                metrics_dir = Path(run_dir) / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)

                de_csv_path = metrics_dir / "de.csv"
                merged.write_csv(de_csv_path)
                logger.info("[vcc_eval] Saved DE CSV to %s", de_csv_path)

                overall_path = metrics_dir / "de_confusion_overall.json"
                by_pert_path = metrics_dir / "de_confusion_by_pert.csv"
                overall_path.write_text(json.dumps(overall_payload, indent=2))
                if records:
                    by_pert_df = pd.DataFrame(records)
                    by_pert_sorted = by_pert_df.sort_values(["FDR", "TPR"], ascending=[True, False])
                    by_pert_sorted.to_csv(by_pert_path, index=False)
                else:
                    pd.DataFrame(
                        columns=[
                            "pert",
                            "TP",
                            "FP",
                            "FN",
                            "TN",
                            "m",
                            "R",
                            "m0",
                            "PPV",
                            "TPR",
                            "FDR",
                            "FNR",
                            "Accuracy",
                            "F1",
                        ]
                    ).to_csv(by_pert_path, index=False)
                logger.info(
                    "[vcc_eval] Saved DE confusion summaries to %s and %s",
                    overall_path,
                    by_pert_path,
                )
        except Exception as exc:  # pragma: no cover - avoid breaking evaluation on export issues
            logger.warning("[vcc_eval] Failed to export DE CSV: %s", exc)

    if save_json and run_dir is not None:
        metrics_path = Path(run_dir) / "metrics" / "metrics.json"
        _json_dump(metrics_path, results)
        logger.info("[vcc_eval] Saved metrics to %s", metrics_path)

    return results


# ---------------------------------------------------------------------------
# CLI (optional, mirrors the original standalone behaviour)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MAE, PDS, and DES using the VCC backend",
    )
    parser.add_argument("--real", required=True, type=Path, help="Path to ground-truth AnnData (.h5ad)")
    parser.add_argument("--pred", required=True, type=Path, help="Path to predicted AnnData (.h5ad)")
    parser.add_argument("--pert-col", default="target_gene", help="Column in obs with perturbation labels")
    parser.add_argument("--control", default="non-targeting", help="Name of the control perturbation")
    parser.add_argument("--de-method", default="wilcoxon", help="Differential expression method for pdex")
    parser.add_argument("--num-threads", type=int, default=-1, help="Number of workers for pdex (-1 uses CPU count)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for pdex work splitting")
    parser.add_argument("--allow-discrete", action="store_true", help="Skip normalization even if data looks discrete")
    parser.add_argument("--real-cache", type=Path, default=None, help="Directory containing cached ground-truth artifacts")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    metrics = ["MAE", "PDS", "DES"]
    results = evaluate(
        pred_adata_or_path=args.pred,
        true_adata_or_path=args.real,
        pert_col=args.pert_col,
        control_name=args.control,
        metrics=metrics,
        run_dir=None,
        cache_path=args.real_cache,
        n_jobs=args.num_threads,
        normalize=not args.allow_discrete,
        save_json=False,
        de_method=args.de_method,
        batch_size=args.batch_size,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nDone in {time.time() - start:.2f} seconds")
