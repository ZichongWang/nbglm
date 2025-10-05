#!/usr/bin/env python3
"""Standalone VCC metric computation without importing cell_eval.

Supports optional use of a ground-truth cache produced by
`scripts/build_ground_truth_cache.py` to avoid recomputing differential
expression and pseudobulk profiles.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, Optional

import time
import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
from pdex import parallel_differential_expression
from scipy.sparse import issparse
from sklearn.metrics import mean_absolute_error, pairwise_distances


# ---------------------------------------------------------------------------
# Normalization helpers (mirrors cell_eval.utils.guess_is_lognorm + converter)
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
        logging.info("%s already appears log-normalized; skipping conversion", label)
        return
    if allow_discrete:
        logging.info("%s contains integer counts; leaving data as-is (allow_discrete)", label)
        return
    logging.info("Normalizing and log-transforming %s", label)
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
            logging.info("Converting sparse matrix to dense for pseudobulk computation")
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

def compute_mae(pair: PerturbationPair) -> dict[str, float]:
    scores: dict[str, float] = {}
    for bulk in pair.iter_bulk_arrays():
        value = mean_absolute_error(bulk.pert_pred, bulk.pert_real)
        scores[bulk.key] = float(value)
    return scores


def compute_discrimination_score_l1(pair: PerturbationPair) -> dict[str, float]:
    real_effects = []
    pred_effects = []
    for bulk in pair.iter_bulk_arrays():
        real_effects.append(bulk.perturbation_effect("real"))
        pred_effects.append(bulk.perturbation_effect("pred"))
    real_effects = np.vstack(real_effects)
    pred_effects = np.vstack(pred_effects)

    norm_ranks: dict[str, float] = {}
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

def compute_pdex(adata: ad.AnnData, control: str, pert_col: str, de_method: str, num_workers: int, batch_size: int) -> pl.DataFrame:
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
) -> dict[str, float]:
    real_rank = build_rank_matrix(real_df, perts, fdr_threshold=fdr_threshold)
    pred_rank = build_rank_matrix(pred_df, perts, fdr_threshold=fdr_threshold)

    if real_rank.height == 0 or pred_rank.height == 0:
        return {str(pert): 0.0 for pert in perts}

    overlaps: dict[str, float] = {}
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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mae, discrimination_score_l1, and overlap_at_N without importing cell_eval",
    )
    parser.add_argument("--real", required=True, type=Path, help="Path to ground-truth AnnData (.h5ad)")
    parser.add_argument("--pred", required=True, type=Path, help="Path to predicted AnnData (.h5ad)")
    parser.add_argument("--pert-col", default="target_gene", help="Column in obs with perturbation labels")
    parser.add_argument("--control", default="non-targeting", help="Name of the control perturbation")
    parser.add_argument("--de-method", default="wilcoxon", help="Differential expression method for pdex")
    parser.add_argument("--num-threads", type=int, default=-1, help="Number of workers for pdex (-1 uses CPU count)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for pdex work splitting")
    parser.add_argument("--allow-discrete", action="store_true", help="Skip normalization even if data looks discrete")
    parser.add_argument("--real-cache", type=Path, default="/home/wzc26/work/vcc/nbglm/data/test_de_cache", help="Directory containing cached ground-truth artifacts")
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
                level=logging.INFO,
                format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                force=True 
            )
    pl.enable_string_cache()
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_width_chars(0)

    cache: Optional[dict[str, object]] = None
    metadata: dict[str, object] | None = None
    if args.real_cache:
        cache_dir = args.real_cache
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")
        cache = load_real_cache(cache_dir)
        metadata = cache["metadata"]  # type: ignore[index]
        if metadata.get("pert_col") != args.pert_col:
            raise ValueError(
                "Cache pert_col mismatch: cache uses "
                f"'{metadata.get('pert_col')}', requested '{args.pert_col}'"
            )
        if metadata.get("control") != args.control:
            raise ValueError(
                "Cache control mismatch: cache uses "
                f"'{metadata.get('control')}', requested '{args.control}'"
            )

    logging.info("Loading ground-truth AnnData from %s", args.real)
    real = ad.read_h5ad(args.real)
    logging.info("Loading predicted AnnData from %s", args.pred)
    pred = ad.read_h5ad(args.pred)

    ensure_norm_log(real, args.allow_discrete, label="Real AnnData")
    ensure_norm_log(pred, args.allow_discrete, label="Predicted AnnData")

    pred_labels = pred.obs[args.pert_col].to_numpy(str)
    pred_perts = np.unique(pred_labels)
    if args.control not in pred_perts:
        raise ValueError("Control perturbation missing from predicted AnnData")

    real_labels_full = real.obs[args.pert_col].to_numpy(str)
    real_perts_full = set(np.unique(real_labels_full))
    if args.control not in real_perts_full:
        raise ValueError("Control perturbation missing from real AnnData")
    missing_in_real = sorted(set(pred_perts) - real_perts_full)
    if missing_in_real:
        raise ValueError(
            "Real AnnData is missing perturbations present in prediction: "
            + ", ".join(missing_in_real)
        )

    required_all = np.union1d(pred_perts, np.array([args.control]))
    real_mask = np.isin(real_labels_full, required_all)
    if real_mask.sum() == 0:
        raise ValueError("No overlapping perturbations between real and predicted AnnData")
    real = real[real_mask].copy()

    cached_bulk = None
    if cache is not None:
        assert metadata is not None
        cached_genes = cache["genes"]  # type: ignore[index]
        if not np.array_equal(cached_genes, real.var.index.to_numpy(str)):
            raise ValueError("Gene order in cache does not match real AnnData")
        bulk_keys = np.asarray(cache["bulk_keys"], dtype=str)  # type: ignore[index]
        bulk_values = np.asarray(cache["bulk_values"], dtype=float)  # type: ignore[index]
        required_all_list = [str(p) for p in required_all]
        metadata_perts = {str(p) for p in metadata.get("perturbations", [])}
        if metadata_perts and not set(required_all_list).issubset(metadata_perts):
            missing_meta = sorted(set(required_all_list) - metadata_perts)
            raise ValueError(
                "Cache metadata missing required perturbations: " + ", ".join(missing_meta)
            )
        key_to_idx = {k: i for i, k in enumerate(bulk_keys)}
        missing = [p for p in required_all_list if p not in key_to_idx]
        if missing:
            raise ValueError(
                "Cached bulk missing required perturbations: " + ", ".join(missing)
            )
        order_idx = [key_to_idx[p] for p in required_all_list]
        cached_bulk = (
            bulk_keys[order_idx],
            bulk_values[order_idx],
        )

    pair = PerturbationPair(
        real=real,
        pred=pred,
        pert_col=args.pert_col,
        control_pert=args.control,
        cached_real_bulk=cached_bulk,  # type: ignore[arg-type]
    )

    mae_scores = compute_mae(pair)
    logging.info(f"Mean MAE: {np.mean(list(mae_scores.values())):.4f}")
    discrimination_scores = compute_discrimination_score_l1(pair)
    logging.info(f"Mean Discrimination Score L1: {np.mean(list(discrimination_scores.values())):.4f}")

    cpu_count = mp.cpu_count() or 1
    num_workers = args.num_threads if args.num_threads != -1 else cpu_count
    num_workers = max(1, num_workers)

    with pl.StringCache():
        if cache is not None:
            real_de_path = cache["real_de_path"]  # type: ignore[index]
            real_de = pl.read_parquet(real_de_path)  # type: ignore[arg-type]
            real_de = sanitize_de_results(real_de)
        else:
            real_de = sanitize_de_results(
                compute_pdex(
                    real,
                    control=args.control,
                    pert_col=args.pert_col,
                    de_method=args.de_method,
                    num_workers=num_workers,
                    batch_size=args.batch_size,
                )
            )
        required_targets = {str(p) for p in pair.perts}
        required_targets.add(args.control)
        real_de = real_de.filter(pl.col("target").is_in(list(required_targets)))
        pred_de = sanitize_de_results(
            compute_pdex(
                pred,
                control=args.control,
                pert_col=args.pert_col,
                de_method=args.de_method,
                num_workers=num_workers,
                batch_size=args.batch_size,
            )
        )
        pred_de = pred_de.filter(pl.col("target").is_in(list(required_targets)))

    overlap_scores = compute_overlap_at_n(real_de, pred_de, pair.perts, k=None, metric="overlap", fdr_threshold=None)
    logging.info(f"Overlap at N: {np.mean(list(overlap_scores.values())):.4f}")

    # rows = []
    # for pert in pair.perts:
    #     key = str(pert)
    #     rows.append(
    #         {
    #             "perturbation": key,
    #             "mae": mae_scores.get(key, float("nan")),
    #             "discrimination_score_l1": discrimination_scores.get(key, float("nan")),
    #             "overlap_at_N": overlap_scores.get(key, float("nan")),
    #         }
    #     )
    # results = pl.DataFrame(rows).sort("perturbation")
    # agg_results = results.drop("perturbation").describe()

    # print("Per-perturbation metrics:")
    # print(results)
    # print("\nAggregate summary:")
    # print(agg_results)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nDone in {time.time() - start:.2f} seconds")
