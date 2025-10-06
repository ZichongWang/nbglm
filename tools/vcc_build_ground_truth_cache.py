#!/usr/bin/env python3
"""Pre-compute reusable ground-truth artifacts for minimal VCC evaluation.

This script normalizes a ground-truth AnnData file, builds pseudobulk
representations for each perturbation, runs differential expression once, and
stores the results to a cache directory for later reuse.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
from pdex import parallel_differential_expression
from scipy.sparse import issparse


def guess_is_lognorm(adata: ad.AnnData, n_cells: int | float = 5e2, epsilon: float = 1e-2) -> bool:
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
    if guess_is_lognorm(adata):
        logging.info("%s already appears log-normalized; skipping conversion", label)
        return
    if allow_discrete:
        logging.info("%s contains integer counts; leaving data as-is (allow_discrete)", label)
        return
    logging.info("Normalizing and log-transforming %s", label)
    sc.pp.normalize_total(adata=adata, inplace=True)
    sc.pp.log1p(adata)


def build_pseudobulk(adata: ad.AnnData, pert_col: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = adata.X
    if issparse(matrix):
        logging.info("Converting sparse matrix to dense for pseudobulk computation")
        matrix = matrix.toarray()  # type: ignore[attr-defined]
    frame = pl.DataFrame(matrix).with_columns(
        pl.Series("groupby_key", adata.obs[pert_col].to_numpy(str))
    )
    bulked = frame.group_by("groupby_key").mean().sort("groupby_key")
    keys = bulked["groupby_key"].to_numpy()
    values = bulked.drop("groupby_key").to_numpy()
    return keys.astype(str), values


def compute_pdex(
    adata: ad.AnnData,
    control: str,
    pert_col: str,
    de_method: str,
    num_workers: int,
    batch_size: int,
) -> pl.DataFrame:
    return parallel_differential_expression(
        adata=adata,
        reference=control,
        groupby_key=pert_col,
        metric=de_method,
        num_workers=num_workers,
        batch_size=batch_size,
        as_polars=True,
    )


def sanitize_de_results(frame: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = ["fold_change", "p_value", "fdr", "log2_fold_change", "abs_log2_fold_change"]
    categorical_cols = ["target", "feature"]

    if "log2_fold_change" not in frame.columns:
        frame = frame.with_columns(
            pl.col("fold_change").log(base=2).alias("log2_fold_change").fill_nan(0.0)
        )
    if "abs_log2_fold_change" not in frame.columns:
        frame = frame.with_columns(
            pl.col("log2_fold_change").abs().alias("abs_log2_fold_change")
        )

    frame = frame.with_columns([pl.col(c).cast(pl.Float32) for c in numeric_cols if c in frame.columns])
    frame = frame.with_columns([pl.col(c).cast(pl.Categorical) for c in categorical_cols if c in frame.columns])
    keep = [c for c in numeric_cols + categorical_cols if c in frame.columns]
    return frame.select(keep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute ground-truth cache for minimal VCC evaluation",
    )
    parser.add_argument("--real", required=True, type=Path, help="Path to ground-truth AnnData (.h5ad)")
    parser.add_argument("--pert-col", default="target_gene", help="Column in obs with perturbation labels")
    parser.add_argument("--control", default="non-targeting", help="Name of the control perturbation")
    parser.add_argument("--de-method", default="wilcoxon", help="Differential expression method for pdex")
    parser.add_argument("--num-threads", type=int, default=-1, help="Number of workers for pdex (-1 uses CPU count)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for pdex work splitting")
    parser.add_argument("--allow-discrete", action="store_true", help="Skip normalization even if data looks discrete")
    parser.add_argument("--output", required=True, type=Path, help="Directory to store cache artifacts")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache directory contents")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    pl.enable_string_cache()

    output_dir: Path = args.output
    if output_dir.exists():
        if not args.overwrite and any(output_dir.iterdir()):
            raise SystemExit(
                f"Output directory {output_dir} is not empty. Use --overwrite to replace existing cache."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading ground-truth AnnData from %s", args.real)
    real = ad.read_h5ad(args.real)

    if args.pert_col not in real.obs.columns:
        raise SystemExit(f"Perturbation column '{args.pert_col}' not found in AnnData obs")

    ensure_norm_log(real, args.allow_discrete, label="Ground truth AnnData")

    genes = real.var.index.to_numpy(str)
    labels = real.obs[args.pert_col].to_numpy(str)
    unique_perts = np.unique(labels)
    if args.control not in unique_perts:
        raise SystemExit(f"Control perturbation '{args.control}' not present in obs")

    logging.info("Building pseudobulk profiles for %d perturbations", unique_perts.size)
    bulk_keys, bulk_values = build_pseudobulk(real, args.pert_col)

    cpu_count = mp.cpu_count() or 1
    num_workers = args.num_threads if args.num_threads != -1 else cpu_count
    num_workers = max(1, num_workers)

    logging.info("Running differential expression once for cache")
    with pl.StringCache():
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

    bulk_path = output_dir / "ground_truth_bulk.npz"
    np.savez(
        bulk_path,
        bulk_keys=bulk_keys,
        bulk_values=bulk_values,
        genes=genes,
    )

    de_path = output_dir / "real_de.parquet"
    real_de.write_parquet(de_path)

    metadata = {
        "pert_col": args.pert_col,
        "control": args.control,
        "bulk_file": bulk_path.name,
        "real_de_file": de_path.name,
        "de_method": args.de_method,
        "perturbations": bulk_keys.astype(str).tolist(),
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logging.info("Cache written to %s", output_dir)


if __name__ == "__main__":
    main()
