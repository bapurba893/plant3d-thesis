"""
05_preprocess_pointclouds.py

Runs the full preprocessing pipeline (outlier removal -> FPS downsample to
a fixed point count -> normalize/center) over every file in the dataset
manifest produced by 01_dataset_statistics.py, and caches each result to
disk as a .npz so this expensive step runs once per file, not on every
training run.

Cache layout:
  <out_dir>/<dataset>/<species>/<original_filename_stem>.npz
    points: (target_n, 3) float32
    labels: (target_n,) or (target_n, 2) int32   [only if source was annotated]
    n_raw, n_after_outlier_removal, n_outliers_removed, n_predecimated: scalars

Also writes <out_dir>/preprocessed_manifest.csv so downstream training code
can load cached arrays directly without re-deriving file paths.

Usage:
    python 05_preprocess_pointclouds.py \
        --manifest ../data/dataset_manifest.csv \
        --out_dir ../data/preprocessed \
        --target_n 4096

Re-run any time; already-cached files are skipped unless --overwrite is passed.
Use --limit N to smoke-test on a handful of files before committing to a full run.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_io import load_crops3d_ply, load_pheno4d_xyz
from preprocessing import preprocess_pointcloud


def cache_path_for(row, out_dir: Path) -> Path:
    species = row["species"]
    stem = Path(row["filepath"]).stem
    return out_dir / row["dataset"] / species / f"{stem}.npz"


def process_row(row, target_n, sor_k, sor_std_ratio, seed):
    filepath = row["filepath"]
    if row["dataset"] == "Crops3D":
        pts, labels = load_crops3d_ply(filepath)
    else:
        pts, labels = load_pheno4d_xyz(filepath)

    pts_out, labels_out, stats = preprocess_pointcloud(
        pts, labels, target_n=target_n,
        sor_k=sor_k, sor_std_ratio=sor_std_ratio, seed=seed,
    )
    return pts_out, labels_out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_manifest.csv",
                     help="Output of 01_dataset_statistics.py")
    ap.add_argument("--out_dir", default="data/preprocessed")
    ap.add_argument("--target_n", type=int, default=4096,
                     help="Fixed output point count. 4096 (default) preserves thin "
                          "structures (stems/leaf edges) better for segmentation and "
                          "trait geometry; use 2048 if training speed/memory is the "
                          "binding constraint.")
    ap.add_argument("--sor_k", type=int, default=20)
    ap.add_argument("--sor_std_ratio", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N rows per dataset (smoke-testing)")
    ap.add_argument("--dataset", choices=["all", "Crops3D", "Pheno4D"], default="all")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.dataset != "all":
        manifest = manifest[manifest["dataset"] == args.dataset]
    if args.limit is not None:
        manifest = manifest.groupby("dataset", group_keys=False).head(args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifest.empty:
        print("[warn] manifest is empty for the requested dataset(s) — nothing to do.")
        return

    results_rows = []
    n_processed, n_skipped, n_failed = 0, 0, 0
    t_start = time.time()

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Preprocessing"):
        cache_file = cache_path_for(row, out_dir)
        manifest_row = {
            "dataset": row["dataset"], "species": row["species"],
            "source_filepath": row["filepath"], "cache_path": str(cache_file),
            "n_points": args.target_n,
        }
        for extra in ("plant_id", "scan_date", "is_annotated"):
            if extra in row:
                manifest_row[extra] = row[extra]

        if cache_file.exists() and not args.overwrite:
            n_skipped += 1
            with np.load(cache_file) as cached:
                for stat_key in ("n_raw", "n_after_outlier_removal",
                                  "n_outliers_removed", "n_predecimated", "n_final"):
                    if stat_key in cached:
                        manifest_row[stat_key] = int(cached[stat_key])
                for stat_key in ("norm_scale", "norm_center_x", "norm_center_y", "norm_center_z"):
                    if stat_key in cached:
                        manifest_row[stat_key] = float(cached[stat_key])
            results_rows.append(manifest_row)
            continue

        try:
            pts_out, labels_out, stats = process_row(
                row, args.target_n, args.sor_k, args.sor_std_ratio, args.seed)
        except Exception as e:
            print(f"[error] failed to preprocess {row['filepath']}: {e}")
            n_failed += 1
            continue

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"points": pts_out, **stats}
        if labels_out is not None:
            save_kwargs["labels"] = labels_out
        np.savez_compressed(cache_file, **save_kwargs)

        manifest_row.update(stats)
        results_rows.append(manifest_row)
        n_processed += 1

    elapsed = time.time() - t_start
    out_manifest = pd.DataFrame(results_rows)
    out_manifest_path = out_dir / "preprocessed_manifest.csv"

    # Merge with any existing manifest rather than overwrite it — running with
    # --dataset/--limit filters must not drop previously-cached rows for
    # datasets not touched by this invocation.
    #
    # Dedup on source_filepath, NOT cache_path: cache_path is derived from
    # --out_dir, which can differ between machines/invocations (e.g. an
    # absolute path on one machine vs a repo-relative path on another) even
    # when both runs describe the exact same raw file. Deduping on
    # cache_path let a later run with a different --out_dir just append a
    # second row per file instead of replacing the first — the stale row's
    # cache_path (from a since-transferred machine) would win on lookup if
    # it happened to sort last, breaking every downstream Dataset that
    # resolves this manifest (found 2026-09-11 backfilling Crops3D labels).
    if out_manifest_path.exists():
        prior = pd.read_csv(out_manifest_path)
        combined = pd.concat([prior, out_manifest], ignore_index=True)
        out_manifest = combined.drop_duplicates(subset="source_filepath", keep="last")

    out_manifest.to_csv(out_manifest_path, index=False)

    print(f"\nDone in {elapsed:.1f}s — processed {n_processed}, "
          f"skipped (already cached) {n_skipped}, failed {n_failed}.")
    if "n_outliers_removed" in out_manifest.columns and n_processed > 0:
        processed_mask = out_manifest["n_outliers_removed"].notna()
        avg_removed_pct = (
            out_manifest.loc[processed_mask, "n_outliers_removed"]
            / out_manifest.loc[processed_mask, "n_raw"] * 100
        ).mean()
        print(f"Average outlier removal: {avg_removed_pct:.2f}% of raw points")
    print(f"Preprocessed manifest written to {out_manifest_path}")


if __name__ == "__main__":
    main()
