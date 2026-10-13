"""
07_preprocess_variants.py

Block PP, Stage 2 step 1: CLI driver for the alternative preprocessing
pipelines in scripts/preprocessing_variants.py. Mirrors
05_preprocess_pointclouds.py's structure (manifest in, .npz cache + CSV
manifest out, skip-if-cached, --limit for smoke tests) but writes to its own
per-pipeline directory and NEVER touches the existing data/preprocessed/
cache -- refuses to run if --out_dir would collide with it.

Usage:
    python scripts/07_preprocess_variants.py --pipeline P4 --limit 5
    python scripts/07_preprocess_variants.py --pipeline P1
    python scripts/07_preprocess_variants.py --remove_soil true    # P4 via preprocess_cloud
    python scripts/07_preprocess_variants.py --remove_soil false   # P4a via preprocess_cloud

--remove_soil (instead of --pipeline) runs the single switchable function
scripts/preprocessing_pipeline.py::preprocess_cloud; its default --out_dir is
data/preprocessed_variants/single_remove_soil_<true|false>/, so it never
writes into the existing P4/P4a caches unless --out_dir points there.

Writes:
    <out_dir>/<dataset>/<species>/<stem>.npz   (points [+labels] + per-step stats)
    <out_dir>/preprocessed_manifest.csv        (same columns as
        data/preprocessed/preprocessed_manifest.csv, plus: pipeline_id,
        n_after_soil_removal, n_soil_pot_removed, voxel_size_mm,
        voxel_size_basis, sampling_method, up_axis_aligned, norm_kind)

Run from the repo root (matches scripts/08_soil_removal_check.py's
convention); --out_dir defaults to data/preprocessed_variants/<pipeline>/.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from data_io import load_crops3d_ply, load_pheno4d_xyz  # noqa: E402
from preprocessing_variants import (  # noqa: E402
    PIPELINES, DEFAULT_TARGET_N, DEFAULT_SEED, DEFAULT_SOR_K, DEFAULT_SOR_STD_RATIO,
)
from preprocessing_pipeline import preprocess_cloud  # noqa: E402

EXISTING_CACHE_DIR = (_REPO_ROOT / "data" / "preprocessed").resolve()

# Scalar stat keys that may appear in a pipeline's `info` dict (after
# `labels_out` is popped out) -- used both for the resume-from-cache path
# (reading them back out of an existing .npz) and for deciding which keys
# are safe to drop straight into a manifest CSV row.
_INT_STAT_KEYS = ("n_raw", "n_after_outlier_removal", "n_outliers_removed",
                   "n_after_soil_removal", "n_soil_pot_removed",
                   "n_predecimated", "n_final")
_FLOAT_STAT_KEYS = ("norm_scale", "norm_center_x", "norm_center_y", "norm_center_z",
                     "voxel_size_mm")
_BOOL_STAT_KEYS = ("up_axis_aligned", "remove_soil")
_STR_STAT_KEYS = ("voxel_size_basis", "sampling_method", "pipeline_id", "norm_kind")


def cache_path_for(row, out_dir: Path) -> Path:
    species = row["species"]
    stem = Path(row["filepath"]).stem
    return out_dir / row["dataset"] / species / f"{stem}.npz"


def to_portable_source_filepath(raw_filepath: str, dataset: str) -> str:
    """
    Converts an absolute raw-data path (as stored in data/dataset_manifest.csv,
    which was built against this laptop's absolute raw-data root) into the
    "../data/<dataset>/..." relative form that the split CSVs (data/
    crops3d_train.csv, data/pheno4d_heldout_eval.csv, etc. -- built earlier, by a
    different manifest-scanning run, using repo-relative paths) use in their own
    `filepath` column.

    This match matters independently of the cluster: adapters/dataset.py's
    _load_manifest_lookup() keys its lookup dict on
    _strip_leading_dotdot(source_filepath), and PlantClsSegDataset/
    PlantSpeciesDataset key their own samples on _strip_leading_dotdot(split_csv's
    filepath) -- if the two don't produce identical strings, EVERY row silently
    fails the `manifest_lookup.get(key)` lookup and the dataset ends up empty, on
    any machine, not just the cluster (confirmed: 0/263 matched before this fix).
    """
    marker = f"/{dataset.lower()}/"
    normalized = raw_filepath.replace("\\", "/")
    idx = normalized.find(marker)
    if idx == -1:
        raise ValueError(f"Could not find {marker!r} in {raw_filepath!r} -- cannot "
                          f"build a portable source_filepath for this row")
    return "../data" + normalized[idx:]


def to_portable_cache_path(cache_file: Path) -> str:
    """
    Converts an absolute cache .npz path into a path relative to the repo root
    (styled like the original data/preprocessed/preprocessed_manifest.csv's own
    "../data/preprocessed/..." cache_path values), instead of the fully absolute,
    machine-specific path this script used to store.

    Required for adapters/dataset.py's `_REPO_ROOT / _strip_leading_dotdot(cache_path)`
    resolution to work on ANY machine: pathlib's `/` operator DISCARDS the left
    operand entirely when the right operand is already an absolute path, so an
    absolute cache_path silently resolves to itself (the original laptop path) even
    when _REPO_ROOT is a different machine's repo checkout -- it never raises, it
    just points at a path that doesn't exist there, and PlantClsSegDataset treats a
    missing cache_path as "skip this row", not an error.
    """
    rel = Path(cache_file).resolve().relative_to(_REPO_ROOT)
    return "../" + str(rel).replace("\\", "/")


def write_manifest(results_rows, out_manifest_path: Path):
    """
    Writes/updates the manifest CSV from `results_rows` (this run's rows so
    far), merged with and deduplicated against whatever is already on disk
    at `out_manifest_path` -- same merge-not-overwrite + dedup-on-
    source_filepath convention as 05_preprocess_pointclouds.py.

    Called after EVERY row (not just once at the end of the whole run) so
    that killing the process at any point (OOM, Ctrl-C, closed terminal
    without nohup, power loss) leaves a manifest that is missing at most the
    one row in flight -- not "no manifest at all" despite hundreds of .npz
    files already sitting on disk. Writes to a temp file and renames into
    place (os.replace is atomic on POSIX) so a kill mid-write can't leave a
    half-written/corrupt CSV behind either. CSV writes at this scale (~500
    rows) take low-single-digit milliseconds, so doing this every row (not
    just every N) has no measurable effect on the real per-cloud runtime,
    which is dominated by outlier removal/voxelling/FPS on raw point clouds.
    """
    out_manifest = pd.DataFrame(results_rows)
    if out_manifest_path.exists():
        prior = pd.read_csv(out_manifest_path)
        combined = pd.concat([prior, out_manifest], ignore_index=True)
        out_manifest = combined.drop_duplicates(subset="source_filepath", keep="last")
    tmp_path = out_manifest_path.with_suffix(".csv.tmp")
    out_manifest.to_csv(tmp_path, index=False)
    tmp_path.replace(out_manifest_path)
    return out_manifest


def parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def main():
    ap = argparse.ArgumentParser()
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--pipeline", choices=sorted(PIPELINES.keys()))
    which.add_argument("--remove_soil", type=parse_bool, default=None,
                       help="true/false: run preprocessing_pipeline.preprocess_cloud "
                            "(true = P4, soil/pot removed; false = P4a, soil/pot kept) "
                            "instead of a --pipeline")
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "data" / "dataset_manifest.csv"))
    ap.add_argument("--out_dir", default=None,
                     help="Default: data/preprocessed_variants/<pipeline>/")
    ap.add_argument("--target_n", type=int, default=DEFAULT_TARGET_N)
    ap.add_argument("--sor_k", type=int, default=DEFAULT_SOR_K)
    ap.add_argument("--sor_std_ratio", type=float, default=DEFAULT_SOR_STD_RATIO)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N rows per (dataset, species) group "
                          "(smoke-testing)")
    ap.add_argument("--dataset", choices=["all", "Crops3D", "Pheno4D"], default="all")
    ap.add_argument("--shared_cache_dir",
                     default=str(_REPO_ROOT / "data" / "preprocessed_variants" / "_shared_cache"),
                     help="Caches the outlier-removal + soil/pot-removal result (identical "
                          "across all 8 pipelines at default --sor_k/--sor_std_ratio; profiling "
                          "found outlier removal alone is ~60-70%% of a single pipeline's "
                          "per-cloud time, soil removal <1%%) once per raw cloud and reuses it "
                          "across pipeline runs/processes -- safe to point multiple concurrently "
                          "running pipelines (e.g. different --pipeline values) at the SAME "
                          "shared_cache_dir; see preprocessing_variants.py::"
                          "outlier_and_soil_removed(). Pass 'none' to disable.")
    args = ap.parse_args()

    shared_cache_dir = None if args.shared_cache_dir.strip().lower() == "none" else args.shared_cache_dir

    if args.remove_soil is not None:
        run_name = f"single_remove_soil_{str(args.remove_soil).lower()}"
        remove_soil = args.remove_soil

        def pipeline_fn(pts, **kw):
            return preprocess_cloud(pts, remove_soil=remove_soil, **kw)
    else:
        run_name = args.pipeline
        pipeline_fn = PIPELINES[args.pipeline]

    out_dir = (Path(args.out_dir) if args.out_dir
               else (_REPO_ROOT / "data" / "preprocessed_variants" / run_name))
    out_dir = out_dir.resolve()

    # Hard safety check: the plan requires data/preprocessed/ to never be
    # touched by this script. Refuse outright rather than relying on the
    # caller to pass a sane --out_dir.
    if (out_dir == EXISTING_CACHE_DIR
            or EXISTING_CACHE_DIR in out_dir.parents
            or out_dir in EXISTING_CACHE_DIR.parents):
        raise SystemExit(f"[fatal] --out_dir resolves to {out_dir}, which collides with the "
                          f"existing cache at {EXISTING_CACHE_DIR} -- refusing to run.")

    manifest = pd.read_csv(args.manifest)
    if args.dataset != "all":
        manifest = manifest[manifest["dataset"] == args.dataset]
    if args.limit is not None:
        manifest = manifest.groupby(["dataset", "species"], group_keys=False).head(args.limit)
    if manifest.empty:
        print("[warn] manifest is empty for the requested filter(s) -- nothing to do.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_path = out_dir / "preprocessed_manifest.csv"

    results_rows = []
    n_processed, n_skipped, n_failed = 0, 0, 0
    per_file_times = []
    t_start = time.time()

    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                        desc=f"Preprocessing [{run_name}]"):
        cache_file = cache_path_for(row, out_dir)
        manifest_row = {
            "dataset": row["dataset"], "species": row["species"],
            "source_filepath": to_portable_source_filepath(row["filepath"], row["dataset"]),
            "cache_path": to_portable_cache_path(cache_file),
            "n_points": args.target_n,
        }
        for extra in ("plant_id", "scan_date", "is_annotated"):
            if extra in row:
                manifest_row[extra] = row[extra]

        if cache_file.exists() and not args.overwrite:
            n_skipped += 1
            with np.load(cache_file, allow_pickle=False) as cached:
                for k in _INT_STAT_KEYS:
                    if k in cached:
                        manifest_row[k] = int(cached[k])
                for k in _FLOAT_STAT_KEYS:
                    if k in cached:
                        manifest_row[k] = float(cached[k])
                for k in _BOOL_STAT_KEYS:
                    if k in cached:
                        manifest_row[k] = bool(cached[k])
                for k in _STR_STAT_KEYS:
                    if k in cached:
                        manifest_row[k] = str(cached[k])
            results_rows.append(manifest_row)
            write_manifest(results_rows, out_manifest_path)
            continue

        t0 = time.time()
        try:
            if row["dataset"] == "Crops3D":
                pts, labels = load_crops3d_ply(row["filepath"])
            else:
                pts, labels = load_pheno4d_xyz(row["filepath"])

            pts_out, center, scale, info = pipeline_fn(
                pts, labels=labels, dataset=row["dataset"], species=row["species"],
                target_n=args.target_n, sor_k=args.sor_k, sor_std_ratio=args.sor_std_ratio,
                seed=args.seed, shared_cache_dir=shared_cache_dir,
                stem=Path(row["filepath"]).stem)
        except Exception as e:
            print(f"[error] failed to preprocess ({run_name}) {row['filepath']}: {e}")
            n_failed += 1
            continue
        dt = time.time() - t0
        per_file_times.append(dt)

        labels_out = info.pop("labels_out", None)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "points": pts_out,
            "norm_scale": scale,
            "norm_center_x": float(center[0]),
            "norm_center_y": float(center[1]),
            "norm_center_z": float(center[2]),
        }
        for k, v in info.items():
            if v is not None:
                save_kwargs[k] = v
        if labels_out is not None:
            save_kwargs["labels"] = labels_out
        np.savez_compressed(cache_file, **save_kwargs)

        manifest_row.update({k: v for k, v in info.items() if v is not None})
        manifest_row["norm_scale"] = scale
        manifest_row["norm_center_x"] = float(center[0])
        manifest_row["norm_center_y"] = float(center[1])
        manifest_row["norm_center_z"] = float(center[2])
        results_rows.append(manifest_row)
        write_manifest(results_rows, out_manifest_path)
        n_processed += 1

    elapsed = time.time() - t_start

    print(f"\n[{run_name}] Done in {elapsed:.1f}s -- processed {n_processed}, "
          f"skipped (already cached) {n_skipped}, failed {n_failed}.")
    if per_file_times:
        arr = np.array(per_file_times)
        print(f"  per-cloud time: mean={arr.mean():.3f}s  median={np.median(arr):.3f}s  "
              f"max={arr.max():.3f}s  (n={len(arr)} freshly processed)")
    print(f"Manifest written to {out_manifest_path}")


if __name__ == "__main__":
    main()
