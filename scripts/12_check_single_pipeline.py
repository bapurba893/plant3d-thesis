"""
12_check_single_pipeline.py

Block PP: checks that preprocessing_pipeline.preprocess_cloud reproduces the
existing caches exactly --
    remove_soil=False  vs  data/preprocessed_variants/P4a/
    remove_soil=True   vs  data/preprocessed_variants/P4/
for every cloud in data/dataset_manifest.csv (531).

For each cloud and setting it reruns the function on the RAW cloud and
compares against the cached .npz: points with np.allclose (default
tolerances), labels with np.array_equal, and norm_scale/norm_center with
np.allclose. Needs the raw Crops3D PLY / Pheno4D txt files, which live on
the laptop, not the cluster (see docs/00_README_download.md).

Usage (repo root):
    python scripts/12_check_single_pipeline.py
    python scripts/12_check_single_pipeline.py --limit 3          # smoke test
    python scripts/12_check_single_pipeline.py --shared_cache_dir data/preprocessed_variants/_shared_cache

--shared_cache_dir defaults to 'none' (outlier removal recomputed from raw,
the stricter check); pointing it at the existing shared cache is faster but
then reuses the cached outlier-removal result instead of re-testing it.

Writes results/pp_checks/single_pipeline_check.csv (one row per cloud and
setting) and prints the match counts and every mismatch with its maximum
absolute point difference. Exit code 0 only if every cloud matches in both
settings.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from data_io import load_crops3d_ply, load_pheno4d_xyz  # noqa: E402
from preprocessing_pipeline import preprocess_cloud  # noqa: E402

SETTINGS = ((False, "P4a"), (True, "P4"))


def compare(pts_out, center, scale, labels_out, cache_file: Path):
    with np.load(cache_file, allow_pickle=False) as d:
        ref_pts = d["points"]
        ref_labels = d["labels"] if "labels" in d else None
        ref_center = np.array([float(d["norm_center_x"]), float(d["norm_center_y"]),
                               float(d["norm_center_z"])])
        ref_scale = float(d["norm_scale"])

    if ref_pts.shape != pts_out.shape:
        return dict(match=False, max_abs_diff=np.nan, labels_equal=False,
                    norm_match=False, note=f"shape {pts_out.shape} vs cached {ref_pts.shape}")
    max_diff = float(np.max(np.abs(pts_out.astype(np.float64) - ref_pts.astype(np.float64))))
    pts_ok = bool(np.allclose(pts_out, ref_pts))
    if ref_labels is None and labels_out is None:
        labels_ok = True
    elif ref_labels is None or labels_out is None:
        labels_ok = False
    else:
        labels_ok = bool(np.array_equal(labels_out, ref_labels))
    norm_ok = bool(np.allclose(center, ref_center) and np.isclose(scale, ref_scale))
    return dict(match=pts_ok and labels_ok and norm_ok, max_abs_diff=max_diff,
                labels_equal=labels_ok, norm_match=norm_ok, note="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "data" / "dataset_manifest.csv"))
    ap.add_argument("--variants_dir", default=str(_REPO_ROOT / "data" / "preprocessed_variants"))
    ap.add_argument("--shared_cache_dir", default="none")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only the first N rows per (dataset, species) group")
    ap.add_argument("--out_csv",
                    default=str(_REPO_ROOT / "results" / "pp_checks" / "single_pipeline_check.csv"))
    args = ap.parse_args()

    shared_cache_dir = (None if args.shared_cache_dir.strip().lower() == "none"
                        else args.shared_cache_dir)
    variants_dir = Path(args.variants_dir)

    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.groupby(["dataset", "species"], group_keys=False).head(args.limit)

    rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="check"):
        stem = Path(row["filepath"]).stem
        base = dict(dataset=row["dataset"], species=row["species"], stem=stem)
        try:
            if row["dataset"] == "Crops3D":
                pts, labels = load_crops3d_ply(row["filepath"])
            else:
                pts, labels = load_pheno4d_xyz(row["filepath"])
        except Exception as e:  # missing raw file etc. -- counts as not checked
            for remove_soil, ref in SETTINGS:
                rows.append(dict(base, remove_soil=remove_soil, ref_pipeline=ref, match=False,
                                 max_abs_diff=np.nan, labels_equal=False, norm_match=False,
                                 note=f"raw load failed: {e}"))
            continue

        for remove_soil, ref in SETTINGS:
            cache_file = variants_dir / ref / row["dataset"] / row["species"] / f"{stem}.npz"
            if not cache_file.exists():
                rows.append(dict(base, remove_soil=remove_soil, ref_pipeline=ref, match=False,
                                 max_abs_diff=np.nan, labels_equal=False, norm_match=False,
                                 note=f"no cached file {cache_file}"))
                continue
            pts_out, center, scale, info = preprocess_cloud(
                pts, labels=labels, dataset=row["dataset"], species=row["species"],
                remove_soil=remove_soil, shared_cache_dir=shared_cache_dir, stem=stem)
            res = compare(pts_out, center, scale, info["labels_out"], cache_file)
            rows.append(dict(base, remove_soil=remove_soil, ref_pipeline=ref, **res))

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    all_ok = True
    for remove_soil, ref in SETTINGS:
        sub = df[df["remove_soil"] == remove_soil]
        unchecked = sub["note"].str.startswith(("raw load failed", "no cached file"))
        n_ok = int(sub["match"].sum())
        print(f"remove_soil={remove_soil} vs {ref}: {n_ok}/{len(sub)} match, "
              f"{int((~sub['match'] & ~unchecked).sum())} mismatch, "
              f"{int(unchecked.sum())} not checked")
        for _, b in sub[unchecked].iterrows():
            all_ok = False
            print(f"  NOT CHECKED {b['dataset']}/{b['species']}/{b['stem']}: {b['note']}")
        for _, b in sub[~sub["match"] & ~unchecked].iterrows():
            all_ok = False
            print(f"  MISMATCH {b['dataset']}/{b['species']}/{b['stem']}: "
                  f"max_abs_diff={b['max_abs_diff']:.3g} labels_equal={b['labels_equal']} "
                  f"norm_match={b['norm_match']} {b['note']}")
    print(f"written {out_csv}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
