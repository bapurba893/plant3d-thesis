"""
06_before_after_preprocessing.py

Renders a genuine before/after preprocessing comparison for one real Crops3D sample and one real
Pheno4D sample (same species) -- loads each RAW file with no processing applied, then runs it
through the actual `preprocessing.py::preprocess_pointcloud` pipeline (the same function/defaults
`05_preprocess_pointclouds.py` uses to build the committed .npz cache: outlier removal -> voxel
pre-decimate + farthest-point sampling to target_n -> normalize/center), and plots both stages
side by side for each domain.

Requires RAW files -- unlike 02_visualize_pointclouds.py/04_augmentation_sanity_check.py (which
gained .npz-cache fallback support since raw files aren't normally present on this cluster, see
their own docstrings), a genuine "before" state cannot be reconstructed from the .npz cache at
all: each cached file stores only the FINAL processed `points` array plus scalar counts
(n_raw, n_after_outlier_removal, etc.), never an intermediate point array. This script is only
usable when raw files have been manually transferred onto the machine.

Usage:
    python 06_before_after_preprocessing.py \
        --crops3d_file "../data/11-2_20-28.ply" \
        --pheno4d_file "../data/T01_0325_a.txt" \
        --out "../data/before_after_tomato.png"
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
from data_io import load_crops3d_ply, load_pheno4d_xyz
from preprocessing import preprocess_pointcloud

# Matches 05_preprocess_pointclouds.py's own CLI defaults exactly -- this script reproduces the
# REAL pipeline that built the committed .npz cache, not an independently-chosen parameterization.
TARGET_N = 4096
SOR_K = 20
SOR_STD_RATIO = 2.0
SEED = 42


def subsample(pts: np.ndarray, max_points: int = 8000) -> np.ndarray:
    """Display-only subsampling for the "before" (raw, millions-of-points) panels -- NOT part of
    the real pipeline, purely so matplotlib's 3D scatter stays responsive. The "after" panels
    already have exactly TARGET_N points from the real pipeline, no subsampling needed."""
    if len(pts) <= max_points:
        return pts
    idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
    return pts[idx]


def plot_cloud(ax, pts: np.ndarray, title: str, color: str, max_points: int = 8000):
    pts = subsample(pts, max_points)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=color, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops3d_file", required=True, help="RAW .ply file")
    ap.add_argument("--pheno4d_file", required=True, help="RAW .txt file")
    ap.add_argument("--out", default="before_after.png")
    args = ap.parse_args()

    print(f"Loading raw Crops3D file: {args.crops3d_file}")
    c3d_raw, c3d_labels = load_crops3d_ply(args.crops3d_file)
    print(f"  {len(c3d_raw)} raw points")
    c3d_after, _, c3d_stats = preprocess_pointcloud(
        c3d_raw, c3d_labels, target_n=TARGET_N, sor_k=SOR_K, sor_std_ratio=SOR_STD_RATIO, seed=SEED)
    print(f"  after pipeline: {c3d_stats}")

    print(f"Loading raw Pheno4D file: {args.pheno4d_file}")
    p4d_raw, p4d_labels = load_pheno4d_xyz(args.pheno4d_file)
    print(f"  {len(p4d_raw)} raw points")
    p4d_after, _, p4d_stats = preprocess_pointcloud(
        p4d_raw, p4d_labels, target_n=TARGET_N, sor_k=SOR_K, sor_std_ratio=SOR_STD_RATIO, seed=SEED)
    print(f"  after pipeline: {p4d_stats}")

    fig = plt.figure(figsize=(14, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_cloud(ax1, c3d_raw, f"Crops3D — BEFORE (raw)\nN={c3d_stats['n_raw']:,} points "
                             f"({min(len(c3d_raw), 8000):,} shown)", "tab:blue")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_cloud(ax2, c3d_after, f"Crops3D — AFTER (outlier removal + decimation + normalize)\n"
                               f"N={c3d_stats['n_final']:,} points "
                               f"({c3d_stats['n_outliers_removed']:,} outliers removed)", "tab:blue")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    plot_cloud(ax3, p4d_raw, f"Pheno4D — BEFORE (raw)\nN={p4d_stats['n_raw']:,} points "
                             f"({min(len(p4d_raw), 8000):,} shown)", "tab:orange")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    plot_cloud(ax4, p4d_after, f"Pheno4D — AFTER (outlier removal + decimation + normalize)\n"
                               f"N={p4d_stats['n_final']:,} points "
                               f"({p4d_stats['n_outliers_removed']:,} outliers removed)", "tab:orange")

    fig.suptitle("Real preprocessing pipeline, before vs. after — "
                  f"outlier removal (k={SOR_K}, std_ratio={SOR_STD_RATIO}) + FPS to {TARGET_N} + normalize",
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
