"""
02_visualize_pointclouds.py

Renders a side-by-side comparison of a Crops3D sample vs. a Pheno4D sample
(same species) and saves it as a PNG. This is the single image that most
convincingly demonstrates "the pipeline loads real data from both domains
and the domain gap is visually real."

Usage (raw files, if available):
    python 02_visualize_pointclouds.py \
        --crops3d_file "data/crops3d/Tomato/some_sample.ply" \
        --pheno4d_file "data/pheno4d/Tomato01/T01_0305_a.txt" \
        --out comparison_tomato.png

Usage (preprocessed .npz cache -- raw Crops3D/Pheno4D files are NOT present on the cluster, see
CLAUDE.md's Repository layout section; --crops3d_file/--pheno4d_file also accept a path ending
in .npz, in which case the cached ('points' key, already outlier-removed/decimated/normalized --
not truly "raw" -- see CLAUDE.md's Stage 0 pipeline) array is loaded directly instead of going
through load_crops3d_ply/load_pheno4d_xyz):
    python 02_visualize_pointclouds.py \
        --crops3d_file "data/preprocessed/Crops3D/Tomato/some_sample.npz" \
        --pheno4d_file "data/preprocessed/Pheno4D/Tomato/T01_0325_a.npz" \
        --out comparison_tomato.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
from data_io import load_crops3d_ply, load_pheno4d_xyz


def load_points(filepath: str, loader_fn, *loader_args) -> np.ndarray:
    """Dispatches to the cached-.npz loader when filepath ends in .npz, else the original raw
    loader (loader_fn). raw loaders (load_crops3d_ply/load_pheno4d_xyz) have differing return
    shapes (plain array vs. (array, labels) tuple) -- callers pass the right one and receive back
    just the (N, 3) points array either way."""
    if filepath.endswith(".npz"):
        return np.load(filepath)["points"]
    result = loader_fn(filepath, *loader_args)
    return result[0] if isinstance(result, tuple) else result


def subsample(pts: np.ndarray, max_points: int = 8000) -> np.ndarray:
    if len(pts) <= max_points:
        return pts
    idx = np.random.choice(len(pts), max_points, replace=False)
    return pts[idx]


def plot_cloud(ax, pts: np.ndarray, title: str, color: str):
    pts = subsample(pts)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=color, alpha=0.6)
    ax.set_title(f"{title}\n(N={len(pts)} pts shown)", fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    # equal-ish aspect so the plant isn't visually squashed
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops3d_file", required=True)
    ap.add_argument("--pheno4d_file", required=True)
    ap.add_argument("--out", default="comparison.png")
    args = ap.parse_args()

    crops3d_pts = load_points(args.crops3d_file, load_crops3d_ply)
    pheno4d_pts = load_points(args.pheno4d_file, load_pheno4d_xyz)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    plot_cloud(ax1, crops3d_pts, "Crops3D (Source)\nStructured-light / RGB-D", "tab:blue")
    plot_cloud(ax2, pheno4d_pts, "Pheno4D (Target)\nLaser triangulation scanner", "tab:orange")

    fig.suptitle("Same species, different sensor/domain — this gap is what domain adaptation must close",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved comparison figure to {args.out}")


if __name__ == "__main__":
    main()
