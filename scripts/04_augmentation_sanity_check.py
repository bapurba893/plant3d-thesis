"""
04_augmentation_sanity_check.py

Loads ONE sample point cloud and renders a grid showing the effect of
each augmentation individually, plus the full composed pipeline —
proving the augmentation code runs and doesn't destroy the plant shape.

Usage (Pheno4D example):
    python 04_augmentation_sanity_check.py \
        --file "data/pheno4d/Tomato01/T01_0305_a.txt" --source pheno4d \
        --out augmentation_check.png

Usage (Crops3D example):
    python 04_augmentation_sanity_check.py \
        --file "data/crops3d/Tomato/sample.ply" --source crops3d \
        --out augmentation_check.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from data_io import load_crops3d_ply, load_pheno4d_xyz
import augmentations as aug


def subsample(pts, max_points=6000):
    if len(pts) <= max_points:
        return pts
    idx = np.random.choice(len(pts), max_points, replace=False)
    return pts[idx]


def plot_one(ax, pts, title):
    pts = subsample(pts)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="tab:green", alpha=0.6)
    ax.set_title(f"{title}\n(N={len(pts)})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--source", choices=["crops3d", "pheno4d"], required=True)
    ap.add_argument("--out", default="augmentation_check.png")
    args = ap.parse_args()

    if args.source == "crops3d":
        pts = load_crops3d_ply(args.file)
    else:
        pts, _ = load_pheno4d_xyz(args.file)

    rng = np.random.default_rng(0)
    original = aug.normalize(pts)

    variants = [
        ("Original (normalized)", original),
        ("Random Rotation (Z)", aug.random_rotation_z(original, rng)),
        ("Gaussian Noise", aug.gaussian_noise(original, rng=rng)),
        ("Random Scale", aug.random_scale(original, rng=rng)),
        ("Random Flip (X)", aug.random_flip(original, axis="x", rng=rng)),
        ("RandomCrop3D (0.85)", aug.random_crop3d(original, keep_frac=0.85, rng=rng)),
        ("CoarseDropout3D", aug.coarse_dropout3d(original, n_holes=3, rng=rng)),
        ("Cubic Symmetry", aug.cubic_symmetry(original, rng=rng)),
        ("Full Composed Pipeline", aug.compose_pipeline(pts, rng=rng)),
    ]

    n = len(variants)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, (title, v) in enumerate(variants):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        plot_one(ax, v, title)

    fig.suptitle("Augmentation Sanity Check — plant shape should stay recognizable\n"
                 f"Source file: {args.file}", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved to {args.out}")
    print("\nManually check: does every panel still look like a plant?")
    print("If CoarseDropout3D or RandomCrop3D deleted most of the points, reduce")
    print("hole_frac / increase keep_frac in augmentations.py before using in training.")


if __name__ == "__main__":
    main()
