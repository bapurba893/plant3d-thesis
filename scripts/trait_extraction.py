"""
trait_extraction.py

D0 (trait-extraction pipeline, Block D prerequisite): extracts 5 geometric
traits (height, stem diameter, leaf area, leaf count, volume) per scan
from the organ-segmented Pheno4D point clouds written by
`adapters/infer_segmentation.py --mode full` (`data/segmented/Pheno4D/
<species>/<stem>.npz`, holding `{"points": normalized_pts, "pred_organ":
(N,) int in {0: soil, 1: stem, 2: leaf}}`).

**Hard requirement: real-world units.** Every cached Pheno4D scan is
unit-sphere-normalized per-scan independently (`scripts/preprocessing.py`
::normalize_and_center); two scans of the same plant at different growth
stages get rescaled to the same unit sphere, so trait values computed in
normalized space carry NO growth signal at all. `norm_scale`/
`norm_center_x/y/z` (persisted in `data/preprocessed/preprocessed_
manifest.csv`, added specifically for this pipeline -- see
`scripts/preprocessing.py`'s change and step_notes/
D0_Trait_Extraction_Pipeline.md) are required to invert back to real
units before any trait is computed: `real_pts = normalized_pts *
norm_scale + norm_center`. `unnormalize()` below raises loudly if these
fields are missing for a given scan rather than silently computing
meaningless normalized-space "traits".

**Axis convention**: Z is treated as the up/height axis, per
`scripts/augmentations.py`'s existing convention that rotation
augmentation is Z-axis-only ("the plant's up-axis should usually NOT be
rotated") -- an inference grounded in that comment, not independently
re-verified against Pheno4D's own publication here.

**Label-scheme caveat, propagated from adapters/infer_segmentation.py**:
every trait below is only as reliable as the organ segmentation it's
computed from. Maize's segmentation uses B3b's standard data-driven
remap (held-out mIoU ~0.55-0.59). Tomato's uses the height_split fallback
(held-out recall: soil 92.84%, stem 76.69%, leaf 68.66%, mIoU 0.571) --
see step_notes/D0_Trait_Extraction_Pipeline.md for the full derivation
of both. Neither is perfect; traits computed from misclassified points
will carry that error forward.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull, Delaunay, QhullError
from sklearn.cluster import DBSCAN

_REPO_ROOT = Path(__file__).resolve().parent.parent

MIN_POINTS = {"soil": 20, "stem": 15, "leaf": 30}  # below this, a trait involving that organ is NaN


def unnormalize(points: np.ndarray, norm_scale: float, norm_center) -> np.ndarray:
    """Inverts scripts/preprocessing.py::normalize_and_center. Raises a
    clear ValueError if norm_scale is missing/NaN rather than silently
    computing traits in meaningless unit-sphere units -- see module
    docstring."""
    if norm_scale is None or (isinstance(norm_scale, float) and np.isnan(norm_scale)):
        raise ValueError(
            "norm_scale is missing/NaN -- this scan's manifest row predates the "
            "preprocessing fix that persists norm_scale/norm_center (see scripts/"
            "preprocessing.py and step_notes/D0_Trait_Extraction_Pipeline.md). "
            "Re-run scripts/05_preprocess_pointclouds.py on the laptop (raw Pheno4D "
            "files aren't on the cluster) and transfer the regenerated cache before "
            "extracting real-unit traits for this scan.")
    center = np.asarray(norm_center, dtype=np.float64)
    return points.astype(np.float64) * float(norm_scale) + center


def height(real_pts: np.ndarray, pred_organ: np.ndarray) -> float:
    """Z-extent of all non-soil (stem+leaf) points."""
    mask = pred_organ != 0
    if mask.sum() < MIN_POINTS["stem"] + MIN_POINTS["leaf"]:
        return float("nan")
    z = real_pts[mask, 2]
    return float(z.max() - z.min())


def stem_diameter(real_pts: np.ndarray, pred_organ: np.ndarray) -> float:
    """Basal (bottom 20% of stem height) cross-sectional diameter, via a
    ROBUST (median-distance) circular-disk approximation instead of a
    covariance/variance-based one.

    REVISED after the first version (variance-based: diameter = 4*std of
    the XY covariance) produced physically impossible results on real
    data -- 35/223 scans (16%, concentrated in bushy, late-stage Tomato
    scans) came out with stem_diameter >= height. Root cause, confirmed
    by direct inspection (not guessed): the basal slice's false-positive
    "stem" points (misclassified leaf/canopy points -- B3b's stem recall
    is 77% for Tomato / 40-48% for Maize, not 100%, see step_notes/
    D0_Trait_Extraction_Pipeline.md) are scattered far from the true
    stem axis, and variance is quadratic in deviation -- a small minority
    of scattered outliers dominates it. Example (T03_0325_a.npz, 122
    basal points): median distance from the slice's own center was 0.069
    (a tight, plausible stem core) while the mean was 0.463 and the max
    1.244 -- the variance-based formula was reporting the outliers'
    scale, not the stem's.

    Fix: use the MEDIAN distance from the slice's own median XY center
    instead. For a uniform 2D disk of radius R, the CDF of radial
    distance is (r/R)^2, so the median radius is R/sqrt(2) -- giving
    R = median_dist * sqrt(2), diameter = 2R = median_dist * 2*sqrt(2).
    Medians are robust to contamination up to ~50%, unlike variance,
    which is exactly the property needed here given segmentation isn't
    perfect. Falls back to using all stem points if the basal slice is
    too sparse."""
    stem_pts = real_pts[pred_organ == 1]
    if len(stem_pts) < MIN_POINTS["stem"]:
        return float("nan")
    z = stem_pts[:, 2]
    z0, span = z.min(), z.max() - z.min()
    if span <= 1e-9:
        slice_pts = stem_pts
    else:
        slice_mask = z <= z0 + 0.2 * span
        slice_pts = stem_pts[slice_mask] if slice_mask.sum() >= 5 else stem_pts
    xy = slice_pts[:, :2]
    center = np.median(xy, axis=0)
    dists = np.linalg.norm(xy - center, axis=1)
    median_dist = float(np.median(dists))
    return 2.0 * np.sqrt(2.0) * median_dist


def _cluster_leaves(leaf_pts: np.ndarray, eps: float = None, min_samples: int = 6):
    """Shared DBSCAN clustering for leaf_area and leaf_count -- both are
    downstream of the same per-leaf-instance clusters, computed once here
    rather than twice. min_samples=6 follows the standard DBSCAN
    minPts>=2*dim heuristic for noisy data (dim=3). eps, if not given, is
    adaptive per-scan: 2x the median distance to the 4th-nearest
    neighbor (a standard k-distance-elbow convention) -- not an absolute
    threshold, so this generalizes across scans/growth stages without
    per-scan retuning. Returns cluster labels (-1 = noise, per sklearn
    convention)."""
    if len(leaf_pts) < min_samples:
        return np.full(len(leaf_pts), -1, dtype=np.int64)
    if eps is None:
        tree = cKDTree(leaf_pts)
        k = min(5, len(leaf_pts))
        dists, _ = tree.query(leaf_pts, k=k)
        d4 = dists[:, -1]
        eps = 2.0 * float(np.median(d4))
        if eps <= 0:
            eps = 1e-6
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(leaf_pts)
    return labels


def leaf_count(real_pts: np.ndarray, pred_organ: np.ndarray) -> float:
    """Cannot come from the model directly -- B3b's segmentation head
    only predicts 3 semantic classes (soil/stem/leaf), no per-instance
    output (pheno4d_collapse_organ_labels collapses raw per-leaf instance
    ids into one "leaf" class before any training). Geometric heuristic:
    DBSCAN cluster count on leaf-predicted points, excluding noise."""
    leaf_pts = real_pts[pred_organ == 2]
    if len(leaf_pts) < MIN_POINTS["leaf"]:
        return float("nan")
    labels = _cluster_leaves(leaf_pts)
    unique = set(labels.tolist())
    unique.discard(-1)
    return float(len(unique))


def leaf_area(real_pts: np.ndarray, pred_organ: np.ndarray) -> float:
    """Sum of per-leaf-cluster PROJECTED planar area (not curved 3D
    surface area -- the standard, accepted simplification in point-cloud
    leaf-area-estimation literature). Per cluster: PCA-project onto the
    leaf's own dominant plane, Delaunay-triangulate in 2D, then an
    adaptive alpha-shape filter (drop triangles whose circumradius
    exceeds a density-derived threshold) before summing triangle areas --
    plain convex hull would overestimate area for lobed/serrated leaf
    margins (tomato leaves are deeply lobed)."""
    leaf_pts = real_pts[pred_organ == 2]
    if len(leaf_pts) < MIN_POINTS["leaf"]:
        return float("nan")
    labels = _cluster_leaves(leaf_pts)
    total_area = 0.0
    for cluster_id in set(labels.tolist()):
        if cluster_id == -1:
            continue
        cluster_pts = leaf_pts[labels == cluster_id]
        if len(cluster_pts) < 4:
            continue
        cov3d = np.cov((cluster_pts - cluster_pts.mean(axis=0)).T)
        eigvals, eigvecs = np.linalg.eigh(cov3d)
        basis = eigvecs[:, np.argsort(eigvals)[::-1][:2]]  # top-2 eigenvectors
        pts2d = (cluster_pts - cluster_pts.mean(axis=0)) @ basis
        try:
            tri = Delaunay(pts2d)
        except QhullError:
            continue
        tree2d = cKDTree(pts2d)
        k = min(4, len(pts2d))
        dists, _ = tree2d.query(pts2d, k=k)
        alpha = 2.5 * float(np.median(dists[:, -1]))
        for simplex in tri.simplices:
            a, b, c = pts2d[simplex]
            ab, ac = b - a, c - a
            edge_lens = [np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)]
            circumradius_proxy = max(edge_lens)  # cheap proxy: longest edge, not true circumradius
            if circumradius_proxy > alpha:
                continue
            tri_area = 0.5 * abs(ab[0] * ac[1] - ab[1] * ac[0])
            total_area += tri_area
    return float(total_area)


def volume(real_pts: np.ndarray, pred_organ: np.ndarray) -> float:
    """Convex hull volume over all non-soil (stem+leaf) points -- the
    standard literature proxy for plant volume from point clouds. Known,
    accepted tradeoff: overestimates true volume for non-convex canopy
    shapes (gaps between leaves), not a limitation unique to this
    pipeline."""
    plant_pts = real_pts[pred_organ != 0]
    if len(plant_pts) < 4:
        return float("nan")
    try:
        return float(ConvexHull(plant_pts).volume)
    except QhullError:
        return float("nan")


def extract_all_traits(real_pts: np.ndarray, pred_organ: np.ndarray) -> dict:
    return {
        "height": height(real_pts, pred_organ),
        "stem_diameter": stem_diameter(real_pts, pred_organ),
        "leaf_area": leaf_area(real_pts, pred_organ),
        "leaf_count": leaf_count(real_pts, pred_organ),
        "volume": volume(real_pts, pred_organ),
        "n_soil_pts": int((pred_organ == 0).sum()),
        "n_stem_pts": int((pred_organ == 1).sum()),
        "n_leaf_pts": int((pred_organ == 2).sum()),
    }


# Threshold justified empirically (not an arbitrary round number): see
# step_notes/D0_Trait_Extraction_Pipeline.md's "known segmentation failure
# mode" section. STEM_FRAC_THRESHOLD=0.15 sits between the dataset's
# small-plant "cluster A" outliers (2.4-4.3% stem-of-points, benign --
# just tiny absolute scale, not a segmentation problem) and the
# bushy/late-stage "cluster B" failures (20.9-59.9%, confirmed via real
# ground truth on 3 scans to be genuine over-prediction of stem, 3.2-4.3x
# too many stem points, 51-68% pixel agreement) -- combined with the
# already-physically-motivated stem_diameter>=height check, this exact
# pair of conditions reproduces the 6 manually-identified cluster-B scans
# with no other false positives or negatives (verified against the full
# 223-scan dataset).
STEM_FRAC_THRESHOLD = 0.15


def flag_stem_leaf_boundary_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `stem_frac_of_points` and `stem_leaf_boundary_low_confidence`
    (bool) columns. The flag does NOT blank out any values (raw numbers
    are kept for transparency/debugging) -- it marks which scans'
    stem/leaf-boundary-dependent traits should be excluded from anything
    trait-value-sensitive downstream (confirmed via ground truth on 3
    annotated scans to specifically corrupt stem_diameter [13-24x
    overestimate], leaf_area [38-64% UNDERestimate -- true leaf points
    misclassified as stem are excluded from the leaf-area computation],
    and leaf_count [40-71% OVERestimate -- removing points from a leaf
    cluster can fragment it into multiple smaller DBSCAN clusters].
    height and volume were confirmed NOT meaningfully affected on the
    same 3 scans (height: -1.8% to +0.0% difference; volume: -7.9% to
    +1.1%) -- both stay well within normal noise because they're
    computed over ALL non-soil points regardless of the stem/leaf split,
    so misclassifying a point AS stem instead of leaf, or vice versa,
    doesn't change either computation. See step_notes/
    D0_Trait_Extraction_Pipeline.md for the full derivation and the
    per-scan predicted-vs-ground-truth comparison this is based on."""
    df = df.copy()
    df["stem_frac_of_points"] = df["n_stem_pts"] / (df["n_soil_pts"] + df["n_stem_pts"] + df["n_leaf_pts"])
    df["stem_leaf_boundary_low_confidence"] = (
        (df["stem_diameter"] >= df["height"]) & (df["stem_frac_of_points"] > STEM_FRAC_THRESHOLD)
    )
    return df


def main():
    ap = argparse.ArgumentParser(description="D0: extract traits from segmented Pheno4D scans")
    ap.add_argument("--seg_dir", type=str, default=str(_REPO_ROOT / "data" / "segmented" / "Pheno4D"))
    ap.add_argument("--manifest", type=str,
                     default=str(_REPO_ROOT / "data" / "preprocessed" / "preprocessed_manifest.csv"))
    ap.add_argument("--out", type=str, default=str(_REPO_ROOT / "data" / "traits" / "pheno4d_traits_per_scan.csv"))
    args = ap.parse_args()

    seg_dir = Path(args.seg_dir)
    manifest = pd.read_csv(args.manifest)
    pheno = manifest[manifest["dataset"] == "Pheno4D"].copy()
    pheno["cache_name"] = pheno["cache_path"].apply(lambda p: Path(p).name)

    rows = []
    n_ok, n_missing_scale, n_missing_file, n_failed = 0, 0, 0, 0
    for _, row in pheno.iterrows():
        seg_path = seg_dir / row["species"] / row["cache_name"]
        if not seg_path.exists():
            n_missing_file += 1
            continue
        with np.load(seg_path) as d:
            pts_norm = d["points"]
            pred_organ = d["pred_organ"]

        norm_scale = row["norm_scale"]
        if pd.isna(norm_scale):
            n_missing_scale += 1
            continue
        norm_center = [row["norm_center_x"], row["norm_center_y"], row["norm_center_z"]]

        try:
            real_pts = unnormalize(pts_norm, float(norm_scale), norm_center)
            traits = extract_all_traits(real_pts, pred_organ)
        except Exception as e:
            print(f"[error] {seg_path}: {e}")
            n_failed += 1
            continue

        rows.append({
            "plant_id": row["plant_id"],
            "species": row["species"],
            "scan_date": row["scan_date"],
            "source_filepath": row["source_filepath"],
            **traits,
        })
        n_ok += 1

    print(f"Processed {n_ok} scans (missing segmented file: {n_missing_file}, "
          f"missing norm_scale: {n_missing_scale}, failed: {n_failed})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = flag_stem_leaf_boundary_confidence(df)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    n_flagged = int(df["stem_leaf_boundary_low_confidence"].sum())
    print(f"stem_leaf_boundary_low_confidence flagged {n_flagged} scans -- "
          f"stem_diameter/leaf_area/leaf_count unreliable for these (height/volume unaffected). "
          f"See step_notes/D0_Trait_Extraction_Pipeline.md.")
    if n_flagged:
        print(df.loc[df["stem_leaf_boundary_low_confidence"],
                      ["plant_id", "species", "scan_date"]].to_string(index=False))
    print(df.groupby("species")[["height", "stem_diameter", "leaf_area", "leaf_count", "volume"]]
          .agg(["mean", "std", lambda x: x.isna().sum()]))


if __name__ == "__main__":
    main()
