"""
preprocessing.py

Core point cloud preprocessing: statistical outlier removal, farthest-point
sampling (FPS) to a fixed point count, and normalization. Pure numpy +
scipy (cKDTree), no open3d dependency (open3d has no Python 3.14 wheel yet).

Pipeline order (matches the caching script): outlier removal -> downsample
to target_n -> normalize/center. Downsampling itself is two-stage for
speed on real Pheno4D scans (1-4.5M raw points):
  1. voxel-grid pre-decimation (deterministic, density-adaptive, NOT random)
     down to ~20-40k points when the raw cloud is large enough to need it.
  2. true farthest-point sampling from that reduced set down to target_n.
Naive FPS straight from a 2M-point cloud takes ~3-4 minutes; voxel
pre-decimation first cuts that to a few seconds with negligible structural
loss, since the voxel step keeps one real point per occupied cell rather
than dropping points randomly.
"""

import numpy as np
from scipy.spatial import cKDTree


def remove_statistical_outliers(pts: np.ndarray, labels: np.ndarray = None,
                                 k: int = 20, std_ratio: float = 2.0):
    """
    Statistical outlier removal (same idea as Open3D's remove_statistical_outlier):
    for each point, compute its mean distance to its k nearest neighbors; a
    point is an outlier if that mean distance exceeds the dataset-wide
    mean + std_ratio * std.

    Returns (clean_pts, clean_labels_or_None, n_removed).
    """
    n = len(pts)
    if n <= k:
        return pts, labels, 0
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=k + 1, workers=-1)  # includes self at dist 0
    mean_dist = dist[:, 1:].mean(axis=1)
    thresh = mean_dist.mean() + std_ratio * mean_dist.std()
    keep = mean_dist <= thresh
    n_removed = int((~keep).sum())
    clean_labels = labels[keep] if labels is not None else None
    return pts[keep], clean_labels, n_removed


def voxel_grid_downsample(pts: np.ndarray, labels: np.ndarray = None,
                           voxel_size: float = None, target_count: int = 30000):
    """
    Deterministic spatial-grid downsampling: bins points into a regular
    voxel grid and keeps, per occupied voxel, the real point closest to
    that voxel's centroid (not the centroid itself, so labels stay valid
    and no synthetic points are introduced). This is density-adaptive and
    NOT random subsampling — it exists purely to make the subsequent FPS
    pass fast on million-point raw scans.

    If voxel_size is None, it's derived from the cloud's bounding-box
    diagonal so the result lands near target_count points regardless of
    the plant's physical scale.
    """
    n = len(pts)
    if voxel_size is None:
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if diag == 0:
            return pts, labels
        voxel_size = diag / 200.0  # empirically ~20-40k points on real Pheno4D scans

    mins = pts.min(axis=0)
    vidx = np.floor((pts - mins) / voxel_size).astype(np.int64)
    # hash the 3D grid index to a single int key (spatial hash, collisions negligible
    # at plant-cloud scale)
    key = (vidx[:, 0] * 73856093) ^ (vidx[:, 1] * 19349663) ^ (vidx[:, 2] * 83492791)

    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 3), dtype=np.float64)
    np.add.at(sums, inv, pts)
    centroids = sums / counts[:, None]

    d = np.sum((pts - centroids[inv]) ** 2, axis=1)
    order = np.lexsort((d, inv))
    inv_sorted = inv[order]
    first_mask = np.concatenate(([True], inv_sorted[1:] != inv_sorted[:-1]))
    selected_idx = order[first_mask]

    sel_labels = labels[selected_idx] if labels is not None else None
    return pts[selected_idx], sel_labels


def farthest_point_sampling(pts: np.ndarray, n_samples: int, seed: int = None) -> np.ndarray:
    """
    Classic iterative FPS, vectorized in numpy (float32). Returns an
    (n_samples,) int array of indices into pts. O(N * n_samples) —
    only tractable because callers pre-decimate large clouds first.
    """
    pts32 = pts.astype(np.float32)
    rng = np.random.default_rng(seed)
    n = len(pts32)
    selected = np.empty(n_samples, dtype=np.int64)
    farthest = int(rng.integers(0, n))
    selected[0] = farthest
    dist = np.full(n, np.inf, dtype=np.float32)
    for i in range(1, n_samples):
        diff = pts32 - pts32[farthest]
        d = np.einsum("ij,ij->i", diff, diff)
        np.minimum(dist, d, out=dist)
        farthest = int(np.argmax(dist))
        selected[i] = farthest
    return selected


def normalize_and_center(pts: np.ndarray):
    """Center at centroid, scale to unit sphere (max radius = 1).

    Returns (pts_normalized, center (3,) float64, scale float) -- center/scale
    are the exact values needed to invert this transform later
    (real_pts = pts_normalized * scale + center), e.g. for trait extraction
    in physical units. Persisted downstream by preprocess_pointcloud's stats
    dict -> 05_preprocess_pointclouds.py's manifest/.npz -- see
    scripts/trait_extraction.py for the consumer.
    """
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.max(np.linalg.norm(pts, axis=1))
    return pts / (scale + 1e-8), center, float(scale)


def preprocess_pointcloud(pts: np.ndarray, labels: np.ndarray = None,
                           target_n: int = 4096,
                           sor_k: int = 20, sor_std_ratio: float = 2.0,
                           voxel_predecimate_above: int = 50000,
                           voxel_target_count: int = 30000,
                           seed: int = None):
    """
    Full pipeline: outlier removal -> downsample to target_n -> normalize.

    Downsample behavior:
      - N <= target_n: keep all points, pad up to target_n by duplicating
        random existing points (so every cached cloud has exactly target_n
        points regardless of the raw scan's density).
      - N > target_n: voxel pre-decimate (only if N > voxel_predecimate_above,
        which real Crops3D_10k-scale clouds won't need) then true FPS to
        target_n.

    Returns (pts_out (target_n,3) float32, labels_out or None, stats dict).
    """
    n_raw = len(pts)
    pts_c, labels_c, n_removed = remove_statistical_outliers(pts, labels, k=sor_k, std_ratio=sor_std_ratio)
    n_clean = len(pts_c)

    if n_clean <= target_n:
        rng = np.random.default_rng(seed)
        if n_clean < target_n:
            pad_idx = rng.choice(n_clean, target_n - n_clean, replace=True)
            pts_out = np.concatenate([pts_c, pts_c[pad_idx]], axis=0)
            labels_out = (np.concatenate([labels_c, labels_c[pad_idx]], axis=0)
                          if labels_c is not None else None)
        else:
            pts_out, labels_out = pts_c, labels_c
        n_predecimated = n_clean
    else:
        work_pts, work_labels = pts_c, labels_c
        if n_clean > voxel_predecimate_above:
            work_pts, work_labels = voxel_grid_downsample(
                pts_c, labels_c, target_count=voxel_target_count)
            if len(work_pts) < target_n:
                # pathological voxel sizing (very sparse cloud) — fall back to raw clean set
                work_pts, work_labels = pts_c, labels_c
        n_predecimated = len(work_pts)
        fps_idx = farthest_point_sampling(work_pts, target_n, seed=seed)
        pts_out = work_pts[fps_idx]
        labels_out = work_labels[fps_idx] if work_labels is not None else None

    pts_out, norm_center, norm_scale = normalize_and_center(pts_out.astype(np.float32))

    stats = {
        "n_raw": n_raw,
        "n_after_outlier_removal": n_clean,
        "n_outliers_removed": n_removed,
        "n_predecimated": n_predecimated,
        "n_final": target_n,
        "norm_scale": norm_scale,
        "norm_center_x": float(norm_center[0]),
        "norm_center_y": float(norm_center[1]),
        "norm_center_z": float(norm_center[2]),
    }
    return pts_out.astype(np.float32), labels_out, stats
