"""
augmentations.py

3D point cloud augmentation functions matching the pipeline you listed
(Rotation/Affine, Scale, Gaussian Noise, Flip, RandomCrop3D, CenterCrop3D,
Pad3D, CoarseDropout3D, Cubic Symmetry).

Every function takes and returns an (N, 3) float32 numpy array.
Kept dependency-free (pure numpy) so they're easy to port into a
PyTorch Dataset's __getitem__ later.
"""

import numpy as np


def random_rotation_z(pts: np.ndarray, rng=None) -> np.ndarray:
    """Random rotation about the vertical (Z) axis — the plant's up-axis
    should usually NOT be rotated (a sideways-grown plant is unrealistic),
    so we only rotate around Z, not all 3 axes."""
    rng = rng or np.random.default_rng()
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return pts @ R.T


def gaussian_noise(pts: np.ndarray, sigma: float = 0.01, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return pts + rng.normal(0, sigma, size=pts.shape).astype(np.float32)


def random_scale(pts: np.ndarray, lo: float = 0.8, hi: float = 1.2, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    s = rng.uniform(lo, hi)
    return pts * s


def random_flip(pts: np.ndarray, axis: str = "x", rng=None) -> np.ndarray:
    """Flip along a horizontal axis only (x or y) — flipping Z would put
    the plant's roots in the air, which isn't a realistic augmentation."""
    rng = rng or np.random.default_rng()
    if rng.random() > 0.5:
        return pts
    out = pts.copy()
    ax = {"x": 0, "y": 1}[axis]
    out[:, ax] = -out[:, ax]
    return out


def _random_crop3d_mask(pts: np.ndarray, keep_frac: float, rng) -> np.ndarray:
    axis = rng.integers(0, 3)
    lo, hi = pts[:, axis].min(), pts[:, axis].max()
    span = hi - lo
    crop_span = span * keep_frac
    start = rng.uniform(lo, hi - crop_span) if hi > lo + crop_span else lo
    mask = (pts[:, axis] >= start) & (pts[:, axis] <= start + crop_span)
    if not mask.any():  # guard against empty result
        mask = np.ones(len(pts), dtype=bool)
    return mask


def random_crop3d(pts: np.ndarray, keep_frac: float = 0.8, rng=None) -> np.ndarray:
    """Randomly crop a contiguous spatial chunk, keeping ~keep_frac of the
    bounding volume along a randomly chosen axis (simulates partial
    occlusion / scan cutoff)."""
    rng = rng or np.random.default_rng()
    mask = _random_crop3d_mask(pts, keep_frac, rng)
    return pts[mask]


def center_crop3d(pts: np.ndarray, keep_frac: float = 0.8) -> np.ndarray:
    """Deterministic crop centered on the point cloud's centroid — used
    at eval time (no randomness) rather than during training."""
    center = pts.mean(axis=0)
    extent = (pts.max(axis=0) - pts.min(axis=0)) * keep_frac / 2.0
    mask = np.all(np.abs(pts - center) <= extent, axis=1)
    cropped = pts[mask]
    return cropped if len(cropped) > 0 else pts


def pad_if_needed3d(pts: np.ndarray, target_n: int, rng=None) -> np.ndarray:
    """Pad up to target_n points by duplicating random existing points.
    Needed after crop/dropout ops if your model expects a fixed N."""
    rng = rng or np.random.default_rng()
    if len(pts) >= target_n:
        return pts
    n_pad = target_n - len(pts)
    pad_idx = rng.choice(len(pts), n_pad, replace=True)
    return np.concatenate([pts, pts[pad_idx]], axis=0)


def pad_if_needed3d_with_labels(pts: np.ndarray, labels: np.ndarray, target_n: int, rng=None):
    """Same as pad_if_needed3d but duplicates the parallel per-point label
    array with the same sampled indices, so labels stay aligned with points."""
    rng = rng or np.random.default_rng()
    if len(pts) >= target_n:
        return pts, labels
    n_pad = target_n - len(pts)
    pad_idx = rng.choice(len(pts), n_pad, replace=True)
    return (np.concatenate([pts, pts[pad_idx]], axis=0),
            np.concatenate([labels, labels[pad_idx]], axis=0))


def _coarse_dropout3d_mask(pts: np.ndarray, n_holes: int, hole_frac: float, rng) -> np.ndarray:
    extent = pts.max(axis=0) - pts.min(axis=0)
    radius = float(np.linalg.norm(extent)) * hole_frac
    keep_mask = np.ones(len(pts), dtype=bool)
    for _ in range(n_holes):
        center_idx = rng.integers(0, len(pts))
        center = pts[center_idx]
        dist = np.linalg.norm(pts - center, axis=1)
        keep_mask &= dist > radius
    if not keep_mask.any():  # guard against empty result
        keep_mask = np.ones(len(pts), dtype=bool)
    return keep_mask


def coarse_dropout3d(pts: np.ndarray, n_holes: int = 3, hole_frac: float = 0.05, rng=None) -> np.ndarray:
    """Remove n_holes small spherical neighborhoods from the cloud —
    simulates sensor dropout / leaf self-occlusion."""
    rng = rng or np.random.default_rng()
    mask = _coarse_dropout3d_mask(pts, n_holes, hole_frac, rng)
    return pts[mask]


def cubic_symmetry(pts: np.ndarray, rng=None) -> np.ndarray:
    """Random sign flip per axis (the 8 symmetries of a cube), a cheap
    generalization of random_flip across all 3 axes at once."""
    rng = rng or np.random.default_rng()
    signs = rng.choice([-1, 1], size=3).astype(np.float32)
    return pts * signs


def normalize(pts: np.ndarray) -> np.ndarray:
    """Center at origin and scale to unit sphere — standard preprocessing,
    not technically an augmentation, but included since every other
    op assumes roughly-centered input."""
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.max(np.linalg.norm(pts, axis=1))
    return pts / (scale + 1e-8)


def compose_pipeline(pts: np.ndarray, rng=None) -> np.ndarray:
    """The full augmentation pipeline applied together, matching the
    pipeline diagram: normalize -> rotate -> scale -> noise -> flip ->
    crop -> dropout -> cubic symmetry."""
    rng = rng or np.random.default_rng()
    pts = normalize(pts)
    pts = random_rotation_z(pts, rng)
    pts = random_scale(pts, rng=rng)
    pts = gaussian_noise(pts, rng=rng)
    pts = random_flip(pts, axis="x", rng=rng)
    pts = random_crop3d(pts, keep_frac=0.85, rng=rng)
    pts = coarse_dropout3d(pts, n_holes=2, rng=rng)
    pts = cubic_symmetry(pts, rng=rng)
    return pts


def compose_pipeline_with_labels(pts: np.ndarray, labels: np.ndarray, rng=None):
    """Same pipeline as compose_pipeline, but also carries a parallel (N,)
    int label array through the point-count-changing steps (crop, dropout)
    so per-point segmentation labels stay aligned with their points.
    Point-wise-only ops (rotate/scale/noise/flip/cubic symmetry) don't
    change point count or order, so labels pass through those untouched."""
    rng = rng or np.random.default_rng()
    pts = normalize(pts)
    pts = random_rotation_z(pts, rng)
    pts = random_scale(pts, rng=rng)
    pts = gaussian_noise(pts, rng=rng)
    pts = random_flip(pts, axis="x", rng=rng)

    mask = _random_crop3d_mask(pts, keep_frac=0.85, rng=rng)
    pts, labels = pts[mask], labels[mask]

    mask = _coarse_dropout3d_mask(pts, n_holes=2, hole_frac=0.05, rng=rng)
    pts, labels = pts[mask], labels[mask]

    pts = cubic_symmetry(pts, rng=rng)
    return pts, labels


def compose_pipeline_ln_only(pts: np.ndarray, rng=None) -> np.ndarray:
    """L-N only (Gaussian jitter) -- for row C4 (DGCNN, DA-A, L-N), which
    isolates DGCNN's noise weakness per CLAUDE.md's augmentation taxonomy.
    Skips G-R (rotate/flip/cubic symmetry), G-S (scale), and L-D
    (crop/dropout) -- only L-N is applied, on top of normalize(), which is
    a fixed preprocessing step, not one of the 4 invariance-inducing
    categories (same treatment CLAUDE.md gives Pad3D)."""
    rng = rng or np.random.default_rng()
    pts = normalize(pts)
    pts = gaussian_noise(pts, rng=rng)
    return pts


def compose_pipeline_ln_only_with_labels(pts: np.ndarray, labels: np.ndarray, rng=None):
    """Same as compose_pipeline_ln_only, but threads per-point labels
    through unchanged -- gaussian_noise doesn't alter point count or
    order, so no re-indexing of labels is needed (unlike crop/dropout)."""
    rng = rng or np.random.default_rng()
    pts = normalize(pts)
    pts = gaussian_noise(pts, rng=rng)
    return pts, labels


def compose_pipeline_ld_only(pts: np.ndarray, rng=None) -> np.ndarray:
    """L-D only (RandomCrop3D, CoarseDropout3D) -- for row B4 (KPConv, DA-0, L-D), which
    deliberately runs with NO adaptation method, isolating whether KPConv's claimed
    density-robustness (CLAUDE.md's Backbones section: "density-robust due to grid-subsampled
    neighborhoods, not fixed point count") holds up against local-structural augmentation alone,
    without any adaptation method's help. Skips G-R (rotate/flip/cubic symmetry), G-S (scale),
    and L-N (noise) -- only L-D is applied, on top of normalize(), the same fixed-preprocessing
    treatment given to Pad3D and to compose_pipeline_ln_only's own normalize() call. Same
    keep_frac/n_holes/hole_frac defaults as the L-D steps inside compose_pipeline (0.85, 2, 0.05)
    -- isolating which augmentation is applied, not changing the augmentation's own parameters."""
    rng = rng or np.random.default_rng()
    pts = normalize(pts)
    pts = random_crop3d(pts, keep_frac=0.85, rng=rng)
    pts = coarse_dropout3d(pts, n_holes=2, rng=rng)
    return pts


def compose_pipeline_ld_only_with_labels(pts: np.ndarray, labels: np.ndarray, rng=None):
    """Same as compose_pipeline_ld_only, but threads per-point labels through the
    point-count-changing crop/dropout steps -- same mask-and-reindex pattern
    compose_pipeline_with_labels already uses for its own L-D steps."""
    rng = rng or np.random.default_rng()
    pts = normalize(pts)

    mask = _random_crop3d_mask(pts, keep_frac=0.85, rng=rng)
    pts, labels = pts[mask], labels[mask]

    mask = _coarse_dropout3d_mask(pts, n_holes=2, hole_frac=0.05, rng=rng)
    pts, labels = pts[mask], labels[mask]

    return pts, labels
