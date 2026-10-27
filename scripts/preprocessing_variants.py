"""
preprocessing_variants.py

Block PP, Stage 2 step 1: the 8 alternative preprocessing pipelines defined
in docs/PREPROCESSING_COMPARISON_PLAN.md section 5 (P1-P4, P4a-P4d), plus
their shared building blocks. P0 is intentionally NOT implemented here --
section 5 says to reuse the existing `data/preprocessed/` cache for P0
unchanged, and section 8's CLI spec for `07_preprocess_variants.py` only
lists `P1|P2|P3|P4|P4a|P4b|P4c|P4d`.

Does not modify `scripts/preprocessing.py` or `scripts/data_io.py` --
imports their primitives (`remove_statistical_outliers`,
`voxel_grid_downsample`, `farthest_point_sampling`, `normalize_and_center`)
unchanged and composes them differently per pipeline. Also imports
`find_soil_slab` from `scripts/08_soil_removal_check.py` (the validated,
label-free horizontal-slab Pheno4D soil detector -- step_notes/
PP_0_Prerequisite_Checks.md Part C) via `importlib`, since a filename
starting with a digit cannot be a normal Python import target.

Soil/pot removal rule (plan section 6, all CONFIRMED in Stage 1):
  - Crops3D Maize: drop scalar_sf ids {0 (soil), 5 (pot)} using Crops3D's
    own source labels -- fine, Crops3D is the labeled source dataset.
  - Crops3D Tomato: no-op (CONFIRMED no soil id, no unlabeled ground layer).
  - Pheno4D (both species, annotated or not): the label-free horizontal-slab
    cut, margin = PHENO4D_SOIL_MARGIN_MM = 2.0mm (validated: 95.32% soil
    removed / 0.377% plant lost, pooled over all 126 annotated scans --
    reaches the plan's >95%/<1% starting target). Labels are NEVER used to
    decide this cut, on annotated scans or otherwise -- only this module's
    caller-side validation script (08) uses them, to score the result.

Up-axis rule (plan section 6: "determine per dataset... store the rotation
in a config/constant"):
  - Crops3D Maize, Pheno4D (both species): CONFIRMED +z already (step_notes/
    PP_0 checks 1/2/4) -- "aligning to +Z" is the identity transform.
  - Crops3D Tomato: NOT yet confirmed for all 83 clouds (Stage 1 Part C
    produced a 36-cloud eyeball montage, results/pp_checks/
    tomato_axis_montage.png, for the user to review -- 3 undecided options
    recorded in the step note, none picked). See TOMATO_UP_AXIS_ALIGN below.

Each pipeline function has the signature
    pipeline_PX(pts, labels=None, dataset=None, species=None,
                target_n=4096, sor_k=20, sor_std_ratio=2.0, seed=42)
        -> (pts_out (target_n,3) float32, norm_center (3,) float64,
            norm_scale float, info: dict)
`info` holds per-step point counts, soil/pot-removal and voxel/sampling
metadata (all scalars, safe for a manifest CSV), plus one non-scalar key,
`labels_out` (the (target_n,) or (target_n,2) int array of surviving point
labels, or None) -- callers must pop this before writing a CSV row and can
save it straight into the cache .npz instead. Crops3D Maize's surviving
labels keep their ORIGINAL ids (1-4) after id 0/5 are dropped -- NOT
renumbered to be contiguous from 0; any renumbering is a separate,
not-yet-needed downstream concern.

All randomness (FPS start point, padding, random sampling) is controlled by
`seed`, matching 05_preprocess_pointclouds.py's convention of one global
seed reused for every file, for reproducibility across runs.
"""
import importlib.util
from pathlib import Path

import numpy as np

from preprocessing import (
    remove_statistical_outliers,
    voxel_grid_downsample,
    farthest_point_sampling,
    normalize_and_center,
)

# --- find_soil_slab from 08_soil_removal_check.py (digit-prefixed filename,
# not a valid bare import target -- load by path instead). Does not modify
# that file; only reads its already-validated function. ---
_THIS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_pp_soil_removal_check", _THIS_DIR / "08_soil_removal_check.py")
_soil_check_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_soil_check_mod)
find_soil_slab = _soil_check_mod.find_soil_slab

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UP_AXIS = 2  # +z. CONFIRMED for Crops3D Maize and Pheno4D (both species) --
             # step_notes/PP_0_Prerequisite_Checks.md checks 1, 2 and 4.

# Crops3D tomato's up-axis is the one Stage-1 item deliberately left open
# (see module docstring). False = trust the raw +z convention as-is (every
# spot-checked cloud so far, including the heuristics' own worst
# disagreements, rendered correctly z-up in the montage -- see step_notes/
# PP_0_Prerequisite_Checks.md Part C task 1). Flip to True only AFTER
# deciding which of that step note's 3 options to use AND implementing the
# corresponding rotation in align_up_axis() below -- flipping this constant
# alone does nothing but raise, by design (see align_up_axis's docstring).
TOMATO_UP_AXIS_ALIGN = False

CROPS3D_MAIZE_SOIL_ID = 0  # CONFIRMED, step_notes/PP_0 check 1
CROPS3D_MAIZE_POT_ID = 5   # CONFIRMED, step_notes/PP_0 check 1

PHENO4D_SOIL_MARGIN_MM = 2.0  # validated in step_notes/PP_0 Part C task 2

DEFAULT_TARGET_N = 4096
DEFAULT_SOR_K = 20
DEFAULT_SOR_STD_RATIO = 2.0
DEFAULT_SEED = 42

# Defensive cap on how many points ever reach the O(N*target_n) FPS loop.
# The voxel step upstream of FPS in every pipeline that uses it should
# already have reduced the cloud to well under this, but pipelines that
# skip/short-circuit voxelling (e.g. a tiny raw cloud, or P4b's conditional
# path under 50k) could in principle still hand FPS a large cloud -- this
# keeps worst-case runtime bounded rather than silently taking minutes on a
# single file. A uniform random subsample before FPS, not a structural
# change to FPS itself.
FPS_SAFETY_CAP = 60000


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def align_up_axis(pts: np.ndarray, dataset: str, species: str) -> np.ndarray:
    """
    "Align up-axis to +Z" per plan section 6. Returns pts UNCHANGED for
    every dataset/species whose up axis is CONFIRMED +z already (Crops3D
    Maize; Pheno4D Tomato/Maize) -- the native coordinate frame already has
    z as up, so aligning to +Z is the identity transform for those.

    For Crops3D Tomato (the one open case), behaviour is gated by
    TOMATO_UP_AXIS_ALIGN:
      False (default) -- identity transform, same as the confirmed cases.
      True  -- raises NotImplementedError. Deliberately not guessed at:
               step_notes/PP_0_Prerequisite_Checks.md Part C gives 3 options
               with different rotation logic and different risk profiles
               (one of which, per-cloud PCA/stem-direction alignment, was
               shown there to be ACTIVELY WRONG for at least 2 of the 36
               montage clouds); whichever is chosen needs its own rotation
               written in here, not a default guess.
    """
    if dataset == "Crops3D" and species == "Tomato":
        if TOMATO_UP_AXIS_ALIGN:
            raise NotImplementedError(
                "TOMATO_UP_AXIS_ALIGN=True but no rotation rule is implemented yet. "
                "Decide among the 3 options in step_notes/PP_0_Prerequisite_Checks.md "
                "Part C (task 1) using results/pp_checks/tomato_axis_montage.png, then "
                "implement the corresponding rotation in align_up_axis() before flipping "
                "this constant.")
        return pts
    # Crops3D Maize, Pheno4D Maize/Tomato: CONFIRMED +z already -- identity.
    return pts


def remove_soil_pot(pts: np.ndarray, labels, dataset: str, species: str):
    """
    Soil/pot removal per plan section 6. Returns
    (pts_clean, labels_clean_or_None, n_removed, used_source_labels: bool).

    - Crops3D Maize: labels REQUIRED (source labels permitted -- Crops3D is
      the labeled source dataset); drops ids {CROPS3D_MAIZE_SOIL_ID,
      CROPS3D_MAIZE_POT_ID}.
    - Crops3D Tomato: no-op, n_removed=0.
    - Pheno4D (either species, annotated or not): labels are NEVER used to
      decide the cut -- find_soil_slab() is fully geometric. Cuts at
      z_hi + PHENO4D_SOIL_MARGIN_MM, where (_, z_hi, _, _) is the detector's
      best-scoring slab. If the detector can't find a slab (too few points)
      or the cut would remove every point, the removal is skipped (pts
      returned unchanged, n_removed=0) rather than producing an empty cloud.
    """
    if dataset == "Crops3D" and species == "Maize":
        if labels is None:
            raise ValueError("Crops3D Maize soil/pot removal requires source labels "
                              "(ids 0 and 5), but labels=None was passed.")
        keep = ~np.isin(labels, (CROPS3D_MAIZE_SOIL_ID, CROPS3D_MAIZE_POT_ID))
        if not keep.any():
            return pts, labels, 0, False
        n_removed = int((~keep).sum())
        return pts[keep], labels[keep], n_removed, True

    if dataset == "Crops3D" and species == "Tomato":
        return pts, labels, 0, False

    if dataset == "Pheno4D":
        # find_soil_slab's own internal subsampling seed (default 0) is left
        # at its Part-C-validated value, deliberately NOT tied to this
        # pipeline's own --seed, so the validated detector behaviour is
        # reproduced exactly regardless of which --seed a run uses.
        slab = find_soil_slab(pts.astype(np.float64))
        if slab is None:
            return pts, labels, 0, False
        _, z_hi, _, _ = slab
        z0 = z_hi + PHENO4D_SOIL_MARGIN_MM
        keep = pts[:, UP_AXIS] > z0
        if not keep.any():
            return pts, labels, 0, False
        n_removed = int((~keep).sum())
        labels_clean = labels[keep] if labels is not None else None
        return pts[keep], labels_clean, n_removed, False

    raise ValueError(f"Unhandled dataset/species combination: {dataset}/{species}")


def normalize_unit_cube(pts: np.ndarray):
    """
    Centre at centroid, scale so every coordinate lies in [-1, 1] (a cube of
    half-width 1 -- the natural counterpart to normalize_and_center's
    "max radius = 1" unit-sphere convention already used elsewhere in this
    codebase). Returns (pts_normalized, center (3,) float64, scale float),
    same shape/semantics as normalize_and_center so callers can treat the
    two interchangeably.
    """
    center = pts.mean(axis=0)
    pts_c = pts - center
    scale = float(np.max(np.abs(pts_c))) if len(pts_c) else 1.0
    return pts_c / (scale + 1e-8), center, scale


def voxel_downsample_diag200(pts: np.ndarray, labels):
    """Voxel-downsample at voxel_size = bbox diagonal / 200, applied
    unconditionally (P1's "for every cloud", unlike the old P0 behaviour of
    only voxelling above 50k points). Returns (pts, labels, voxel_size_or_None)."""
    if len(pts) == 0:
        return pts, labels, None
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if diag <= 0:
        return pts, labels, None
    voxel_size = diag / 200.0
    p, l = voxel_grid_downsample(pts, labels, voxel_size=voxel_size)
    return p, l, voxel_size


def voxel_downsample_height_relative(pts: np.ndarray, labels, up_axis: int = UP_AXIS):
    """Voxel-downsample at voxel_size = plant height / 200 (section 6),
    height = extent along `up_axis` of the cloud AS PASSED IN (callers
    measure this after soil removal and alignment, per section 6's
    definition, by calling this after those steps). Returns
    (pts, labels, voxel_size_or_None)."""
    if len(pts) == 0:
        return pts, labels, None
    height = float(np.ptp(pts[:, up_axis]))
    if height <= 0:
        return pts, labels, None
    voxel_size = height / 200.0
    p, l = voxel_grid_downsample(pts, labels, voxel_size=voxel_size)
    return p, l, voxel_size


def fps_to_target(pts: np.ndarray, labels, target_n: int = DEFAULT_TARGET_N,
                   seed: int = DEFAULT_SEED):
    """
    Farthest-point sample down to exactly target_n points, or pad up to it
    by duplicating random existing points (same padding convention as
    preprocessing.py::preprocess_pointcloud, so every pipeline's output is
    exactly target_n points regardless of how few survived upstream steps).
    Applies FPS_SAFETY_CAP before FPS on unexpectedly large inputs.
    """
    n = len(pts)
    rng = np.random.default_rng(seed)
    if n <= target_n:
        if n == 0:
            raise ValueError("fps_to_target received an empty point cloud")
        if n < target_n:
            pad_idx = rng.choice(n, target_n - n, replace=True)
            pts_out = np.concatenate([pts, pts[pad_idx]], axis=0)
            labels_out = (np.concatenate([labels, labels[pad_idx]], axis=0)
                          if labels is not None else None)
        else:
            pts_out, labels_out = pts, labels
        return pts_out, labels_out

    work_pts, work_labels = pts, labels
    if n > FPS_SAFETY_CAP:
        idx = rng.choice(n, FPS_SAFETY_CAP, replace=False)
        work_pts = pts[idx]
        work_labels = labels[idx] if labels is not None else None

    fps_idx = farthest_point_sampling(work_pts, target_n, seed=seed)
    pts_out = work_pts[fps_idx]
    labels_out = work_labels[fps_idx] if work_labels is not None else None
    return pts_out, labels_out


def random_sample_to_target(pts: np.ndarray, labels, target_n: int = DEFAULT_TARGET_N,
                             seed: int = DEFAULT_SEED):
    """Random sampling to exactly target_n points (P2's "no FPS"): without
    replacement if there are enough points, with replacement (padding) if
    not."""
    n = len(pts)
    if n == 0:
        raise ValueError("random_sample_to_target received an empty point cloud")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, target_n, replace=(n < target_n))
    return pts[idx], (labels[idx] if labels is not None else None)


def outlier_and_soil_removed(pts: np.ndarray, labels, dataset: str, species: str,
                              sor_k: int = DEFAULT_SOR_K, sor_std_ratio: float = DEFAULT_SOR_STD_RATIO,
                              shared_cache_dir=None, stem: str = None):
    """
    Runs the two steps every pipeline shares identically: outlier removal
    (all 8 pipelines use the same sor_k/sor_std_ratio) and soil/pot removal
    (6 of 8 use the same rule/margin; P1 and P4a don't remove soil, but
    still call this for the shared outlier-removal result). Profiling
    (step_notes/PP_1_Pipelines.md) found outlier removal alone is ~60-70% of
    a single pipeline's per-cloud time and soil removal under 1% -- outlier
    removal is the real cost being shared here, soil removal just comes
    along for free since it's deterministic given the same inputs.

    If `shared_cache_dir` and `stem` are both given, checks
    <shared_cache_dir>/<dataset>/<species>/<stem>.npz first and returns its
    contents on a hit (and on a sor_k/sor_std_ratio mismatch, treats it as a
    miss rather than returning a stale result). On a miss, computes fresh
    and writes the cache entry via temp-file + os.replace (atomic on POSIX),
    so concurrently-running pipeline processes sharing the same
    shared_cache_dir can never observe a half-written entry -- at worst two
    processes race to compute + write the same missing entry once each
    (wasted duplicate work, not corruption, since the computation is
    deterministic). Passing shared_cache_dir=None (the default) disables
    caching entirely and reproduces the exact pre-caching behaviour.

    Returns a dict: n_raw, pts_after_outlier, labels_after_outlier,
    n_outliers_removed, pts_after_soil, labels_after_soil,
    n_soil_pot_removed, used_source_labels.
    """
    cache_file = None
    if shared_cache_dir is not None and stem is not None:
        cache_file = Path(shared_cache_dir) / dataset / species / f"{stem}.npz"
        if cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as d:
                if int(d["sor_k"]) == sor_k and float(d["sor_std_ratio"]) == sor_std_ratio:
                    return dict(
                        n_raw=int(d["n_raw"]),
                        pts_after_outlier=d["pts_after_outlier"],
                        labels_after_outlier=(d["labels_after_outlier"]
                                               if "labels_after_outlier" in d else None),
                        n_outliers_removed=int(d["n_outliers_removed"]),
                        pts_after_soil=d["pts_after_soil"],
                        labels_after_soil=(d["labels_after_soil"]
                                           if "labels_after_soil" in d else None),
                        n_soil_pot_removed=int(d["n_soil_pot_removed"]),
                        used_source_labels=bool(d["used_source_labels"]),
                    )
                # sor_k/sor_std_ratio mismatch -- stale entry from a different
                # configuration; fall through and recompute.

    n_raw = len(pts)
    pts_c, labels_c, n_rm = remove_statistical_outliers(pts, labels, k=sor_k, std_ratio=sor_std_ratio)
    pts_s, labels_s, n_soil, used_labels = remove_soil_pot(pts_c, labels_c, dataset, species)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = dict(n_raw=n_raw, pts_after_outlier=pts_c, n_outliers_removed=n_rm,
                            pts_after_soil=pts_s, n_soil_pot_removed=n_soil,
                            used_source_labels=used_labels, sor_k=sor_k, sor_std_ratio=sor_std_ratio)
        if labels_c is not None:
            save_kwargs["labels_after_outlier"] = labels_c
        if labels_s is not None:
            save_kwargs["labels_after_soil"] = labels_s
        # NOTE: np.savez_compressed silently APPENDS ".npz" to the filename
        # if it doesn't already end with ".npz" -- so the temp name must
        # itself end in ".npz" (e.g. "<stem>.tmp.npz", not "<stem>.npz.tmp")
        # or the later rename targets a file numpy never actually wrote.
        tmp = cache_file.with_name(cache_file.stem + ".tmp.npz")
        np.savez_compressed(tmp, **save_kwargs)
        tmp.replace(cache_file)

    return dict(n_raw=n_raw, pts_after_outlier=pts_c, labels_after_outlier=labels_c,
                n_outliers_removed=n_rm, pts_after_soil=pts_s, labels_after_soil=labels_s,
                n_soil_pot_removed=n_soil, used_source_labels=used_labels)


def _check_args(dataset, species, fn_name):
    if dataset is None or species is None:
        raise ValueError(f"{fn_name} requires dataset and species (needed to pick the "
                          f"correct soil-removal/up-axis rule per plan section 6)")


def _base_info(pipeline_id, n_raw, n_after_outlier, n_outliers_removed):
    return dict(pipeline_id=pipeline_id, n_raw=n_raw,
                n_after_outlier_removal=n_after_outlier,
                n_outliers_removed=n_outliers_removed)


# ---------------------------------------------------------------------------
# Pipelines (plan section 5)
# ---------------------------------------------------------------------------

def pipeline_P1(pts, labels=None, dataset=None, species=None, target_n=DEFAULT_TARGET_N,
                 sor_k=DEFAULT_SOR_K, sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                 shared_cache_dir=None, stem=None):
    """P1 -- DefRec/PointDA-10 style (Achituve et al., WACV 2021).
    Outlier removal -> align up-axis to +Z -> voxel(diag/200) for every
    cloud -> FPS 4096 -> centre + scale to unit CUBE. No soil/pot removal.

    shared_cache_dir/stem: see outlier_and_soil_removed()'s docstring --
    optional, reuses the (outlier-removal) result across pipelines/processes
    when given; None (default) reproduces the exact uncached behaviour."""
    _check_args(dataset, species, "pipeline_P1")
    shared = outlier_and_soil_removed(pts, labels, dataset, species, sor_k, sor_std_ratio,
                                       shared_cache_dir, stem)
    n_raw, n_rm = shared["n_raw"], shared["n_outliers_removed"]
    pts_c, labels_c = shared["pts_after_outlier"], shared["labels_after_outlier"]
    pts_a = align_up_axis(pts_c, dataset, species)
    pts_v, labels_v, voxel_size = voxel_downsample_diag200(pts_a, labels_c)
    pts_f, labels_f = fps_to_target(pts_v, labels_v, target_n, seed)
    pts_out, center, scale = normalize_unit_cube(pts_f.astype(np.float32))

    info = _base_info("P1", n_raw, len(pts_c), n_rm)
    info.update(n_after_soil_removal=len(pts_c), n_soil_pot_removed=0,
                voxel_size_mm=voxel_size, voxel_size_basis="diag/200_every_cloud",
                n_predecimated=len(pts_v), sampling_method="fps",
                n_final=len(pts_out), up_axis_aligned=True, norm_kind="unit_cube",
                labels_out=labels_f)
    return pts_out.astype(np.float32), center, scale, info


def pipeline_P2(pts, labels=None, dataset=None, species=None, target_n=DEFAULT_TARGET_N,
                 sor_k=DEFAULT_SOR_K, sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                 shared_cache_dir=None, stem=None):
    """P2 -- Pheno4D baselines / AGS-PlantSeg style (Salazar & Nalpantidis 2026).
    Outlier removal -> soil/pot removal -> RANDOM sampling to 4096 (no FPS)
    -> centre + unit sphere. No up-axis alignment step.

    shared_cache_dir/stem: see outlier_and_soil_removed()'s docstring."""
    _check_args(dataset, species, "pipeline_P2")
    shared = outlier_and_soil_removed(pts, labels, dataset, species, sor_k, sor_std_ratio,
                                       shared_cache_dir, stem)
    n_raw, n_rm = shared["n_raw"], shared["n_outliers_removed"]
    pts_c = shared["pts_after_outlier"]
    pts_s, labels_s, n_soil = shared["pts_after_soil"], shared["labels_after_soil"], shared["n_soil_pot_removed"]
    pts_r, labels_r = random_sample_to_target(pts_s, labels_s, target_n, seed)
    pts_out, center, scale = normalize_and_center(pts_r.astype(np.float32))

    info = _base_info("P2", n_raw, len(pts_c), n_rm)
    info.update(n_after_soil_removal=len(pts_s), n_soil_pot_removed=n_soil,
                voxel_size_mm=None, voxel_size_basis=None,
                n_predecimated=len(pts_s), sampling_method="random",
                n_final=len(pts_out), up_axis_aligned=False, norm_kind="unit_sphere",
                labels_out=labels_r)
    return pts_out.astype(np.float32), center, scale, info


def pipeline_P3(pts, labels=None, dataset=None, species=None, target_n=DEFAULT_TARGET_N,
                 sor_k=DEFAULT_SOR_K, sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                 shared_cache_dir=None, stem=None):
    """P3 -- Eff-3DPSeg style (Luo et al. 2022).
    Outlier removal -> soil/pot removal -> voxel at height-relative size
    (section 6) -> FPS 4096 -> centre + unit sphere.

    No explicit "align up-axis" step in this pipeline's own definition
    (section 5's table) -- height is read directly off the z column, which
    is CONFIRMED (Maize, Pheno4D) or TRUSTED (Tomato, TOMATO_UP_AXIS_ALIGN)
    up already for every dataset currently in scope, so this is numerically
    equivalent to aligning first; flagged here as an explicit inference
    rather than left implicit, since section 6's own height-relative-voxel
    definition presupposes alignment has already happened.

    shared_cache_dir/stem: see outlier_and_soil_removed()'s docstring."""
    _check_args(dataset, species, "pipeline_P3")
    shared = outlier_and_soil_removed(pts, labels, dataset, species, sor_k, sor_std_ratio,
                                       shared_cache_dir, stem)
    n_raw, n_rm = shared["n_raw"], shared["n_outliers_removed"]
    pts_c = shared["pts_after_outlier"]
    pts_s, labels_s, n_soil = shared["pts_after_soil"], shared["labels_after_soil"], shared["n_soil_pot_removed"]
    pts_v, labels_v, voxel_size = voxel_downsample_height_relative(pts_s, labels_s)
    pts_f, labels_f = fps_to_target(pts_v, labels_v, target_n, seed)
    pts_out, center, scale = normalize_and_center(pts_f.astype(np.float32))

    info = _base_info("P3", n_raw, len(pts_c), n_rm)
    info.update(n_after_soil_removal=len(pts_s), n_soil_pot_removed=n_soil,
                voxel_size_mm=voxel_size, voxel_size_basis="height/200_every_cloud",
                n_predecimated=len(pts_v), sampling_method="fps",
                n_final=len(pts_out), up_axis_aligned=False, norm_kind="unit_sphere",
                labels_out=labels_f)
    return pts_out.astype(np.float32), center, scale, info


def pipeline_P4(pts, labels=None, dataset=None, species=None, target_n=DEFAULT_TARGET_N,
                 sor_k=DEFAULT_SOR_K, sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                 shared_cache_dir=None, stem=None):
    """P4 -- our proposed v2.
    Outlier removal -> soil/pot removal -> align up-axis to +Z -> voxel at
    height-relative size (measured AFTER soil removal + alignment, per
    section 6) for every cloud -> FPS 4096 -> centre + unit sphere, with
    norm_scale/norm_center always saved (unlike P0, which never saved them
    for Crops3D -- see data/preprocessed/preprocessed_manifest.csv).

    shared_cache_dir/stem: see outlier_and_soil_removed()'s docstring."""
    _check_args(dataset, species, "pipeline_P4")
    shared = outlier_and_soil_removed(pts, labels, dataset, species, sor_k, sor_std_ratio,
                                       shared_cache_dir, stem)
    n_raw, n_rm = shared["n_raw"], shared["n_outliers_removed"]
    pts_c = shared["pts_after_outlier"]
    pts_s, labels_s, n_soil = shared["pts_after_soil"], shared["labels_after_soil"], shared["n_soil_pot_removed"]
    pts_a = align_up_axis(pts_s, dataset, species)
    pts_v, labels_v, voxel_size = voxel_downsample_height_relative(pts_a, labels_s)
    pts_f, labels_f = fps_to_target(pts_v, labels_v, target_n, seed)
    pts_out, center, scale = normalize_and_center(pts_f.astype(np.float32))

    info = _base_info("P4", n_raw, len(pts_c), n_rm)
    info.update(n_after_soil_removal=len(pts_s), n_soil_pot_removed=n_soil,
                voxel_size_mm=voxel_size, voxel_size_basis="height/200_every_cloud",
                n_predecimated=len(pts_v), sampling_method="fps",
                n_final=len(pts_out), up_axis_aligned=True, norm_kind="unit_sphere",
                labels_out=labels_f)
    return pts_out.astype(np.float32), center, scale, info


def pipeline_P4a(pts, labels=None, dataset=None, species=None, target_n=DEFAULT_TARGET_N,
                  sor_k=DEFAULT_SOR_K, sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                  shared_cache_dir=None, stem=None):
    """P4a -- same as P4 except NO soil/pot removal.

    shared_cache_dir/stem: see outlier_and_soil_removed()'s docstring."""
    _check_args(dataset, species, "pipeline_P4a")
    shared = outlier_and_soil_removed(pts, labels, dataset, species, sor_k, sor_std_ratio,
                                       shared_cache_dir, stem)
    n_raw, n_rm = shared["n_raw"], shared["n_outliers_removed"]
    pts_c, labels_c = shared["pts_after_outlier"], shared["labels_after_outlier"]
    pts_a = align_up_axis(pts_c, dataset, species)
    pts_v, labels_v, voxel_size = voxel_downsample_height_relative(pts_a, labels_c)
    pts_f, labels_f = fps_to_target(pts_v, labels_v, target_n, seed)
    pts_out, center, scale = normalize_and_center(pts_f.astype(np.float32))

    info = _base_info("P4a", n_raw, len(pts_c), n_rm)
    info.update(n_after_soil_removal=len(pts_c), n_soil_pot_removed=0,
                voxel_size_mm=voxel_size, voxel_size_basis="height/200_every_cloud",
                n_predecimated=len(pts_v), sampling_method="fps",
                n_final=len(pts_out), up_axis_aligned=True, norm_kind="unit_sphere",
                labels_out=labels_f)
    return pts_out.astype(np.float32), center, scale, info


def pipeline_P4b(pts, labels=None, dataset=None, species=None, target_n=DEFAULT_TARGET_N,
                  sor_k=DEFAULT_SOR_K, sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                  shared_cache_dir=None, stem=None):
    """P4b -- same as P4 except the OLD conditional sampling path: voxel at
    diag/200 ONLY IF the cloud has > 50,000 points (matching the original
    P0/preprocess_pointcloud default), else straight to FPS -- instead of
    P4's height-relative voxel for every cloud.

    shared_cache_dir/stem: see outlier_and_soil_removed()'s docstring."""
    _check_args(dataset, species, "pipeline_P4b")
    shared = outlier_and_soil_removed(pts, labels, dataset, species, sor_k, sor_std_ratio,
                                       shared_cache_dir, stem)
    n_raw, n_rm = shared["n_raw"], shared["n_outliers_removed"]
    pts_c = shared["pts_after_outlier"]
    pts_s, labels_s, n_soil = shared["pts_after_soil"], shared["labels_after_soil"], shared["n_soil_pot_removed"]
    pts_a = align_up_axis(pts_s, dataset, species)
    if len(pts_a) > 50000:
        pts_v, labels_v, voxel_size = voxel_downsample_diag200(pts_a, labels_s)
        voxel_basis = "old_conditional_diag/200_gt50k"
    else:
        pts_v, labels_v, voxel_size = pts_a, labels_s, None
        voxel_basis = "old_conditional_skipped_le50k"
    pts_f, labels_f = fps_to_target(pts_v, labels_v, target_n, seed)
    pts_out, center, scale = normalize_and_center(pts_f.astype(np.float32))

    info = _base_info("P4b", n_raw, len(pts_c), n_rm)
    info.update(n_after_soil_removal=len(pts_s), n_soil_pot_removed=n_soil,
                voxel_size_mm=voxel_size, voxel_size_basis=voxel_basis,
                n_predecimated=len(pts_v), sampling_method="fps",
                n_final=len(pts_out), up_axis_aligned=True, norm_kind="unit_sphere",
                labels_out=labels_f)
    return pts_out.astype(np.float32), center, scale, info


def pipeline_P4c(pts, labels=None, dataset=None, species=None, target_n=DEFAULT_TARGET_N,
                  sor_k=DEFAULT_SOR_K, sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                  shared_cache_dir=None, stem=None):
    """P4c -- same as P4 except NO up-axis alignment (align_up_axis() is not
    called at all, deliberately, for this ablation -- not merely passed a
    no-op flag).

    shared_cache_dir/stem: see outlier_and_soil_removed()'s docstring."""
    _check_args(dataset, species, "pipeline_P4c")
    shared = outlier_and_soil_removed(pts, labels, dataset, species, sor_k, sor_std_ratio,
                                       shared_cache_dir, stem)
    n_raw, n_rm = shared["n_raw"], shared["n_outliers_removed"]
    pts_c = shared["pts_after_outlier"]
    pts_s, labels_s, n_soil = shared["pts_after_soil"], shared["labels_after_soil"], shared["n_soil_pot_removed"]
    # align_up_axis deliberately NOT called here.
    pts_v, labels_v, voxel_size = voxel_downsample_height_relative(pts_s, labels_s)
    pts_f, labels_f = fps_to_target(pts_v, labels_v, target_n, seed)
    pts_out, center, scale = normalize_and_center(pts_f.astype(np.float32))

    info = _base_info("P4c", n_raw, len(pts_c), n_rm)
    info.update(n_after_soil_removal=len(pts_s), n_soil_pot_removed=n_soil,
                voxel_size_mm=voxel_size, voxel_size_basis="height/200_every_cloud",
                n_predecimated=len(pts_v), sampling_method="fps",
                n_final=len(pts_out), up_axis_aligned=False, norm_kind="unit_sphere",
                labels_out=labels_f)
    return pts_out.astype(np.float32), center, scale, info


def pipeline_P4d(pts, labels=None, dataset=None, species=None, target_n=DEFAULT_TARGET_N,
                  sor_k=DEFAULT_SOR_K, sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                  shared_cache_dir=None, stem=None):
    """P4d -- same as P4 except normalisation statistics (center/scale) are
    computed from the point cloud BEFORE soil removal ("soil included in the
    scale") and then applied to the final, post-pipeline output, instead of
    being recomputed from that final output the way every other pipeline
    does. Flagged explicit interpretation of the plan's one-line
    description: every OTHER step (soil removal, alignment, height-relative
    voxel sizing, FPS) still runs in the same order and in the same
    physical-unit space as P4 -- only the final centre/scale VALUES come
    from an earlier snapshot, so voxel-size semantics (which need physical
    units, not normalized ones) are unaffected by this ablation.

    shared_cache_dir/stem: see outlier_and_soil_removed()'s docstring."""
    _check_args(dataset, species, "pipeline_P4d")
    shared = outlier_and_soil_removed(pts, labels, dataset, species, sor_k, sor_std_ratio,
                                       shared_cache_dir, stem)
    n_raw, n_rm = shared["n_raw"], shared["n_outliers_removed"]
    pts_c = shared["pts_after_outlier"]
    _, pre_center, pre_scale = normalize_and_center(pts_c.astype(np.float32))  # soil-inclusive stats

    pts_s, labels_s, n_soil = shared["pts_after_soil"], shared["labels_after_soil"], shared["n_soil_pot_removed"]
    pts_a = align_up_axis(pts_s, dataset, species)
    pts_v, labels_v, voxel_size = voxel_downsample_height_relative(pts_a, labels_s)
    pts_f, labels_f = fps_to_target(pts_v, labels_v, target_n, seed)

    pts_out = (pts_f.astype(np.float32) - pre_center) / (pre_scale + 1e-8)
    center, scale = pre_center, pre_scale

    info = _base_info("P4d", n_raw, len(pts_c), n_rm)
    info.update(n_after_soil_removal=len(pts_s), n_soil_pot_removed=n_soil,
                voxel_size_mm=voxel_size, voxel_size_basis="height/200_every_cloud",
                n_predecimated=len(pts_v), sampling_method="fps",
                n_final=len(pts_out), up_axis_aligned=True,
                norm_kind="unit_sphere_soil_inclusive_stats",
                labels_out=labels_f)
    return pts_out.astype(np.float32), center, scale, info


_p4a_lookup_cache = None  # lazy, module-level: {(dataset, species, stem): (center(3,) float64, scale float)}


_P4A_CACHE_ROOT = _THIS_DIR.parent / "data" / "preprocessed_variants" / "P4a"


def _load_p4a_lookup():
    """
    Lazily builds a {(dataset, species, stem): (center, scale)} dict for
    pipeline_P4s, keyed the same way cache_path_for() in
    07_preprocess_variants.py builds a cache path (dataset/species/<stem>.npz)
    -- no dependence on absolute paths, which differ between the laptop and
    the cluster.

    Reads center/scale from each P4a .npz CACHE FILE directly, not from
    data/preprocessed_variants/P4a/preprocessed_manifest.csv. Found 2026-10-27
    while verifying P4s: P4a's own CSV manifest loses the last 1-2 bits of
    the float64 norm_center/norm_scale values relative to what's actually
    stored in P4a's own .npz files (confirmed on 257/531 rows, diffs of
    ~1e-17 to 1e-14 absolute -- an ordinary text round-trip artifact of
    writing/reading float64 via CSV, not a bug in P4a's pipeline itself).
    Reading the .npz directly avoids inheriting that same loss, so P4s's
    stored center/scale are bit-exactly equal to P4a's, not merely equal to
    full double precision.
    """
    global _p4a_lookup_cache
    if _p4a_lookup_cache is not None:
        return _p4a_lookup_cache
    manifest_path = _P4A_CACHE_ROOT / "preprocessed_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"pipeline_P4s requires the soil-kept P4a cache at {_P4A_CACHE_ROOT}, "
            f"but {manifest_path} does not exist. Generate P4a first (--pipeline P4a).")
    import pandas as pd
    df = pd.read_csv(manifest_path)
    lookup = {}
    for _, r in df.iterrows():
        stem = Path(str(r["source_filepath"])).stem
        dataset, species = str(r["dataset"]), str(r["species"])
        key = (dataset, species, stem)
        npz_path = _P4A_CACHE_ROOT / dataset / species / f"{stem}.npz"
        with np.load(npz_path) as d:
            center = np.array([d["norm_center_x"].item(), d["norm_center_y"].item(),
                                d["norm_center_z"].item()], dtype=np.float64)
            scale = float(d["norm_scale"].item())
        lookup[key] = (center, scale)
    _p4a_lookup_cache = lookup
    return lookup


def pipeline_P4s(pts, labels=None, dataset=None, species=None, target_n=DEFAULT_TARGET_N,
                  sor_k=DEFAULT_SOR_K, sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                  shared_cache_dir=None, stem=None):
    """P4s -- clean soil ablation requested 2026-10-27: do soil/pot removal,
    up-axis alignment, height-relative voxelling and FPS to target_n exactly
    as pipeline_P4 does, but DO NOT recompute norm_center/norm_scale from the
    post-cut, post-FPS cloud. Instead reuse the center/scale that pipeline_P4a
    (same cloud, soil/pot KEPT) already stored for this exact file, via
    _load_p4a_lookup() above. This isolates "soil removed from the geometry"
    from "soil removed from the normalisation statistics" -- P4 confounds the
    two (its own normalize_and_center() runs on the already-soil-removed,
    already-FPS'd cloud), so P4s is the control that holds normalisation
    fixed to the soil-kept reference frame while still cutting soil/pot out
    of the points themselves.

    `stem` is REQUIRED (unlike the other pipelines, where it's only used for
    shared_cache_dir) -- it is the P4a lookup key together with dataset/species.
    Raises KeyError if this exact (dataset, species, stem) has no P4a entry,
    rather than silently falling back to a self-computed normalisation.

    shared_cache_dir: see outlier_and_soil_removed()'s docstring."""
    _check_args(dataset, species, "pipeline_P4s")
    if stem is None:
        raise ValueError("pipeline_P4s requires `stem` to look up P4a's stored center/scale "
                          "for this same cloud.")
    p4a_lookup = _load_p4a_lookup()
    key = (dataset, species, stem)
    if key not in p4a_lookup:
        raise KeyError(f"No P4a manifest entry found for {key} -- cannot borrow its "
                        f"norm_center/norm_scale. Regenerate P4a first.")
    ext_center, ext_scale = p4a_lookup[key]

    shared = outlier_and_soil_removed(pts, labels, dataset, species, sor_k, sor_std_ratio,
                                       shared_cache_dir, stem)
    n_raw, n_rm = shared["n_raw"], shared["n_outliers_removed"]
    pts_c = shared["pts_after_outlier"]
    pts_s, labels_s, n_soil = shared["pts_after_soil"], shared["labels_after_soil"], shared["n_soil_pot_removed"]
    pts_a = align_up_axis(pts_s, dataset, species)
    pts_v, labels_v, voxel_size = voxel_downsample_height_relative(pts_a, labels_s)
    pts_f, labels_f = fps_to_target(pts_v, labels_v, target_n, seed)

    pts_out = (pts_f.astype(np.float32) - ext_center.astype(np.float32)) / (ext_scale + 1e-8)
    center, scale = ext_center, ext_scale

    info = _base_info("P4s", n_raw, len(pts_c), n_rm)
    info.update(n_after_soil_removal=len(pts_s), n_soil_pot_removed=n_soil,
                voxel_size_mm=voxel_size, voxel_size_basis="height/200_every_cloud",
                n_predecimated=len(pts_v), sampling_method="fps",
                n_final=len(pts_out), up_axis_aligned=True,
                norm_kind="unit_sphere_from_P4a",
                labels_out=labels_f)
    return pts_out.astype(np.float32), center, scale, info


PIPELINES = {
    "P1": pipeline_P1,
    "P2": pipeline_P2,
    "P3": pipeline_P3,
    "P4": pipeline_P4,
    "P4a": pipeline_P4a,
    "P4b": pipeline_P4b,
    "P4c": pipeline_P4c,
    "P4d": pipeline_P4d,
    "P4s": pipeline_P4s,
}
