"""
preprocessing_pipeline.py

Block PP: one preprocessing function, `preprocess_cloud`, with a user switch
`remove_soil` that selects between the two pipelines Block PP compares most
directly:

    remove_soil=True  -> reproduces P4  (soil/pot removed)
    remove_soil=False -> reproduces P4a (soil/pot kept)

Both settings run the same code path. The only branch is which point set
(after outlier removal, or after outlier + soil/pot removal) goes into the
remaining steps:

    outlier removal
    -> soil/pot removal                      (remove_soil=True only)
    -> align up-axis to +Z
    -> voxel at height-relative size         (height measured AFTER the
                                              step above, so after soil
                                              removal when it is on)
    -> farthest-point sampling to target_n (4096)
    -> centre + scale to unit sphere         (always computed from the final
                                              cloud, so after soil removal
                                              when it is on)

Soil/pot removal rule is the one in preprocessing_variants.remove_soil_pot:
Crops3D Maize drops source label ids {0 soil, 5 pot}; Crops3D Tomato is a
no-op (no soil in those clouds); Pheno4D (both species) uses the label-free
horizontal-slab cut, never the labels.

Does not modify scripts/preprocessing_variants.py -- imports its step helpers
unchanged. scripts/12_check_single_pipeline.py checks this function against
the existing data/preprocessed_variants/P4 and P4a caches.
"""
import numpy as np

from preprocessing import normalize_and_center
from preprocessing_variants import (
    DEFAULT_SEED,
    DEFAULT_SOR_K,
    DEFAULT_SOR_STD_RATIO,
    DEFAULT_TARGET_N,
    align_up_axis,
    fps_to_target,
    outlier_and_soil_removed,
    voxel_downsample_height_relative,
)


def preprocess_cloud(points, labels=None, dataset=None, species=None, remove_soil=True,
                     target_n=DEFAULT_TARGET_N, sor_k=DEFAULT_SOR_K,
                     sor_std_ratio=DEFAULT_SOR_STD_RATIO, seed=DEFAULT_SEED,
                     shared_cache_dir=None, stem=None):
    """
    Preprocess one raw cloud. `dataset` ("Crops3D"/"Pheno4D") and `species`
    ("Tomato"/"Maize") are required: they pick the soil/pot-removal rule.
    `labels` is required for Crops3D Maize when remove_soil=True (the cut
    uses its source labels); optional otherwise, and carried through to the
    output when given.

    Returns (pts_out (target_n,3) float32, norm_center (3,), norm_scale float,
    info dict) -- the same contract as the preprocessing_variants.pipeline_PX
    functions, including info["labels_out"] and info["pipeline_id"] ("P4" when
    remove_soil=True, "P4a" when False), so 07_preprocess_variants.py can
    write its output exactly as it does for those pipelines.

    shared_cache_dir/stem: optional outlier-removal cache, see
    preprocessing_variants.outlier_and_soil_removed().
    """
    if dataset is None or species is None:
        raise ValueError("preprocess_cloud requires dataset and species (needed to pick the "
                         "soil/pot-removal rule)")
    remove_soil = bool(remove_soil)

    shared = outlier_and_soil_removed(points, labels, dataset, species, sor_k, sor_std_ratio,
                                      shared_cache_dir, stem)
    pts_c = shared["pts_after_outlier"]
    if remove_soil:
        pts_s, labels_s = shared["pts_after_soil"], shared["labels_after_soil"]
        n_soil = shared["n_soil_pot_removed"]
    else:
        pts_s, labels_s = pts_c, shared["labels_after_outlier"]
        n_soil = 0

    pts_a = align_up_axis(pts_s, dataset, species)
    pts_v, labels_v, voxel_size = voxel_downsample_height_relative(pts_a, labels_s)
    pts_f, labels_f = fps_to_target(pts_v, labels_v, target_n, seed)
    pts_out, center, scale = normalize_and_center(pts_f.astype(np.float32))

    info = dict(pipeline_id="P4" if remove_soil else "P4a",
                n_raw=shared["n_raw"],
                n_after_outlier_removal=len(pts_c),
                n_outliers_removed=shared["n_outliers_removed"],
                n_after_soil_removal=len(pts_s), n_soil_pot_removed=n_soil,
                voxel_size_mm=voxel_size, voxel_size_basis="height/200_every_cloud",
                n_predecimated=len(pts_v), sampling_method="fps",
                n_final=len(pts_out), up_axis_aligned=True, norm_kind="unit_sphere",
                remove_soil=remove_soil,
                labels_out=labels_f)
    return pts_out.astype(np.float32), center, scale, info
