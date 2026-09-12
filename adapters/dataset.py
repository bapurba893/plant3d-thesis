"""
dataset.py

PyTorch Dataset adapters that plug our Stage-0 preprocessed Crops3D/Pheno4D
.npz cache into DefRec_and_PCM's training loop (which expects a Dataset
returning ((N, 3) float32 points, int label)).

PlantSpeciesDataset: classification-only (species Tomato=0/Maize=1). Used
for row C1's original interim result and for target (Pheno4D) domain-gap
classification eval, where no segmentation wiring exists yet.

PlantClsSegDataset: joint species classification + per-point organ
segmentation, for the full L_cls + L_seg spec (every strategy-table row).
Requires the per-point 'labels' key in the .npz cache -- see CLAUDE.md
"Per-point Crops3D organ segmentation labels backfilled" for how Crops3D's
scalar_sf field got there. SEG_NUM_CLASSES gives the confirmed per-species
class count (the numeric ids are NOT confirmed to mean the same organ
across species, so segmentation is always per-species, never a shared
label space -- see adapters/models.py).

Both datasets take an `augment_mode` ("all" [default] or "ln_only") that
selects which augmentation pipeline `augment=True` applies -- "all" is
the full G-R+G-S+L-N+L-D pipeline every row through C3 used; "ln_only" is
L-N (Gaussian jitter) alone, for row C4 (DGCNN, DA-A, L-N), isolating
DGCNN's noise weakness per CLAUDE.md's augmentation taxonomy. See
scripts/augmentations.py's compose_pipeline_ln_only docstring.

PlantClsSegDataset also takes optional `label_transform`/`num_classes` for
row C5 (DGCNN, DA-O): Pheno4D's own annotated files carry a different raw
per-point label scheme than Crops3D's (see PHENO4D_SEG_NUM_CLASSES and
pheno4d_collapse_organ_labels below), so C5 passes those in rather than
relying on the Crops3D-shaped defaults every prior row used.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from augmentations import (  # noqa: E402
    compose_pipeline, pad_if_needed3d,
    compose_pipeline_with_labels, pad_if_needed3d_with_labels,
    compose_pipeline_ln_only, compose_pipeline_ln_only_with_labels,
)

# augment_mode -> (points-only pipeline, points+labels pipeline). "all" is
# the default used by every row so far (C1/A1/C2/C3); "ln_only" is for
# row C4 (DGCNN, DA-A, L-N), isolating DGCNN's noise weakness per
# CLAUDE.md's augmentation taxonomy -- see scripts/augmentations.py's
# compose_pipeline_ln_only docstring.
_AUGMENT_PIPELINES = {
    "all": (compose_pipeline, compose_pipeline_with_labels),
    "ln_only": (compose_pipeline_ln_only, compose_pipeline_ln_only_with_labels),
}

SPECIES_TO_IDX = {"Tomato": 0, "Maize": 1}
IDX_TO_SPECIES = {v: k for k, v in SPECIES_TO_IDX.items()}
TARGET_N = 4096

# Confirmed via a full scan of all 308 cached Crops3D files (2026-09-11) --
# see CLAUDE.md. Not necessarily stable if more raw files are added later
# without re-scanning.
SEG_NUM_CLASSES = {"Tomato": 3, "Maize": 6}

# Pheno4D's OWN per-point organ labels (used only for row C5, DA-O -- see
# CLAUDE.md "point-wise soil/stem/leaf labels available for evaluation and
# the Oracle upper-bound only"). This is a DIFFERENT label space than
# Crops3D's SEG_NUM_CLASSES above -- do not conflate the two; there is no
# shared semantics or class-id correspondence between them, they're just two
# separate per-domain segmentation tasks.
#
# Raw label layout (per CLAUDE.md's Datasets section): Tomato annotated
# files carry one label column, Maize carry two ("label1,label2"). A full
# scan of all 126 annotated Pheno4D files (2026-09-12, both the adaptation
# pool's 90 and the held-out eval's 36) found the raw ids are NOT a flat
# soil/stem/leaf 3-class scheme -- they grow over the growing season (up to
# 42 distinct ids for Tomato, 5 for Maize's label1), consistent with
# per-leaf INSTANCE ids being folded into the same column (id 0 dominates
# every file by a wide margin in point count, consistent with "soil"; id 1
# is the next-largest and far bigger than any higher id in both species,
# consistent with "stem"; every id >= 2 is a much smaller, roughly
# similarly-sized class, consistent with individual leaf instances).
# INFERENCE, not confirmed against Pheno4D's original publication in this
# session: raw id 0 -> soil, id 1 -> stem, id >= 2 -> collapsed into a
# single "leaf" class, giving a 3-class organ scheme for both species
# comparable in kind (not in exact class count or id semantics) to
# Crops3D's per-species organ segmentation. For Maize's two label columns,
# label1 (the first of the two) is used and label2 is unused -- CLAUDE.md's
# own text doesn't say which is the organ-semantic column and which is a
# leaf-instance-only column, and both showed the same soil-dominant/
# stem-next/many-small-rest histogram shape, so this is a default choice,
# not a verified one. See `pheno4d_collapse_organ_labels` below.
PHENO4D_SEG_NUM_CLASSES = {"Tomato": 3, "Maize": 3}


def pheno4d_collapse_organ_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Collapses Pheno4D's raw per-point label column(s) into the 3-class
    {0: soil, 1: stem, 2: leaf} scheme documented above. Maize's (N, 2)
    labels use column 0 only; Tomato's (N,) or (N, 1) labels are used as-is.
    Flagged as an inference, not a confirmed id->organ mapping -- see the
    PHENO4D_SEG_NUM_CLASSES docstring above."""
    raw = raw_labels[:, 0] if raw_labels.ndim == 2 else raw_labels.reshape(-1)
    return np.where(raw == 0, 0, np.where(raw == 1, 1, 2)).astype(np.int64)


def _strip_leading_dotdot(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/")
    while rel_path.startswith("../"):
        rel_path = rel_path[3:]
    return rel_path


def _load_manifest_lookup(manifest_csv: Path) -> dict:
    """source_filepath (normalized, no leading ../) -> cache_path (absolute)"""
    df = pd.read_csv(manifest_csv)
    lookup = {}
    for _, row in df.iterrows():
        key = _strip_leading_dotdot(row["source_filepath"])
        cache_path = _REPO_ROOT / _strip_leading_dotdot(row["cache_path"])
        lookup[key] = cache_path
    return lookup


class PlantSpeciesDataset(Dataset):
    """
    Wraps a split CSV (columns: dataset,species,filepath,...) against the
    preprocessed .npz cache. Returns (points [N,3] float32, label int64).
    """

    def __init__(self, split_csv: str, manifest_csv: str, augment: bool = False,
                 target_n: int = TARGET_N, seed: int = 0, augment_mode: str = "all"):
        split_csv = Path(split_csv)
        manifest_csv = Path(manifest_csv)
        if not split_csv.is_absolute():
            split_csv = _REPO_ROOT / split_csv
        if not manifest_csv.is_absolute():
            manifest_csv = _REPO_ROOT / manifest_csv

        manifest_lookup = _load_manifest_lookup(manifest_csv)
        split_df = pd.read_csv(split_csv)

        self.samples = []  # list of (cache_path, label)
        missing = 0
        for _, row in split_df.iterrows():
            key = _strip_leading_dotdot(row["filepath"])
            cache_path = manifest_lookup.get(key)
            if cache_path is None or not cache_path.exists():
                missing += 1
                continue
            label = SPECIES_TO_IDX[row["species"]]
            self.samples.append((cache_path, label))
        if missing:
            print(f"[warn] {missing} rows in {split_csv.name} had no matching "
                  f"preprocessed cache entry and were skipped")
        if not self.samples:
            raise RuntimeError(f"No samples resolved for {split_csv}")

        self.augment = augment
        self.target_n = target_n
        self.rng = np.random.default_rng(seed)
        self._compose_pipeline, _ = _AUGMENT_PIPELINES[augment_mode]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache_path, label = self.samples[idx]
        with np.load(cache_path) as d:
            pts = d["points"].astype(np.float32)

        if self.augment:
            pts = self._compose_pipeline(pts, rng=self.rng)
            pts = pad_if_needed3d(pts, self.target_n, rng=self.rng)

        # compose_pipeline/crop can reorder or drop points -- enforce exact N
        if len(pts) != self.target_n:
            pts = pad_if_needed3d(pts, self.target_n, rng=self.rng)[:self.target_n]

        return pts.astype(np.float32), np.int64(label)

    def class_counts(self):
        labels = [l for _, l in self.samples]
        return {IDX_TO_SPECIES[k]: labels.count(k) for k in set(labels)}


class PlantClsSegDataset(Dataset):
    """
    Wraps a split CSV against the preprocessed .npz cache, requiring the
    per-point 'labels' key (organ segmentation). Returns
    (points [N,3] float32, species_label int64, seg_labels [N] int64).
    Samples whose cache entry has no 'labels' key are skipped (e.g. the
    rare file that fell back to the open3d loader -- see
    data_io.py::load_crops3d_ply).

    `label_transform` (optional): applied to each sample's raw cached
    'labels' array right after loading, before augmentation/padding --
    e.g. `pheno4d_collapse_organ_labels` for row C5 (DA-O), which needs a
    different raw-id-to-class-id mapping than Crops3D's labels (already
    flat per-species integer ids, no transform needed). Default None is a
    no-op, so every existing caller (C1-C4, all Crops3D-based) is
    unaffected.

    `num_classes` (optional): per-species class-count dict used by
    `seg_class_histogram` (and by callers building segmentation heads/
    class weights) instead of the module-level Crops3D-specific
    SEG_NUM_CLASSES -- e.g. PHENO4D_SEG_NUM_CLASSES for row C5. Default
    None falls back to SEG_NUM_CLASSES, preserving old behavior.
    """

    def __init__(self, split_csv: str, manifest_csv: str, augment: bool = False,
                 target_n: int = TARGET_N, seed: int = 0, augment_mode: str = "all",
                 label_transform=None, num_classes: dict = None):
        split_csv = Path(split_csv)
        manifest_csv = Path(manifest_csv)
        if not split_csv.is_absolute():
            split_csv = _REPO_ROOT / split_csv
        if not manifest_csv.is_absolute():
            manifest_csv = _REPO_ROOT / manifest_csv

        manifest_lookup = _load_manifest_lookup(manifest_csv)
        split_df = pd.read_csv(split_csv)

        self.samples = []  # list of (cache_path, species_label)
        missing, no_seg_labels = 0, 0
        for _, row in split_df.iterrows():
            key = _strip_leading_dotdot(row["filepath"])
            cache_path = manifest_lookup.get(key)
            if cache_path is None or not cache_path.exists():
                missing += 1
                continue
            with np.load(cache_path) as d:
                if "labels" not in d:
                    no_seg_labels += 1
                    continue
            label = SPECIES_TO_IDX[row["species"]]
            self.samples.append((cache_path, label))
        if missing:
            print(f"[warn] {missing} rows in {split_csv.name} had no matching "
                  f"preprocessed cache entry and were skipped")
        if no_seg_labels:
            print(f"[warn] {no_seg_labels} rows in {split_csv.name} had a cache "
                  f"entry with no 'labels' (segmentation) key and were skipped")
        if not self.samples:
            raise RuntimeError(f"No samples resolved for {split_csv}")

        self.augment = augment
        self.target_n = target_n
        self.rng = np.random.default_rng(seed)
        self.label_transform = label_transform
        self.num_classes = num_classes if num_classes is not None else SEG_NUM_CLASSES
        _, self._compose_pipeline_with_labels = _AUGMENT_PIPELINES[augment_mode]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache_path, species_label = self.samples[idx]
        with np.load(cache_path) as d:
            pts = d["points"].astype(np.float32)
            seg_labels = d["labels"]
        if self.label_transform is not None:
            seg_labels = self.label_transform(seg_labels)
        seg_labels = seg_labels.astype(np.int64)

        if self.augment:
            pts, seg_labels = self._compose_pipeline_with_labels(pts, seg_labels, rng=self.rng)
        pts, seg_labels = pad_if_needed3d_with_labels(pts, seg_labels, self.target_n, rng=self.rng)

        # compose_pipeline/crop can reorder or drop points -- enforce exact N
        if len(pts) != self.target_n:
            pts, seg_labels = pad_if_needed3d_with_labels(pts, seg_labels, self.target_n, rng=self.rng)
            pts, seg_labels = pts[:self.target_n], seg_labels[:self.target_n]

        return pts.astype(np.float32), np.int64(species_label), seg_labels.astype(np.int64)

    def class_counts(self):
        labels = [l for _, l in self.samples]
        return {IDX_TO_SPECIES[k]: labels.count(k) for k in set(labels)}

    def seg_class_histogram(self, species: str) -> np.ndarray:
        """Point-level class counts for one species' segmentation labels,
        pooled across every training sample of that species -- used to
        build L_seg's w_c = (f_c+eps)^-1 per-class weights."""
        n_classes = self.num_classes[species]
        species_idx = SPECIES_TO_IDX[species]
        hist = np.zeros(n_classes, dtype=np.int64)
        for cache_path, label in self.samples:
            if label != species_idx:
                continue
            with np.load(cache_path) as d:
                seg_labels = d["labels"]
            if self.label_transform is not None:
                seg_labels = self.label_transform(seg_labels)
            hist += np.bincount(seg_labels, minlength=n_classes)[:n_classes]
        return hist
