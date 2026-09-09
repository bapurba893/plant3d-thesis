"""
dataset.py

PyTorch Dataset adapters that plug our Stage-0 preprocessed Crops3D/Pheno4D
.npz cache into DefRec_and_PCM's training loop (which expects a Dataset
returning ((N, 3) float32 points, int label)).

Row C1 (DGCNN, DA-0, ALL) is classification-only: Crops3D lacks point-wise
segmentation labels (Crops3D_IS was not downloaded in Stage 0 -- see project
memory / task tracker), so the label here is species (Tomato=0, Maize=1),
known from the source directory structure for both datasets.
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

from augmentations import compose_pipeline, pad_if_needed3d  # noqa: E402

SPECIES_TO_IDX = {"Tomato": 0, "Maize": 1}
IDX_TO_SPECIES = {v: k for k, v in SPECIES_TO_IDX.items()}
TARGET_N = 4096


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
                 target_n: int = TARGET_N, seed: int = 0):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache_path, label = self.samples[idx]
        with np.load(cache_path) as d:
            pts = d["points"].astype(np.float32)

        if self.augment:
            pts = compose_pipeline(pts, rng=self.rng)
            pts = pad_if_needed3d(pts, self.target_n, rng=self.rng)

        # compose_pipeline/crop can reorder or drop points -- enforce exact N
        if len(pts) != self.target_n:
            pts = pad_if_needed3d(pts, self.target_n, rng=self.rng)[:self.target_n]

        return pts.astype(np.float32), np.int64(label)

    def class_counts(self):
        labels = [l for _, l in self.samples]
        return {IDX_TO_SPECIES[k]: labels.count(k) for k in set(labels)}
