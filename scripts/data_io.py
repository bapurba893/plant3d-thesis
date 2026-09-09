"""
data_io.py
Core loaders for Crops3D (PLY) and Pheno4D (XYZ) point clouds.

Both loaders return a plain (N, 3) numpy array of XYZ coordinates so that
downstream code (stats, visualization, augmentation) doesn't care which
dataset a point cloud came from.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import re


# ---------------------------------------------------------------------------
# Crops3D (PLY)
# ---------------------------------------------------------------------------

def load_crops3d_ply(filepath: str) -> np.ndarray:
    """
    Load a single Crops3D PLY file. Returns (N, 3) float32 XYZ array.
    RGB, if present, is dropped here (add it back in load if you need color).
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(filepath))
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.size == 0:
            raise ValueError("open3d returned 0 points, falling back to plyfile")
        return pts
    except Exception:
        # Fallback: parse with plyfile directly (handles some PLY variants
        # open3d is picky about)
        from plyfile import PlyData
        ply = PlyData.read(str(filepath))
        v = ply["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        return pts


def scan_crops3d_directory(root: str, species: list[str] = ("Tomato", "Maize")) -> pd.DataFrame:
    """
    Walk the Crops3D directory and build a manifest of every PLY file found
    for the requested species, with point counts.

    Expects: root/<Species>/*.ply   (matches the dataset's native layout)
    """
    root = Path(root)
    rows = []
    for sp in species:
        sp_dir = root / sp
        if not sp_dir.exists():
            print(f"[warn] {sp_dir} does not exist — skipping {sp}")
            continue
        for f in sorted(sp_dir.glob("*.ply")):
            try:
                pts = load_crops3d_ply(f)
                rows.append({
                    "dataset": "Crops3D",
                    "species": sp,
                    "filepath": str(f),
                    "n_points": len(pts),
                })
            except Exception as e:
                print(f"[error] failed to load {f}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pheno4D (XYZ)
# ---------------------------------------------------------------------------

def load_pheno4d_xyz(filepath: str):
    """
    Load a single Pheno4D .xyz file.
    Returns (points, labels) where:
      points: (N, 3) float32 XYZ array
      labels: (N,) or (N, 2) int array if the file is annotated, else None
    Handles 3-col (unlabeled), 4-col (tomato labeled), 5-col (maize labeled,
    two label conventions) files automatically based on column count.
    """
    arr = np.loadtxt(filepath, dtype=np.float64)
    if arr.ndim == 1:  # single-point file edge case
        arr = arr.reshape(1, -1)
    pts = arr[:, :3].astype(np.float32)
    labels = arr[:, 3:].astype(np.int32) if arr.shape[1] > 3 else None
    return pts, labels


_PHENO4D_FNAME_RE = re.compile(r"^([MT])(\d+)_(\d+)(_a)?\.(?:xyz|txt)$", re.IGNORECASE)


def scan_pheno4d_directory(root: str) -> pd.DataFrame:
    """
    Walk the Pheno4D directory and build a manifest of every point cloud
    file, with species, plant id, scan date, annotation status, and point
    count.

    Expects: root/<PlantFolder>/*.txt   where filenames look like
             M01_0313_a.txt (annotated) or T01_0305.txt (unannotated).
    Note: the official Pheno4D.zip (ipb.uni-bonn.de) ships plain-text files
    with a .txt extension despite the space-separated XYZ[+labels] content —
    the dataset is NOT named *.xyz on disk. Both extensions are matched here
    for portability.
    """
    root = Path(root)
    rows = []
    candidates = list(root.rglob("*.txt")) + list(root.rglob("*.xyz"))
    for f in sorted(candidates):
        m = _PHENO4D_FNAME_RE.match(f.name)
        if not m:
            print(f"[warn] filename didn't match expected Pheno4D pattern: {f.name}")
            continue
        species_code, plant_num, scan_date, annotated_flag = m.groups()
        species = "Maize" if species_code.upper() == "M" else "Tomato"
        try:
            pts, labels = load_pheno4d_xyz(f)
        except Exception as e:
            print(f"[error] failed to load {f}: {e}")
            continue
        rows.append({
            "dataset": "Pheno4D",
            "species": species,
            "plant_id": f"{species_code.upper()}{plant_num}",
            "scan_date": scan_date,
            "is_annotated": annotated_flag is not None,
            "filepath": str(f),
            "n_points": len(pts),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("This module provides loaders only — import it from the numbered scripts.")
