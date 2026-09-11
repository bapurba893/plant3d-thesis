"""
data_io.py
Core loaders for Crops3D (PLY) and Pheno4D (XYZ) point clouds.

Both loaders return (points, labels) where points is a plain (N, 3) numpy
array of XYZ coordinates and labels is an int array (or None if the source
file carries no annotation), so downstream code (stats, visualization,
augmentation) doesn't care which dataset a point cloud came from.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import re


# ---------------------------------------------------------------------------
# Crops3D (PLY)
# ---------------------------------------------------------------------------

# Property names CloudCompare/plyfile have been observed to use for the
# per-point organ-category id in Crops3D PLYs. "scalar_sf" is what's actually
# in the released files (confirmed 2026-09-11 by reading a raw PLY header);
# the others are defensive fallbacks in case a differently-exported file
# shows up.
_CROPS3D_LABEL_PROPERTY_CANDIDATES = (
    "scalar_sf", "sf", "scalar_Sf", "Sf", "scalar_Segment", "label",
)


def load_crops3d_ply(filepath: str):
    """
    Load a single Crops3D PLY file. Returns (pts, labels):
      pts: (N, 3) float32 XYZ array.
      labels: (N,) int32 array holding the PLY's organ-category scalar
              field (see _CROPS3D_LABEL_PROPERTY_CANDIDATES), or None if
              the file has none of those properties.

    NOTE on label semantics: there is no published id->organ-name table for
    Crops3D (checked the paper's Table 2, the clawCa/Crops3D and
    harpreetsahota204/crops3d_to_fiftyone GitHub repos, and the Voxel51 HF
    dataset card -- none give a numeric mapping, and it likely isn't even
    consistent in meaning between species). We therefore keep the raw
    integer id as the class label rather than renaming it -- L_seg only
    needs consistent per-species integer ids to train, not human-readable
    names. Empirical RGB fingerprinting (2026-09-11, 15 files/species) gives
    high confidence for two ids: Maize id 0 = soil (clearly brown), and each
    species' highest-point-count green id = leaf. The remaining ids are
    unconfirmed -- do not assume a name for them without re-checking.

    RGB, if present, is dropped here (add it back in load if you need color).
    """
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(filepath))
        v = ply["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        prop_names = [p.name for p in v.properties]
        label_prop = next(
            (c for c in _CROPS3D_LABEL_PROPERTY_CANDIDATES if c in prop_names), None
        )
        labels = (np.rint(np.asarray(v[label_prop])).astype(np.int32)
                  if label_prop is not None else None)
        return pts, labels
    except Exception:
        # Fallback: open3d handles some PLY variants plyfile is picky about,
        # but read_point_cloud only exposes xyz/rgb/normals -- never custom
        # scalar fields -- so labels are unavailable via this path.
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(filepath))
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.size == 0:
            raise ValueError(f"both plyfile and open3d failed to load points from {filepath}")
        print(f"[warn] {filepath}: loaded via open3d fallback -- organ labels "
              f"unavailable for this file")
        return pts, None


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
                pts, labels = load_crops3d_ply(f)
                rows.append({
                    "dataset": "Crops3D",
                    "species": sp,
                    "filepath": str(f),
                    "n_points": len(pts),
                    "is_annotated": labels is not None,
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
