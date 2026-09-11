"""
inspect_crops3d_sf.py

One-off diagnostic script -- NOT part of the regular pipeline (01-05).
Run this on the LAPTOP where the raw Crops3D archive was extracted (the
raw PLY files are not checked into this repo -- see docs/00_README_download.md).

Usage:
    python inspect_crops3d_sf.py --root Crops3D_raw

Expects --root to be the directory you passed to Expand-Archive (the one
that directly contains Crops3D/, Crops3D_10k/, Crops3D_IS/, etc. as
top-level subfolders per docs/00_README_download.md).

What it does:
  1. Lists any non-.ply files anywhere under --root (README/txt/json/csv) and
     prints small ones in full -- in case the archive ships its own label key
     that never made it into the paper/GitHub writeups.
  2. For Tomato and Maize, scans up to N_FILES_PER_SPECIES files under
     Crops3D/<Species>/ and reports:
       - the union of unique scalar_sf values seen across all scanned files
       - per-file unique value count (so we can see whether any single file
         actually contains the full category set)
       - for each sf value, the mean R,G,B (0-255) and point-count fraction,
         pooled across all scanned files -- to fingerprint which value is
         soil/stem/leaf/fruit/etc by color.
No writes, no dependency beyond plyfile/numpy (pip install plyfile numpy if needed).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    print("pip install plyfile", file=sys.stderr)
    raise

N_FILES_PER_SPECIES = 15  # bump if this machine can spare the time


def find_sf_property_name(vertex_props):
    names = [p.name for p in vertex_props]
    for cand in ("scalar_sf", "sf", "scalar_Sf", "Sf", "scalar_Segment", "label"):
        if cand in names:
            return cand
    # fallback: any property beyond x,y,z,red,green,blue,alpha
    known = {"x", "y", "z", "red", "green", "blue", "alpha",
             "nx", "ny", "nz"}
    extra = [n for n in names if n not in known]
    return extra[0] if extra else None


def inspect_species(root: Path, species: str):
    sp_dir = root / "Crops3D" / species
    files = sorted(sp_dir.glob("*.ply"))[:N_FILES_PER_SPECIES]
    if not files:
        print(f"[warn] no files found under {sp_dir}")
        return

    print(f"\n=== {species}: scanning {len(files)} files from {sp_dir} ===")

    union_values = set()
    per_file_counts = []
    # value -> [sum_r, sum_g, sum_b, n_points]
    color_by_value = defaultdict(lambda: np.zeros(4, dtype=np.float64))

    sf_name = None
    for f in files:
        try:
            ply = PlyData.read(str(f))
        except Exception as e:
            print(f"  [error] {f.name}: {e}")
            continue
        v = ply["vertex"]
        if sf_name is None:
            sf_name = find_sf_property_name(v.properties)
            print(f"  detected label property name: {sf_name!r} "
                  f"(all properties: {[p.name for p in v.properties]})")
            if sf_name is None:
                print("  [error] could not find a label property -- aborting this species")
                return
        sf = np.asarray(v[sf_name])
        uniq, counts = np.unique(sf, return_counts=True)
        union_values.update(uniq.tolist())
        per_file_counts.append((f.name, sorted(uniq.tolist())))

        has_rgb = all(c in v.data.dtype.names for c in ("red", "green", "blue"))
        if has_rgb:
            r = np.asarray(v["red"], dtype=np.float64)
            g = np.asarray(v["green"], dtype=np.float64)
            b = np.asarray(v["blue"], dtype=np.float64)
            for val in uniq:
                mask = sf == val
                color_by_value[float(val)] += np.array([
                    r[mask].sum(), g[mask].sum(), b[mask].sum(), mask.sum()
                ])

    print(f"\n  Union of sf values across all {len(files)} scanned files: {sorted(union_values)}")
    print("  Per-file unique value sets:")
    for name, vals in per_file_counts:
        print(f"    {name}: {vals}")

    if color_by_value:
        print("\n  Mean RGB per sf value (pooled across all scanned files):")
        for val in sorted(color_by_value):
            s = color_by_value[val]
            n = s[3]
            if n == 0:
                continue
            r, g, b = s[0] / n, s[1] / n, s[2] / n
            frac_files = sum(1 for _, vals in per_file_counts if val in vals) / len(per_file_counts)
            print(f"    sf={val:g}: mean RGB=({r:6.1f},{g:6.1f},{b:6.1f})  "
                  f"n_points_total={int(n):>9d}  present_in={frac_files*100:.0f}% of scanned files")


def scan_for_side_files(root: Path):
    print(f"=== Non-.ply files under {root} (looking for a bundled label key) ===")
    exts = (".txt", ".json", ".csv", ".md", ".yaml", ".yml")
    found = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    if not found:
        print("  (none found)")
        return
    for p in sorted(found):
        size = p.stat().st_size
        print(f"  {p}  ({size} bytes)")
        if size < 4000:
            print("  ---content---")
            try:
                print(p.read_text(errors="replace"))
            except Exception as e:
                print(f"  [could not read: {e}]")
            print("  ---end---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                     help="Dir containing Crops3D/, Crops3D_10k/, Crops3D_IS/ as subfolders")
    args = ap.parse_args()
    root = Path(args.root)

    scan_for_side_files(root)
    for species in ("Tomato", "Maize"):
        inspect_species(root, species)


if __name__ == "__main__":
    main()
