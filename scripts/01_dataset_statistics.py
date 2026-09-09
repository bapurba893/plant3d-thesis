"""
01_dataset_statistics.py

Produces the dataset statistics table to show your professor:
counts per species, point-cloud counts, avg points/cloud, per dataset.

Usage:
    python 01_dataset_statistics.py --crops3d_root data/crops3d --pheno4d_root data/pheno4d
"""

import argparse
import pandas as pd
from data_io import scan_crops3d_directory, scan_pheno4d_directory


def summarize(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{
            "Dataset": dataset_name, "Species": "-", "N_PointClouds": 0,
            "Avg_Points": 0, "Min_Points": 0, "Max_Points": 0,
        }])
    g = df.groupby("species")["n_points"].agg(["count", "mean", "min", "max"]).reset_index()
    g.columns = ["Species", "N_PointClouds", "Avg_Points", "Min_Points", "Max_Points"]
    g.insert(0, "Dataset", dataset_name)
    g["Avg_Points"] = g["Avg_Points"].round(0).astype(int)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops3d_root", required=True, help="Path to Crops3D/ (containing Tomato/, Maize/)")
    ap.add_argument("--pheno4d_root", required=True, help="Path to Pheno4D/ (containing plant folders)")
    ap.add_argument("--out_csv", default="dataset_statistics.csv")
    ap.add_argument("--out_manifest_csv", default="dataset_manifest.csv")
    args = ap.parse_args()

    print("Scanning Crops3D ...")
    crops3d_df = scan_crops3d_directory(args.crops3d_root)
    print(f"  found {len(crops3d_df)} point clouds")

    print("Scanning Pheno4D ...")
    pheno4d_df = scan_pheno4d_directory(args.pheno4d_root)
    print(f"  found {len(pheno4d_df)} point clouds "
          f"({pheno4d_df['is_annotated'].sum() if not pheno4d_df.empty else 0} annotated)")

    # Full manifest (every file, useful for later split scripts)
    full_manifest = pd.concat([crops3d_df, pheno4d_df], ignore_index=True, sort=False)
    full_manifest.to_csv(args.out_manifest_csv, index=False)
    print(f"Full file manifest written to {args.out_manifest_csv}")

    # Summary table (the one to show your professor)
    summary = pd.concat([
        summarize(crops3d_df, "Crops3D (Source)"),
        summarize(pheno4d_df, "Pheno4D (Target)"),
    ], ignore_index=True)

    print("\n=== Dataset Statistics ===")
    print(summary.to_string(index=False))
    summary.to_csv(args.out_csv, index=False)
    print(f"\nSummary table written to {args.out_csv}")

    # Extra: Pheno4D plant-level breakdown (needed later for leakage-safe splits)
    if not pheno4d_df.empty:
        plant_level = pheno4d_df.groupby(["species", "plant_id"]).agg(
            n_scans=("filepath", "count"),
            n_annotated=("is_annotated", "sum"),
        ).reset_index()
        print("\n=== Pheno4D per-plant scan counts ===")
        print(plant_level.to_string(index=False))
        plant_level.to_csv("pheno4d_plant_level.csv", index=False)


if __name__ == "__main__":
    main()
