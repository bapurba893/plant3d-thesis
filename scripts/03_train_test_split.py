"""
03_train_test_split.py

Builds the split manifests described in the Phase 0 plan:

  Crops3D (source, labeled)      -> stratified train / val split by species
  Pheno4D (target)                -> split BY PLANT, not by individual scan,
                                      so no plant's time-series is split
                                      across train and held-out sets
                                      (that would leak temporal information).

Produces 4 CSVs:
  crops3d_train.csv, crops3d_val.csv
  pheno4d_adaptation_pool.csv   <- used unlabeled during DA training
  pheno4d_heldout_eval.csv      <- labels only ever touched at final evaluation

Usage:
    python 03_train_test_split.py --manifest dataset_manifest.csv --seed 42
"""

import argparse
import numpy as np
import pandas as pd


def split_crops3d(df: pd.DataFrame, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_rows, val_rows = [], []
    for species, group in df.groupby("species"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_rows.append(df.loc[val_idx])
        train_rows.append(df.loc[train_idx])
    return pd.concat(train_rows).reset_index(drop=True), pd.concat(val_rows).reset_index(drop=True)


def split_pheno4d_by_plant(df: pd.DataFrame, n_heldout_plants_per_species: int, seed: int):
    """
    Reserve N whole plants per species entirely for held-out evaluation.
    All scans (all days) of a reserved plant go to heldout_eval.
    Every other plant's scans go to the adaptation pool.
    """
    rng = np.random.default_rng(seed)
    heldout_rows, pool_rows = [], []
    for species, group in df.groupby("species"):
        plant_ids = sorted(group["plant_id"].unique())
        rng.shuffle(plant_ids)
        heldout_plants = set(plant_ids[:n_heldout_plants_per_species])
        heldout_rows.append(group[group["plant_id"].isin(heldout_plants)])
        pool_rows.append(group[~group["plant_id"].isin(heldout_plants)])
        print(f"  {species}: held out plants {sorted(heldout_plants)}, "
              f"adaptation-pool plants {sorted(set(plant_ids) - heldout_plants)}")
    return (pd.concat(pool_rows).reset_index(drop=True),
            pd.concat(heldout_rows).reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_manifest.csv",
                     help="Output of 01_dataset_statistics.py")
    ap.add_argument("--val_frac", type=float, default=0.15,
                     help="Fraction of Crops3D held out for source-domain validation")
    ap.add_argument("--n_heldout_plants", type=int, default=2,
                     help="Whole Pheno4D plants per species reserved only for final eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)

    crops3d = manifest[manifest["dataset"] == "Crops3D"].reset_index(drop=True)
    pheno4d = manifest[manifest["dataset"] == "Pheno4D"].reset_index(drop=True)

    print("Splitting Crops3D (stratified by species) ...")
    c3d_train, c3d_val = split_crops3d(crops3d, args.val_frac, args.seed)
    c3d_train.to_csv("crops3d_train.csv", index=False)
    c3d_val.to_csv("crops3d_val.csv", index=False)
    print(f"  train={len(c3d_train)}  val={len(c3d_val)}")

    print("Splitting Pheno4D (by whole plant, no temporal leakage) ...")
    p4d_pool, p4d_heldout = split_pheno4d_by_plant(pheno4d, args.n_heldout_plants, args.seed)
    p4d_pool.to_csv("pheno4d_adaptation_pool.csv", index=False)
    p4d_heldout.to_csv("pheno4d_heldout_eval.csv", index=False)
    print(f"  adaptation_pool={len(p4d_pool)} scans   heldout_eval={len(p4d_heldout)} scans")

    print("\nDone. Rule for training code:")
    print("  - Crops3D labels are usable throughout training (source, supervised).")
    print("  - Pheno4D labels in pheno4d_adaptation_pool.csv must NOT be used during")
    print("    DA training (treat as unlabeled target) — only for optional diagnostics.")
    print("  - Pheno4D labels in pheno4d_heldout_eval.csv are touched ONLY when")
    print("    reporting final results.")


if __name__ == "__main__":
    main()
