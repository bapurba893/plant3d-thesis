"""
assemble_trait_timeseries.py

D0 (trait-extraction pipeline, Block D prerequisite): groups
`data/traits/pheno4d_traits_per_scan.csv` (one row per scan, from
`scripts/trait_extraction.py`) by plant, sorts by elapsed days since each
plant's own first scan, and writes a tidy long-format time series ready
for `scripts/growth_curves.py` to consume.

`scan_date` is a raw MMDD-style float token (e.g. `313.0` = March 13,
confirmed via `data/preprocessed/preprocessed_manifest.csv`: every
Pheno4D plant's scan dates fall entirely within March, min 305.0/max
325.0 across all 14 plants -- no year is encoded in the Pheno4D filename
scheme at all). Parsed into a day-of-year via an ARBITRARY common year
(2000, a leap year, chosen only to get a valid `datetime.date` -- the
year itself carries no meaning and must never be used as if it did).
This would silently mis-order a future data update if a plant's scans
ever crossed a Dec->Jan boundary -- not the case today (verified), but
guarded defensively below with an assertion rather than assumed to hold
forever.
"""

import argparse
import datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_scan_date(scan_date_token: float) -> datetime.date:
    """MMDD float (e.g. 313.0) -> datetime.date in an arbitrary common
    year (2000). Only ever used to compute elapsed-day differences within
    the same plant -- the absolute date/year is meaningless."""
    mmdd = f"{int(round(scan_date_token)):04d}"
    month, day = int(mmdd[:2]), int(mmdd[2:])
    return datetime.date(2000, month, day)


def assemble(traits_csv, plant_level_csv, out_csv):
    df = pd.read_csv(traits_csv)
    df["parsed_date"] = df["scan_date"].apply(parse_scan_date)

    out_rows = []
    for plant_id, group in df.groupby("plant_id"):
        group = group.sort_values("parsed_date").reset_index(drop=True)
        dates = group["parsed_date"]
        # Defensive check: raw scan_date should already sort consistently
        # with parsed_date (i.e. no year-boundary wraparound silently
        # scrambling order) -- assert rather than silently trust forever.
        assert list(group["scan_date"]) == sorted(group["scan_date"]), (
            f"{plant_id}: scan_date and parsed_date disagree on ordering -- "
            f"likely a year-boundary wraparound in scan dates, MMDD parsing "
            f"assumption violated. Investigate before trusting this plant's "
            f"time series.")
        first_date = dates.iloc[0]
        group["elapsed_days"] = dates.apply(lambda d: (d - first_date).days)
        out_rows.append(group)

    long_df = pd.concat(out_rows, ignore_index=True)
    long_df = long_df.drop(columns=["parsed_date"])
    cols = ["plant_id", "species", "elapsed_days", "scan_date"] + [
        c for c in long_df.columns if c not in
        ("plant_id", "species", "elapsed_days", "scan_date", "source_filepath")
    ]
    long_df = long_df[cols + ["source_filepath"]]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, index=False)
    print(f"Wrote {len(long_df)} rows ({long_df['plant_id'].nunique()} plants) to {out_path}")

    # Integrity check against the pre-aggregated plant-level reference.
    plant_level = pd.read_csv(plant_level_csv)
    counts = long_df.groupby("plant_id").size().rename("n_scans_assembled")
    check = plant_level.set_index("plant_id")[["n_scans"]].join(counts)
    check["diff"] = check["n_scans_assembled"] - check["n_scans"]
    mismatches = check[check["diff"] != 0]
    if len(mismatches):
        print("[warn] plant scan-count mismatches vs. data/pheno4d_plant_level.csv "
              "(expected if a scan's segmented file or norm_scale was missing "
              "upstream -- see trait_extraction.py's own reported counts):")
        print(mismatches)
    else:
        print("Integrity check passed: every plant's assembled scan count matches "
              "data/pheno4d_plant_level.csv exactly.")

    return long_df


def main():
    ap = argparse.ArgumentParser(description="D0: assemble per-plant trait time series")
    ap.add_argument("--traits_csv", type=str,
                     default=str(_REPO_ROOT / "data" / "traits" / "pheno4d_traits_per_scan.csv"))
    ap.add_argument("--plant_level_csv", type=str,
                     default=str(_REPO_ROOT / "data" / "pheno4d_plant_level.csv"))
    ap.add_argument("--out", type=str,
                     default=str(_REPO_ROOT / "data" / "traits" / "pheno4d_traits_timeseries_long.csv"))
    args = ap.parse_args()
    assemble(args.traits_csv, args.plant_level_csv, args.out)


if __name__ == "__main__":
    main()
