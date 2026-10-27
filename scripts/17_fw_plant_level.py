"""
17_fw_plant_level.py

Plant-level view of the FW cross-evaluation: reads the per-cloud predictions written by
scripts/16_fw_cross_eval.py (fw_cross_eval_clouds.csv, one line per run x cache x repeat x cloud)
and reports, per cache:

  1. per-plant accuracy for every row: for each seed, the fraction of that plant's held-out clouds
     predicted correctly (averaged over the evaluation repeats for PointNet++/KPConv), then mean and
     sample SD over the 3 seeds;
  2. each row's mean accuracy (all 63 clouds pooled, mean over seeds -- the same number as
     fw_cross_eval.md's "Acc" column) with a PLANT-LEVEL bootstrap 95% interval: the 4 held-out
     plants are resampled with replacement (scans are never resampled on their own; a plant is
     drawn with all of its clouds), the pooled accuracy is recomputed per seed on the drawn clouds,
     and the seed mean is taken. Percentile interval over --n_boot resamples.

**4 plants are too few for these intervals to be reliable.** Resampling 4 plants with replacement
gives only 35 distinct multisets, some of which contain a single species (always-correct or
always-wrong clouds of one class), so the interval is coarse, can be very wide, and its nominal 95%
coverage is not achieved. The intervals show how much the result depends on which plants happen to
be in the test set; they are not a confidence statement about new plants. This warning is written
into every output.

Usage:
  python scripts/17_fw_plant_level.py --clouds results/fw_plant_level/*/fw_cross_eval_clouds.csv
  python scripts/17_fw_plant_level.py --clouds <csv> [<csv> ...] --cache P4a --out_dir <dir>
"""

import argparse
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

WARNING = ("4 held-out plants are too few for these bootstrap intervals to be reliable: resampling "
           "4 plants with replacement gives only {n_multisets} distinct resamples ({n_one_species} of "
           "them contain a single species), so the intervals are coarse and do not reach their nominal "
           "95% coverage. They show how much each result depends on which plants are in the test set, "
           "not how a new set of plants would score.")
ROW_ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B3b", "B4", "B5",
             "C1", "C2", "C3", "C4", "C5", "C6"]


def row_sort_key(row):
    return ROW_ORDER.index(row) if row in ROW_ORDER else len(ROW_ORDER)


def load(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(["run", "cache", "repeat", "cloud_id"])
    if dup.any():
        sys.exit(f"{int(dup.sum())} duplicate (run, cache, repeat, cloud) lines across the input files")
    return df


def per_seed_plant(df: pd.DataFrame) -> pd.DataFrame:
    """Correct count and cloud count per row x seed x plant (repeat-averaged correct count)."""
    per_rep = (df.groupby(["row", "oracle", "seed", "plant_id", "species", "repeat"])
               .agg(correct=("correct", "sum"), n=("correct", "size")).reset_index())
    out = (per_rep.groupby(["row", "oracle", "seed", "plant_id", "species"])
           .agg(correct=("correct", "mean"), n=("n", "first"), repeats=("repeat", "nunique"))
           .reset_index())
    out["acc"] = out["correct"] / out["n"]
    return out


def check_complete(sp: pd.DataFrame, plants):
    """Every row must have every seed x plant; report what is missing instead of guessing."""
    problems = []
    for row, g in sp.groupby("row"):
        for seed, gs in g.groupby("seed"):
            missing = sorted(set(plants) - set(gs.plant_id))
            if missing:
                problems.append(f"{row} seed {seed}: no clouds for plants {missing}")
    return problems


def bootstrap(sp_row: pd.DataFrame, plants, n_boot: int, rng) -> tuple:
    """Plant-level bootstrap of the seed-mean pooled accuracy. Returns (point, lo, hi, draws)."""
    seeds = sorted(sp_row.seed.unique())
    c = sp_row.pivot(index="seed", columns="plant_id", values="correct").loc[seeds, plants].to_numpy()
    n = sp_row.pivot(index="seed", columns="plant_id", values="n").loc[seeds, plants].to_numpy()
    point = float(np.mean(c.sum(1) / n.sum(1)))
    idx = rng.integers(0, len(plants), size=(n_boot, len(plants)))
    # (n_boot, seeds): pooled accuracy of the drawn plants' clouds, per seed
    accs = c[:, idx].sum(-1).T / n[:, idx].sum(-1).T
    draws = accs.mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), draws


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clouds", nargs="+", required=True, help="fw_cross_eval_clouds.csv file(s)")
    ap.add_argument("--cache", nargs="+", default=None, help="caches to report (default: all present)")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "results" / "fw_plant_level"))
    args = ap.parse_args()

    df = load(args.clouds)
    caches = args.cache or list(dict.fromkeys(df.cache))  # evaluation order (reference first)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plants = sorted(df.plant_id.unique())
    species_of = df.drop_duplicates("plant_id").set_index("plant_id").species.to_dict()
    multisets = list(combinations_with_replacement(plants, len(plants)))
    n_one_species = sum(len({species_of[p] for p in m}) == 1 for m in multisets)
    warning = WARNING.format(n_multisets=len(multisets), n_one_species=n_one_species)
    print(f"WARNING: {warning}\n")

    plant_rows, boot_rows, problems = [], [], []
    L = ["# FW plant-level accuracy and plant-level bootstrap", "",
         f"Written by `scripts/17_fw_plant_level.py` from: "
         + ", ".join(f"`{Path(p).as_posix()}`" for p in args.clouds), "",
         f"**Warning: {warning}**", "",
         f"Held-out plants: " + ", ".join(f"{p} ({species_of[p]}, "
                                          f"{int(df[df.plant_id == p].cloud_id.nunique())} clouds)"
                                          for p in plants) + ".", "",
         "Per-plant accuracy = fraction of that plant's clouds predicted correctly, per seed (mean of "
         "the evaluation repeats for PointNet++/KPConv), then mean ± sample SD over seeds. Row mean = "
         "all clouds pooled, mean over seeds. Bootstrap: resample the plants with replacement (each "
         f"drawn plant brings all of its clouds), {args.n_boot} resamples, percentile 95% interval, "
         f"numpy seed {args.boot_seed}.", ""]

    for cache in caches:
        sub = df[df.cache == cache]
        if sub.empty:
            problems.append(f"cache {cache}: no lines")
            continue
        sp = per_seed_plant(sub)
        problems += [f"cache {cache}: {p}" for p in check_complete(sp, plants)]
        rng = np.random.default_rng(args.boot_seed)
        L += [f"## Cache {cache}", "",
              "| Row | " + " | ".join(f"{p} ({species_of[p]})" for p in plants)
              + " | Row mean (pooled) | Plant-bootstrap 95% interval | Seeds |",
              "|---|" + "---|" * (len(plants) + 3)]
        for row in sorted(sp.row.unique(), key=row_sort_key):
            g = sp[sp.row == row]
            cells = []
            for p in plants:
                v = g[g.plant_id == p].acc
                sd = v.std(ddof=1) if len(v) > 1 else np.nan
                plant_rows.append({"cache": cache, "row": row, "oracle": bool(g.oracle.iloc[0]),
                                   "plant_id": p, "species": species_of[p],
                                   "n_clouds": int(g[g.plant_id == p].n.iloc[0]),
                                   "acc_mean": v.mean(), "acc_sd": sd, "n_seeds": len(v),
                                   "acc_per_seed": " / ".join(f"{a:.4f}" for a in
                                                               g[g.plant_id == p].sort_values("seed").acc)})
                cells.append(f"{v.mean():.3f} ± {sd:.3f}" if not np.isnan(sd) else f"{v.mean():.3f}")
            point, lo, hi, _ = bootstrap(g, plants, args.n_boot, rng)
            boot_rows.append({"cache": cache, "row": row, "oracle": bool(g.oracle.iloc[0]),
                              "n_seeds": g.seed.nunique(), "mean_acc": point, "boot_lo": lo,
                              "boot_hi": hi, "n_boot": args.n_boot, "n_plants": len(plants)})
            label = f"{row} (Oracle)" if g.oracle.iloc[0] else row
            L.append(f"| {label} | " + " | ".join(cells)
                     + f" | {point:.3f} | [{lo:.3f}, {hi:.3f}] | {g.seed.nunique()} |")
        L.append("")

    if problems:
        L += ["## Problems", ""] + [f"- {p}" for p in problems] + [""]
    pd.DataFrame(plant_rows).to_csv(out_dir / "plant_level_per_plant.csv", index=False)
    pd.DataFrame(boot_rows).assign(warning=warning).to_csv(out_dir / "plant_level_bootstrap.csv",
                                                           index=False)
    (out_dir / "plant_level.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if problems:
        print("PROBLEMS:\n" + "\n".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
