"""
13_sw_summarize.py

Block PP, learned soil-weight experiment (jobs/pp_softweight.sbatch,
adapters/train_c1_dgcnn_da0_softw.py): compares the two learned-weight versions against
the soil-kept (P4a) and soil-removed (P4) baselines. Run logs are parsed with the helpers
in scripts/10_pp_summarize.py (imported, not modified), so every metric is defined exactly
as in pp_summary.md:

  full_run_mean_acc      mean over all epochs of the per-epoch target (Pheno4D held-out,
                         63 clouds) accuracy -- PRIMARY metric
  full_run_mean_bal_acc  same, for the balanced ("avg acc") accuracy
  selected_acc           target accuracy of the checkpoint with the lowest source val loss

Versions and run folders (3 seeds each):
  P4a      soil kept             results/PP_P4a_s{1,2,3}
  P4       soil removed          results/PP_P4_s{1,2,3}
  SW_scalar    learned scalar    results/PP_SW_learned_scalar_s{1,2,3}
  SW_features  learned features  results/PP_SW_learned_features_s{1,2,3}

Decision rule: a learned version beats P4a only if ALL 3 of its seeds have a higher
full-run mean target accuracy than the P4a 3-seed mean.

Soil-weight statistics come from the per-epoch "SoilW - Source <e>, ..." lines
(adapters/soil_weight.py::SoilWeightStats). Each line holds statistics over all Crops3D
maize soil/pot points seen in that epoch: raw weight (bounded to [0.2, 5.0]), normalised
weight (batch-normalised to mean 1 over all maize points) and the plant points' normalised
weight, plus the fraction of soil points whose raw weight is within 1% of the bound range
(0.048) of 0.2 ("lo") or 5.0 ("hi"). Over the last 10 epochs this script reports the
average of the per-epoch means and bound fractions, and the lowest min / highest max.

Maize mIoU is not reported: it is not comparable between soil-kept and soil-removed
versions (absent classes scored 0, see 10_pp_summarize.py).

Writes results/pp_summary/sw_summary.md and results/pp_summary/sw_by_version.csv.

Usage:
  python scripts/13_sw_summarize.py
"""

import argparse
import csv
import importlib.util
import re
import statistics
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "pp_summarize", Path(__file__).resolve().parent / "10_pp_summarize.py")
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

SEEDS = (1, 2, 3)
VERSIONS = [  # (name, description, run-folder prefix, is learned)
    ("P4a", "soil kept", "PP_P4a", False),
    ("P4", "soil removed", "PP_P4", False),
    ("SW_scalar", "learned soil weight, scalar", "PP_SW_learned_scalar", True),
    ("SW_features", "learned soil weight, features", "PP_SW_learned_features", True),
]
REFERENCE = "P4a"
METRICS = [("full_run_mean_acc", "Full-run mean target acc (primary)"),
           ("full_run_mean_bal_acc", "Full-run mean target bal. acc"),
           ("selected_acc", "Selected-checkpoint target acc")]
LAST_N_EPOCHS = 10

_NUM = r"([\d.]+)"
SOILW_RE = re.compile(
    pp.TS + r"SoilW - Source (?P<e>\d+), "
    rf"soil_raw: mean={_NUM} std={_NUM} min={_NUM} max={_NUM} \| "
    rf"soil: mean={_NUM} std={_NUM} min={_NUM} max={_NUM} \| "
    rf"plant: mean={_NUM} std={_NUM} min={_NUM} max={_NUM} \| "
    rf"soil_raw at bounds: lo={_NUM} hi={_NUM}")
SOILW_KEYS = ["raw_mean", "raw_std", "raw_min", "raw_max", "soil_mean", "soil_std", "soil_min",
              "soil_max", "plant_mean", "plant_std", "plant_min", "plant_max", "at_lo", "at_hi"]
WEIGHT_FIELDS = ["version", "seed", "epochs_used", "raw_mean", "raw_min", "raw_max", "soil_mean",
                 "soil_min", "soil_max", "plant_mean", "at_lo", "at_hi"]


def parse_soil_weights(log_path):
    """Last-N-epoch soil-weight statistics from one run.log, or (None, reason)."""
    lines = log_path.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if pp.NAMESPACE_RE.match(ln)]
    lines = lines[starts[-1]:] if starts else lines  # same "last run wins" rule as 10_
    per_epoch = {}
    for ln in lines:
        if m := SOILW_RE.match(ln):
            per_epoch[int(m.group("e"))] = dict(zip(SOILW_KEYS, map(float, m.groups()[1:])))
    if not per_epoch:
        return None, "no SoilW lines in run.log"
    last = [per_epoch[e] for e in sorted(per_epoch)[-LAST_N_EPOCHS:]]
    out = {"epochs_used": f"{min(sorted(per_epoch)[-LAST_N_EPOCHS:])}-{max(per_epoch)}"}
    for k in ("raw_mean", "soil_mean", "plant_mean", "at_lo", "at_hi"):
        out[k] = statistics.fmean(d[k] for d in last)
    for k in ("raw_min", "soil_min"):
        out[k] = min(d[k] for d in last)
    for k in ("raw_max", "soil_max"):
        out[k] = max(d[k] for d in last)
    return out, None


def mean_std(vals):
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def main():
    ap = argparse.ArgumentParser(description="Summarise the learned soil-weight runs")
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "results" / "pp_summary"))
    args = ap.parse_args()

    specs = [f"results/{prefix}_s{s}:{name}:{s}" for name, _, prefix, _ in VERSIONS for s in SEEDS]
    runs = pp.collect_runs(_REPO_ROOT / "results", use_glob=False, explicit=specs)
    by_version = {name: {r["seed"]: r for r in runs if r["pipeline"] == name}
                  for name, *_ in VERSIONS}

    missing = []
    for r in runs:
        for w in r["_warnings"]:
            missing.append(f"{r['pipeline']} s{r['seed']}: {w}")

    # Per-version table: per-seed values, mean, std (complete runs only, as in 10_)
    rows = []
    for name, desc, prefix, learned in VERSIONS:
        row = {"version": name, "description": desc}
        for k, _ in METRICS:
            vals = []
            for s in SEEDS:
                r = by_version[name].get(s)
                v = r[k] if r is not None and r["complete"] else None
                row[f"{k}_s{s}"] = v
                if v is not None:
                    vals.append(v)
            row[f"{k}_mean"], row[f"{k}_std"] = mean_std(vals)
            row[f"{k}_n"] = len(vals)
        rows.append(row)
    ref = next(r for r in rows if r["version"] == REFERENCE)
    ref_mean, ref_n = ref["full_run_mean_acc_mean"], ref["full_run_mean_acc_n"]

    # Decision rule
    decisions = []
    for row, (name, desc, _, learned) in zip(rows, VERSIONS):
        if not learned:
            row["decision"] = "reference" if name == REFERENCE else "baseline"
            continue
        seed_vals = [row[f"full_run_mean_acc_s{s}"] for s in SEEDS]
        vals_txt = ", ".join(f"s{s}={pp.fmt(v)}" for s, v in zip(SEEDS, seed_vals))
        if ref_n < len(SEEDS) or any(v is None for v in seed_vals):
            row["decision"] = "pending (need 3 complete seeds for both versions)"
            line = f"{name}: PENDING -- {vals_txt}; P4a mean {pp.fmt(ref_mean)} over {ref_n} seed(s)"
        else:
            n_above = sum(v > ref_mean for v in seed_vals)
            yes = n_above == len(SEEDS)
            row["decision"] = "beats P4a" if yes else "does not beat P4a"
            line = (f"{name} beats P4a: {'YES' if yes else 'NO'} -- {n_above}/{len(SEEDS)} seeds "
                    f"above the P4a 3-seed mean {ref_mean:.4f} ({vals_txt})")
        decisions.append(line)

    # Soil-weight statistics
    weights = []
    for name, _, prefix, learned in VERSIONS:
        if not learned:
            continue
        for s in SEEDS:
            log = _REPO_ROOT / "results" / f"{prefix}_s{s}" / "run.log"
            if not log.exists():
                missing.append(f"{name} s{s}: run.log missing, no soil-weight statistics")
                continue
            w, err = parse_soil_weights(log)
            if w is None:
                missing.append(f"{name} s{s}: {err}")
                continue
            weights.append({"version": name, "seed": s, **w})
    any_lo = any(w["at_lo"] > 0 for w in weights)
    any_hi = any(w["at_hi"] > 0 for w in weights)
    if not weights:
        bounds_line = "Bounds: no soil-weight statistics available."
    elif not (any_lo or any_hi):
        bounds_line = (f"Bounds: NO -- no seed of either learned version had any soil point within "
                       f"0.048 of the 0.2 or 5.0 bound in the last {LAST_N_EPOCHS} epochs "
                       f"(raw weight range {min(w['raw_min'] for w in weights):.4f}-"
                       f"{max(w['raw_max'] for w in weights):.4f}).")
    else:
        hit = [f"{w['version']} s{w['seed']} (lo={w['at_lo']:.4f}, hi={w['at_hi']:.4f})"
               for w in weights if w["at_lo"] > 0 or w["at_hi"] > 0]
        bounds_line = "Bounds: YES -- soil points at a bound in: " + "; ".join(hit) + "."

    # ---- outputs ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["version", "description"] + [f"{k}_{x}" for k, _ in METRICS
                                            for x in [f"s{s}" for s in SEEDS] + ["mean", "std", "n"]] \
        + ["decision"]
    with open(out_dir / "sw_by_version.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}
                    for r in rows)

    def ms(row, k, nd=4):
        m, sd = row[f"{k}_mean"], row[f"{k}_std"]
        return "-" if m is None else (f"{m:.{nd}f} ± {sd:.{nd}f}" if sd is not None else f"{m:.{nd}f}")

    term_rows = []
    for row in rows:
        for k, label in METRICS:
            term_rows.append([row["version"] if k == METRICS[0][0] else "", label.split(" (")[0]]
                             + [pp.fmt(row[f"{k}_s{s}"]) for s in SEEDS] + [ms(row, k)])
    main_tbl = pp.text_table(["version", "metric", "s1", "s2", "s3", "mean ± std"], term_rows)
    w_tbl = pp.text_table(
        ["version", "seed", "epochs", "raw mean", "raw min", "raw max", "norm mean", "norm min",
         "norm max", "plant mean", "frac@0.2", "frac@5.0"],
        [[w["version"], str(w["seed"]), w["epochs_used"]]
         + [f"{w[k]:.4f}" for k in WEIGHT_FIELDS[3:]] for w in weights])
    missing_txt = missing or ["none -- all 12 runs complete, all metrics computed"]

    report = "\n".join(
        ["TARGET = Pheno4D held-out, 63 clouds. Mean ± sample std over 3 seeds.",
         "Selected checkpoint = lowest source val loss (target labels never used).", "",
         main_tbl, "", "DECISION (vs P4a, primary metric):"] + ["  " + d for d in decisions]
        + ["", f"SOIL WEIGHT, last {LAST_N_EPOCHS} epochs (means averaged per epoch; min/max over "
               f"those epochs; frac = fraction of soil points within 0.048 of a bound)",
           w_tbl, bounds_line, "", "MISSING / WARNINGS:"] + ["  " + m for m in missing_txt])
    print(report)

    md = ["**Block PP: learned soil weight vs. soil kept (P4a) and soil removed (P4)** "
          "(DGCNN, DA-0; target = Pheno4D held-out, 63 clouds)", "",
          "| Version | Metric | Seed 1 | Seed 2 | Seed 3 | Mean ± std |", "|---|---|---|---|---|---|"]
    for row in rows:
        for k, label in METRICS:
            md.append(f"| {row['version']} ({row['description']}) | {label} | "
                      + " | ".join(pp.fmt(row[f'{k}_s{s}'], 3) for s in SEEDS)
                      + f" | {ms(row, k, 3)} |")
    md += ["", "Full-run mean = mean over the 100 epochs of the per-epoch target accuracy. "
               "Selected checkpoint = lowest source validation loss; target labels are never used "
               "for selection. Std = sample std over the 3 seeds. Maize mIoU is not reported: it "
               "is not comparable between soil-kept and soil-removed versions.", "",
           f"**Decision rule:** a learned version beats P4a only if all 3 of its seeds exceed the "
           f"P4a 3-seed mean full-run accuracy ({pp.fmt(ref_mean)}).", ""]
    md += [f"- {d}" for d in decisions]
    md += ["", f"**Soil weight, last {LAST_N_EPOCHS} epochs** (raw = bounded weight in [0.2, 5.0]; "
               "normalised = after scaling to mean 1 over all maize points per batch; means "
               "averaged over the epochs, min/max over the epochs; at-bound fractions = share of "
               "soil points within 0.048 of 0.2 or 5.0)", "",
           "| Version | Seed | Raw mean | Raw min | Raw max | Norm. mean | Norm. min | Norm. max "
           "| Plant mean | Frac. at 0.2 | Frac. at 5.0 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {w['version']} | {w['seed']} | " + " | ".join(f"{w[k]:.4f}" for k in WEIGHT_FIELDS[3:])
           + " |" for w in weights]
    md += ["", bounds_line, "", "**Missing / warnings:**", ""] + [f"- {m}" for m in missing_txt]
    (out_dir / "sw_summary.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_dir}/sw_summary.md, sw_by_version.csv")


if __name__ == "__main__":
    main()
