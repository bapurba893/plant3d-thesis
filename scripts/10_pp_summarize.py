"""
10_pp_summarize.py

Block PP, Stage 4 (docs/PREPROCESSING_COMPARISON_PLAN.md section 7): summarises
the pp_compare job array (jobs/pp_compare.sbatch). Reads results/PP_<ID>_s<seed>/run.log
for every run, as written by adapters/train_c1_dgcnn_da0.py, and writes to
results/pp_summary/:

  pp_runs.csv          one row per run (pipeline, seed, metrics, cloud counts, warnings)
  pp_by_pipeline.csv   3-seed mean/std per pipeline + the plan's two reference lines
                       + the plan's decision rule
  pp_summary.txt       the same two tables as plain text (also printed to screen)
  pp_summary.md        per-pipeline table + data-check warnings as Markdown, for pasting

Metrics, all parsed from run.log (nothing recomputed from checkpoints):
  full_run_mean_acc      mean over all epochs of the per-epoch target (Pheno4D held-out)
                         accuracy -- the "Eval(no-train) - Target <e>, acc:" lines
  full_run_mean_bal_acc  same, for the "avg acc" field (sklearn balanced_accuracy_score)
  selected_acc           target accuracy of the protocol-selected checkpoint (lowest source
                         val total loss, strict '<', first minimum wins), from the
                         "FINAL target (Pheno4D held-out) test accuracy" line
  tomato/maize_miou      source-val seg mIoU of that same selected checkpoint, from the
                         "FINAL best-model source val, <species> seg" lines
  n_train/n_val/n_heldout  clouds actually loaded (the dataset silently skips split rows
                         with no cache entry, so these are the real counts)

Decision rule (plan section 7): a pipeline "beats P0" only if ALL 3 of its seeds have a
higher full-run mean target accuracy than P0's 3-seed mean; otherwise "no clear
difference". Anything with fewer than 3 seeds (P0 or the pipeline) is reported as pending.

Not covered here: the shortcut score (plan section 7, item 5) -- it is not a training
metric and is not in run.log; it comes from scripts/09_shortcut_probe.py.

Usage:
  python scripts/10_pp_summarize.py
  # test / stand-in: treat any run folder as <pipeline> seed <seed>
  python scripts/10_pp_summarize.py --run results/C1_dgcnn_da0_clsseg:P0:1 --no_glob
"""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

PIPELINE_ORDER = ["P0", "P1", "P2", "P3", "P4", "P4a", "P4b", "P4c", "P4d", "P5"]
N_SEEDS_REQUIRED = 3
MAJORITY_BASELINE = 0.635  # 40/63, plan section 7
EXISTING_P0 = 0.791        # original single-seed C1 full-run mean, plan section 7

RUN_DIR_RE = re.compile(r"^PP_(?P<pipeline>[A-Za-z0-9]+)_s(?P<seed>\d+)$")
TS = r"^\d\d-\d\d-\d\d \d\d:\d\d:\d\d: "
NAMESPACE_RE = re.compile(TS + r"Namespace\(.*\bepochs=(?P<epochs>\d+)")
TRAIN_N_RE = re.compile(TS + r"Crops3D train: (?P<n>\d+) ")
VAL_N_RE = re.compile(TS + r"Crops3D val:\s+(?P<n>\d+) ")
HELDOUT_N_RE = re.compile(TS + r"Pheno4D held-out eval .*?: (?P<n>\d+) ")
VAL_EPOCH_RE = re.compile(TS + r"Val - Source (?P<e>\d+), cls acc: .*total loss: (?P<loss>[\d.]+)")
TRGT_EPOCH_RE = re.compile(
    TS + r"Eval\(no-train\) - Target (?P<e>\d+), acc: (?P<acc>[\d.]+), avg acc: (?P<bal>[\d.]+)")
BEST_RE = re.compile(TS + r"Best model at epoch (?P<e>\d+), source val total loss (?P<loss>[\d.]+)")
FINAL_SEG_RE = re.compile(
    TS + r"FINAL best-model source val, (?P<sp>Tomato|Maize) seg: .*mIoU=(?P<miou>[\d.]+)")
FINAL_TRGT_RE = re.compile(
    TS + r"FINAL target \(Pheno4D held-out\) test accuracy: (?P<acc>[\d.]+), avg acc: (?P<bal>[\d.]+)")

RUN_FIELDS = ["pipeline", "seed", "run_dir", "complete", "n_epochs", "full_run_mean_acc", "full_run_std_acc",
              "full_run_mean_bal_acc", "selected_epoch", "selected_acc", "selected_bal_acc",
              "tomato_miou", "maize_miou", "n_train", "n_val", "n_heldout", "warnings"]
METRICS = ["full_run_mean_acc", "full_run_mean_bal_acc", "selected_acc",
           "tomato_miou", "maize_miou", "n_train", "n_val", "n_heldout"]
COUNT_FIELDS = ["n_train", "n_val", "n_heldout"]
# Soil-removal pipelines (P2, P3, P4, P4b-d) drop Maize classes 0 and 5 from every cloud, and
# the training script's per-cloud mIoU averages over all 6 classes with absent classes scored 0,
# so Maize mIoU is capped at 4/6 there. Numbers are kept, but labelled.
MAIZE_MIOU_LABEL = "not comparable across pipelines (absent classes scored 0)"
MAIZE_MIOU_NOTE = (f"* Maize mIoU: {MAIZE_MIOU_LABEL}. Soil-removal pipelines (P2, P3, P4, "
                   f"P4b-P4d) have no Maize class 0 or 5 points, so their Maize mIoU is capped "
                   f"at 4/6; P0, P1 and P4a are not.")


def parse_run_log(log_path):
    """Parse one run.log. Returns (dict of metrics, list of warning strings)."""
    lines = log_path.read_text().splitlines()
    warnings = []

    # run.log is opened in append mode, so a re-run into the same folder stacks runs;
    # only the last one is used.
    starts = [i for i, ln in enumerate(lines) if NAMESPACE_RE.match(ln)]
    if not starts:
        return None, ["no Namespace(...) header line -- not a C1 run.log"]
    if len(starts) > 1:
        warnings.append(f"{len(starts)} runs appended in this run.log; using the last one")
    lines = lines[starts[-1]:]
    expected_epochs = int(NAMESPACE_RE.match(lines[0]).group("epochs"))

    r = {k: None for k in RUN_FIELDS}
    val_loss, trgt = {}, {}
    finals_seg = {}
    for ln in lines:
        if m := TRAIN_N_RE.match(ln):
            r["n_train"] = int(m.group("n"))
        elif m := VAL_N_RE.match(ln):
            r["n_val"] = int(m.group("n"))
        elif m := HELDOUT_N_RE.match(ln):
            r["n_heldout"] = int(m.group("n"))
        elif m := VAL_EPOCH_RE.match(ln):
            val_loss[int(m.group("e"))] = float(m.group("loss"))
        elif m := TRGT_EPOCH_RE.match(ln):
            trgt[int(m.group("e"))] = (float(m.group("acc")), float(m.group("bal")))
        elif m := BEST_RE.match(ln):
            r["selected_epoch"] = int(m.group("e"))
        elif m := FINAL_SEG_RE.match(ln):
            finals_seg[m.group("sp")] = float(m.group("miou"))
        elif m := FINAL_TRGT_RE.match(ln):
            r["selected_acc"] = float(m.group("acc"))
            r["selected_bal_acc"] = float(m.group("bal"))

    r["n_epochs"] = len(trgt)
    if len(trgt) != expected_epochs:
        warnings.append(f"only {len(trgt)}/{expected_epochs} epochs logged (run unfinished or crashed)")
    if trgt:
        accs = [trgt[e][0] for e in sorted(trgt)]
        bals = [trgt[e][1] for e in sorted(trgt)]
        r["full_run_mean_acc"] = statistics.fmean(accs)
        r["full_run_std_acc"] = statistics.pstdev(accs)
        r["full_run_mean_bal_acc"] = statistics.fmean(bals)
    r["tomato_miou"] = finals_seg.get("Tomato")
    r["maize_miou"] = finals_seg.get("Maize")
    if r["selected_acc"] is None:
        warnings.append("no FINAL target line -- selected-checkpoint metrics missing")
    r["complete"] = len(trgt) == expected_epochs and r["selected_acc"] is not None

    # Cross-check the script's own selection against the logged per-epoch val losses
    # (same strict '<' rule). Logged losses are rounded to 4 dp, so a mismatch can be a
    # rounding tie rather than a bug; it is flagged, not overridden.
    if val_loss and r["selected_epoch"] is not None:
        recomputed = min(sorted(val_loss), key=lambda e: val_loss[e])
        if recomputed != r["selected_epoch"]:
            warnings.append(f"logged best epoch {r['selected_epoch']} != argmin of logged "
                            f"val total loss {recomputed} (4-dp rounding tie?)")
        elif r["selected_epoch"] in trgt and r["selected_acc"] is not None \
                and abs(trgt[r["selected_epoch"]][0] - r["selected_acc"]) > 1e-4:
            warnings.append("FINAL target acc differs from that epoch's per-epoch target acc")
    for k in COUNT_FIELDS:
        if r[k] is None:
            warnings.append(f"{k} line not found")
    return r, warnings


def collect_runs(results_dir, use_glob, explicit):
    """explicit: list of 'path:PIPELINE:SEED' stand-in specs."""
    specs = []
    if use_glob:
        for d in sorted(results_dir.glob("PP_*")):
            m = RUN_DIR_RE.match(d.name)
            if d.is_dir() and m:
                specs.append((d, m.group("pipeline"), int(m.group("seed"))))
    for s in explicit:
        path, pipeline, seed = s.rsplit(":", 2)
        d = Path(path)
        specs.append((d if d.is_absolute() else _REPO_ROOT / d, pipeline, int(seed)))

    runs = []
    for d, pipeline, seed in specs:
        log = d / "run.log"
        if not log.exists():
            r = {k: None for k in RUN_FIELDS}
            r["complete"] = False
            warns = ["run.log missing (not started yet?)"]
        else:
            r, warns = parse_run_log(log)
            if r is None:
                r = {k: None for k in RUN_FIELDS}
                r["complete"] = False
        r.update(pipeline=pipeline, seed=seed, run_dir=str(d.relative_to(_REPO_ROOT)
                                                           if d.is_relative_to(_REPO_ROOT) else d))
        r["_warnings"] = warns
        runs.append(r)
    return runs


def check_counts(runs):
    """Flag any run whose cloud counts differ from P0's. Returns list of screen warnings."""
    out = []
    p0 = [r for r in runs if r["pipeline"] == "P0" and r["n_train"] is not None]
    if not p0:
        out.append("DATA CHECK SKIPPED: no P0 run with cloud counts to compare against")
        return out
    ref = tuple(p0[0][k] for k in COUNT_FIELDS)
    for r in p0[1:]:
        if tuple(r[k] for k in COUNT_FIELDS) != ref:
            out.append(f"P0 runs disagree on cloud counts ({r['run_dir']} vs {p0[0]['run_dir']})")
    for r in runs:
        diffs = []
        for k, ref_n in zip(COUNT_FIELDS, ref):
            n = r[k]
            if n is None or n == ref_n:
                continue
            if n < ref_n:
                diffs.append(f"{k}={n} < P0 {ref_n} (PARTLY MISSING CACHE?)")
            else:
                diffs.append(f"{k}={n} > P0 {ref_n}")
        if diffs:
            msg = "cloud counts differ from P0: " + "; ".join(diffs)
            r["_warnings"].append(msg)
            out.append(f"WARNING {r['pipeline']} s{r['seed']}: {msg}")
    return out


def pipeline_key(p):
    return (PIPELINE_ORDER.index(p) if p in PIPELINE_ORDER else len(PIPELINE_ORDER), p)


def summarise(runs):
    # Only complete runs (all epochs + FINAL lines) enter the aggregates and the decision;
    # an unfinished run's partial trajectory would bias the full-run mean.
    by, incomplete = {}, {}
    for r in runs:
        by.setdefault(r["pipeline"], [])
        (by if r["complete"] else incomplete).setdefault(r["pipeline"], []).append(r)
    p0_vals = [r["full_run_mean_acc"] for r in by.get("P0", []) if r["full_run_mean_acc"] is not None]
    p0_mean = statistics.fmean(p0_vals) if p0_vals else None

    rows = []
    for p in sorted(by, key=pipeline_key):
        rs = sorted(by[p], key=lambda r: r["seed"])
        row = {"pipeline": p, "seeds": ",".join(str(r["seed"]) for r in rs)}
        if p in incomplete:
            row["seeds"] += " (incomplete: " + ",".join(
                str(r["seed"]) for r in sorted(incomplete[p], key=lambda r: r["seed"])) + ")"
            row["seeds"] = row["seeds"].strip()
        for k in METRICS:
            vals = [r[k] for r in rs if r[k] is not None]
            row[f"{k}_n"] = len(vals)
            row[f"{k}_mean"] = statistics.fmean(vals) if vals else None
            row[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else None
        seed_vals = [r["full_run_mean_acc"] for r in rs if r["full_run_mean_acc"] is not None]
        if p == "P0":
            row["decision"] = "reference"
        elif len(p0_vals) < N_SEEDS_REQUIRED or len(seed_vals) < N_SEEDS_REQUIRED:
            row["decision"] = (f"pending (P0 seeds={len(p0_vals)}, "
                               f"{p} seeds={len(seed_vals)}; need {N_SEEDS_REQUIRED} each)")
        elif all(v > p0_mean for v in seed_vals):
            row["decision"] = "beats P0"
        else:
            row["decision"] = "no clear difference"
        rows.append(row)

    for name, val in [("ref: majority baseline (40/63)", MAJORITY_BASELINE),
                      ("ref: existing P0 single-seed C1", EXISTING_P0)]:
        row = {"pipeline": name, "seeds": "", "decision": ""}
        for k in METRICS:
            row[f"{k}_n"] = ""
            row[f"{k}_mean"] = val if k == "full_run_mean_acc" else None
            row[f"{k}_std"] = None
        rows.append(row)
    return rows, p0_mean


def fmt(v, nd=4):
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def text_table(header, rows):
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(r) for r in rows])


def markdown_report(rows, p0_mean, n_p0, count_warnings):
    """Per-pipeline table (mean ± std), decision, reference rows, data-check warnings."""
    def ms(row, k):
        m, sd = row[f"{k}_mean"], row[f"{k}_std"]
        if m is None:
            return "–"
        return f"{m:.3f} ± {sd:.3f}" if sd is not None else f"{m:.3f}"

    cols = [("full_run_mean_acc", "Full-run mean target acc"),
            ("full_run_mean_bal_acc", "Full-run mean target bal. acc"),
            ("selected_acc", "Selected-checkpoint target acc"),
            ("tomato_miou", "Source seg mIoU, tomato"),
            ("maize_miou", f"Source seg mIoU, maize* ({MAIZE_MIOU_LABEL})")]
    out = ["**Block PP: C1 baseline (DGCNN, DA-0) per preprocessing pipeline** "
           "(mean ± std over seeds; target = Pheno4D held-out, 63 clouds)", "",
           "| Pipeline | Seeds | " + " | ".join(c[1] for c in cols) + " | Decision vs P0 |",
           "|---|---|" + "---|" * len(cols) + "---|"]
    for row in rows:
        name = row["pipeline"]
        if name.startswith("ref: "):
            name = "*" + name[5:] + "*"
        out.append(f"| {name} | {row['seeds'] or '–'} | "
                   + " | ".join(ms(row, k) for k, _ in cols)
                   + f" | {row['decision'] or '–'} |")
    out += ["",
            f"Decision rule: a pipeline beats P0 only if all {N_SEEDS_REQUIRED} of its seeds have "
            f"a higher full-run mean target accuracy than P0's mean "
            f"({'n/a' if p0_mean is None else f'{p0_mean:.3f}'}, over {n_p0} complete seed(s)). "
            f"Selected checkpoint = lowest source validation loss; target labels never used "
            f"for selection. Std is shown only when there are 2 or more seeds.", "",
            "\\" + MAIZE_MIOU_NOTE]  # escaped so Markdown keeps the '*' instead of a bullet
    if count_warnings:
        out += ["", "**Data check warnings:**", ""] + [f"- {w}" for w in count_warnings]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Summarise Block PP runs (results/PP_*/run.log)")
    ap.add_argument("--results_dir", default=str(_REPO_ROOT / "results"))
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "results" / "pp_summary"))
    ap.add_argument("--run", action="append", default=[], metavar="PATH:PIPELINE:SEED",
                    help="add a run folder under an explicit pipeline/seed (e.g. a stand-in)")
    ap.add_argument("--no_glob", action="store_true", help="ignore results/PP_* folders")
    args = ap.parse_args()

    runs = collect_runs(Path(args.results_dir), not args.no_glob, args.run)
    if not runs:
        sys.exit("No runs found (results/PP_* is empty and no --run given)")
    count_warnings = check_counts(runs)
    for r in runs:
        r["warnings"] = " | ".join(r.pop("_warnings"))
    runs.sort(key=lambda r: (pipeline_key(r["pipeline"]), r["seed"]))
    rows, p0_mean = summarise(runs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pp_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        w.writerows({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}
                    for r in runs)
    pipe_fields = ["pipeline", "seeds"] + [f"{k}_{s}" for k in METRICS
                                           for s in ("mean", "std", "n")] + ["decision"]
    with open(out_dir / "pp_by_pipeline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pipe_fields)
        w.writeheader()
        w.writerows({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}
                    for r in rows)

    run_tbl = text_table(
        ["pipeline", "seed", "epochs", "fullrun_acc", "fullrun_bal", "sel_ep", "sel_acc",
         "tom_mIoU", "mai_mIoU*", "n_train", "n_val", "n_heldout"],
        [[r["pipeline"], str(r["seed"]), fmt(r["n_epochs"]), fmt(r["full_run_mean_acc"]),
          fmt(r["full_run_mean_bal_acc"]), fmt(r["selected_epoch"]), fmt(r["selected_acc"]),
          fmt(r["tomato_miou"]), fmt(r["maize_miou"]), fmt(r["n_train"]), fmt(r["n_val"]),
          fmt(r["n_heldout"])] for r in runs])
    ms = lambda row, k: (f"{fmt(row[f'{k}_mean'])} ± {fmt(row[f'{k}_std'])}"
                         if row[f"{k}_mean"] is not None else "-")
    pipe_tbl = text_table(
        ["pipeline", "seeds", "fullrun_acc", "fullrun_bal", "sel_acc", "tom_mIoU", "mai_mIoU*",
         "decision"],
        [[row["pipeline"], row["seeds"] or "-", ms(row, "full_run_mean_acc"),
          ms(row, "full_run_mean_bal_acc"), ms(row, "selected_acc"), ms(row, "tomato_miou"),
          ms(row, "maize_miou"), row["decision"] or "-"] for row in rows])

    run_warns = [f"  {r['pipeline']} s{r['seed']} ({r['run_dir']}): {r['warnings']}"
                 for r in runs if r["warnings"]]
    report = "\n".join(
        ["PER-RUN (target = Pheno4D held-out; sel = checkpoint chosen by lowest source val loss)",
         run_tbl, MAIZE_MIOU_NOTE, "",
         "PER-PIPELINE (mean ± sample std across COMPLETE seeds only; std '-' when n<2)",
         pipe_tbl, MAIZE_MIOU_NOTE, "",
         f"P0 mean full-run acc used by the decision rule: {fmt(p0_mean)} "
         f"(over {sum(1 for r in runs if r['pipeline'] == 'P0' and r['complete'])} complete"
         f" seed(s); the rule needs {N_SEEDS_REQUIRED})", "",
         "DATA CHECK (cloud counts vs P0):"]
        + (["  " + w for w in count_warnings] or ["  OK: all runs match P0's cloud counts"])
        + ["", "RUN WARNINGS:"] + (run_warns or ["  none"]))
    (out_dir / "pp_summary.txt").write_text(report + "\n")
    n_p0 = sum(1 for r in runs if r["pipeline"] == "P0" and r["complete"])
    (out_dir / "pp_summary.md").write_text(markdown_report(rows, p0_mean, n_p0, count_warnings))
    print(report)
    print(f"\nwrote {out_dir}/pp_runs.csv, pp_by_pipeline.csv, pp_summary.txt, pp_summary.md")


if __name__ == "__main__":
    main()
