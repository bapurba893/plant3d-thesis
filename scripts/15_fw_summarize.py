"""
15_fw_summarize.py

Full-workflow rerun (FW, jobs/fw/): summarises every results/FW_<row>_s<seed>/run.log that is
present, so it can be re-run after each submission round. Rows and seeds are discovered from the
folder names; nothing is hard-coded per round. Writes to results/pp_summary/:

  fw_by_row.csv   one line per row: per-seed values, mean, std, n for the three metrics, plus the
                  original single-seed result for the same row (if one exists) and the difference
  fw_summary.md   the same as Markdown, plus incomplete runs and warnings

Metrics (target = Pheno4D held-out, 63 clouds), all parsed from run.log, defined exactly as in
scripts/10_pp_summarize.py (whose helpers are imported, not modified):

  full_run_mean_acc      mean over all epochs of the per-epoch target accuracy -- PRIMARY metric
  full_run_mean_bal_acc  same, for the balanced ("avg acc") accuracy
  selected_acc           target accuracy of the protocol-selected checkpoint, from the
                         "FINAL target (Pheno4D held-out) test accuracy" line. Selection = lowest
                         source val total loss; for the Oracle rows (A5, B5, C5) = lowest Oracle
                         val total loss (Pheno4D oracle val split). Target test labels are never
                         used for selection.

Mean ± sample std over COMPLETE seeds only (all epochs logged + FINAL line); a run that is still
going or crashed is listed as incomplete and kept out of every aggregate.

Log-format differences vs. 10_pp_summarize.py's C1-only regexes (handled here, not by editing 10_):
  - Oracle rows log "Val - Oracle <e>", "Target 'test' <e>" and "oracle val total loss";
  - DA rows log "Crops3D train (source): N"; Oracle rows "Pheno4D oracle train: N";
  - Oracle rows log a second "Pheno4D held-out eval ... (annotated subset, seg): 36" line, so only
    the FIRST held-out line is taken as the classification count.

Comparison: each row is compared with its original single-seed run (results/<row>_<name>/run.log,
parsed with the same code). Those originals are seed 1 on the old default preprocessing
(data/preprocessed/preprocessed_manifest.csv, P0); the FW runs are seeds 1-3 on the soil-kept P4a
manifest. The data differ, so the comparison is not a pure seed-variance check.

Segmentation mIoU is deliberately not reported: it is not comparable across rows (source-val
mIoU for A-C DA rows vs. target-test mIoU on a different label space for Oracle rows).

Usage:
  python scripts/15_fw_summarize.py
"""

import argparse
import csv
import importlib.util
import re
import statistics
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = _load("pp_summarize", "10_pp_summarize.py")
sw = _load("sw_summarize", "13_sw_summarize.py")

METRICS = sw.METRICS  # [(key, label)] -- the same three metrics as sw_summary.md
PRIMARY = METRICS[0][0]

RUN_DIR_RE = re.compile(r"^FW_(?P<row>[A-E]\d+b?)_s(?P<seed>\d+)$")

# Generalised versions of 10_'s per-line regexes (see module docstring)
TRGT_EPOCH_RE = re.compile(
    pp.TS + r"Eval\(no-train\) - Target (?:'test' )?(?P<e>\d+), acc: (?P<acc>[\d.]+), "
            r"avg acc: (?P<bal>[\d.]+)")
VAL_EPOCH_RE = re.compile(pp.TS + r"Val - (?:Source|Oracle) (?P<e>\d+), cls acc: .*total loss: (?P<loss>[\d.]+)")
BEST_RE = re.compile(pp.TS + r"Best model at epoch (?P<e>\d+), (?:source|oracle) val total loss")
TRAIN_N_RE = re.compile(pp.TS + r"(?:Crops3D train|Pheno4D oracle train)[^:]*: (?P<n>\d+) ")
VAL_N_RE = re.compile(pp.TS + r"(?:Crops3D val|Pheno4D oracle val)[^:]*:\s+(?P<n>\d+) ")
MANIFEST_RE = re.compile(r"\bmanifest=(?:'(?P<path>[^']*)'|None)")

EXPECTED_HELDOUT = 63
EXPECTED_COUNTS = {False: (263, 45), True: (72, 18)}  # (train, val) for DA rows / Oracle rows

# Row descriptions (jobs/fw/README.md) and the original single-seed run folder for each row
ROW_INFO = {
    "A1": ("PointNet++", "DA-0", "ALL", "A1_pointnet2_da0_clsseg"),
    "A2": ("PointNet++", "DA-A", "ALL", "A2_pointnet2_da_a"),
    "A3": ("PointNet++", "DA-S", "ALL", "A3_pointnet2_da_s"),
    "A4": ("PointNet++", "DA-A", "L-N", "A4_pointnet2_da_a_ln"),
    "A5": ("PointNet++", "DA-O", "ALL", "A5_pointnet2_da_o"),
    "A6": ("PointNet++", "DA-D", "ALL", "A6_pointnet2_da_d"),
    "B1": ("KPConv", "DA-0", "ALL", "B1_kpconv_da0"),
    "B2": ("KPConv", "DA-A", "ALL", "B2_kpconv_da_a"),
    "B3": ("KPConv", "DA-D", "ALL", "B3_kpconv_da_d"),
    "B3b": ("KPConv", "DA-S", "ALL", "B3b_kpconv_da_s"),
    "B4": ("KPConv", "DA-0", "L-D", "B4_kpconv_da0_ld"),
    "B5": ("KPConv", "DA-O", "ALL", "B5_kpconv_da_o"),
    "C1": ("DGCNN", "DA-0", "ALL", "C1_dgcnn_da0_clsseg"),
    "C2": ("DGCNN", "DA-A", "ALL", "C2_dgcnn_da_a"),
    "C3": ("DGCNN", "DA-S", "ALL", "C3_dgcnn_da_s"),
    "C4": ("DGCNN", "DA-A", "L-N", "C4_dgcnn_da_a_ln"),
    "C5": ("DGCNN", "DA-O", "ALL", "C5_dgcnn_da_o"),
    "C6": ("DGCNN", "DA-D", "ALL", "C6_dgcnn_da_d"),
}
DEFAULT_MANIFEST = "data/preprocessed/preprocessed_manifest.csv (default, P0)"
FPS_NOTE = ("PointNet++ eval is not deterministic (random farthest-point-sampling start), so the "
            "FINAL re-evaluation of the selected checkpoint can differ by a cloud or two from that "
            "epoch's logged accuracy; see step_notes/PointNet2_Eval_Nondeterminism.md. The FINAL "
            "value is used, as in all earlier summaries.")


def row_key(row):
    m = re.match(r"([A-Z])(\d+)(.*)", row)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (row, 0, "")


def is_oracle(row):
    return ROW_INFO.get(row, ("", ""))[1] == "DA-O"


def parse_log(log_path, row):
    """Parse one run.log. Returns (metrics dict, list of warnings); metrics is None if unusable."""
    if not log_path.exists():
        return None, ["run.log missing (not started yet?)"]
    lines = log_path.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if pp.NAMESPACE_RE.match(ln)]
    if not starts:
        return None, ["no Namespace(...) header line"]
    warnings = []
    if len(starts) > 1:
        warnings.append(f"{len(starts)} runs appended in this run.log; using the last one")
    lines = lines[starts[-1]:]
    expected_epochs = int(pp.NAMESPACE_RE.match(lines[0]).group("epochs"))
    m = MANIFEST_RE.search(lines[0])
    manifest = m.group("path") if m and m.group("path") else DEFAULT_MANIFEST

    r = {"manifest": manifest, "selected_epoch": None, "selected_acc": None,
         "selected_bal_acc": None, "n_train": None, "n_val": None, "n_heldout": None}
    trgt, val_loss = {}, {}
    for ln in lines:
        if mm := TRAIN_N_RE.match(ln):
            r["n_train"] = int(mm.group("n"))
        elif mm := VAL_N_RE.match(ln):
            r["n_val"] = int(mm.group("n"))
        elif (mm := pp.HELDOUT_N_RE.match(ln)) and r["n_heldout"] is None:
            r["n_heldout"] = int(mm.group("n"))
        elif mm := VAL_EPOCH_RE.match(ln):
            val_loss[int(mm.group("e"))] = float(mm.group("loss"))
        elif mm := TRGT_EPOCH_RE.match(ln):
            trgt[int(mm.group("e"))] = (float(mm.group("acc")), float(mm.group("bal")))
        elif mm := BEST_RE.match(ln):
            r["selected_epoch"] = int(mm.group("e"))
        elif mm := pp.FINAL_TRGT_RE.match(ln):
            r["selected_acc"] = float(mm.group("acc"))
            r["selected_bal_acc"] = float(mm.group("bal"))

    r["n_epochs"] = len(trgt)
    if len(trgt) != expected_epochs:
        warnings.append(f"only {len(trgt)}/{expected_epochs} epochs logged (running, or crashed)")
    if trgt:
        r["full_run_mean_acc"] = statistics.fmean(trgt[e][0] for e in sorted(trgt))
        r["full_run_mean_bal_acc"] = statistics.fmean(trgt[e][1] for e in sorted(trgt))
    else:
        r["full_run_mean_acc"] = r["full_run_mean_bal_acc"] = None
    if r["selected_acc"] is None:
        warnings.append("no FINAL target line -- selected-checkpoint accuracy missing")
    r["complete"] = len(trgt) == expected_epochs and r["selected_acc"] is not None

    # Same cross-checks as 10_pp_summarize.py, flagged rather than overridden
    if val_loss and r["selected_epoch"] is not None:
        recomputed = min(sorted(val_loss), key=lambda e: val_loss[e])
        if recomputed != r["selected_epoch"]:
            warnings.append(f"logged best epoch {r['selected_epoch']} != argmin of logged val total "
                            f"loss {recomputed} (4-dp rounding tie?)")
        elif r["selected_epoch"] in trgt and r["selected_acc"] is not None \
                and abs(trgt[r["selected_epoch"]][0] - r["selected_acc"]) > 1e-4:
            warnings.append(f"FINAL target acc {r['selected_acc']:.4f} != epoch {r['selected_epoch']}'s "
                            f"logged {trgt[r['selected_epoch']][0]:.4f}"
                            + (" (PointNet++ eval nondeterminism)" if row.startswith("A") else ""))
    exp_train, exp_val = EXPECTED_COUNTS[is_oracle(row)]
    for k, exp in (("n_train", exp_train), ("n_val", exp_val), ("n_heldout", EXPECTED_HELDOUT)):
        if r[k] is None:
            warnings.append(f"{k} line not found")
        elif r[k] != exp:
            warnings.append(f"{k}={r[k]}, expected {exp}")
    return r, warnings


def collect_fw(results_dir):
    runs = []
    for d in sorted(results_dir.glob("FW_*")):
        m = RUN_DIR_RE.match(d.name)
        if not (d.is_dir() and m):
            continue
        row, seed = m.group("row"), int(m.group("seed"))
        r, warns = parse_log(d / "run.log", row)
        if r is None:
            r = {"complete": False, "n_epochs": 0}
        r.update(row=row, seed=seed, run_dir=f"results/{d.name}", warnings=warns)
        runs.append(r)
    return runs


def summarise(runs, results_dir):
    seeds_all = sorted({r["seed"] for r in runs})
    rows = []
    for row in sorted({r["row"] for r in runs}, key=row_key):
        rs = {r["seed"]: r for r in runs if r["row"] == row}
        complete = {s: r for s, r in rs.items() if r["complete"]}
        backbone, da, aug, orig_name = ROW_INFO.get(row, ("?", "?", "?", None))
        out = {"row": row, "backbone": backbone, "da": da, "aug": aug,
               "seeds_complete": ",".join(map(str, sorted(complete))),
               "seeds_incomplete": ",".join(str(s) for s in sorted(rs) if s not in complete),
               "manifest": ";".join(sorted({r["manifest"] for r in complete.values()}))}
        for k, _ in METRICS:
            vals = []
            for s in seeds_all:
                v = complete[s][k] if s in complete else None
                out[f"{k}_s{s}"] = v
                if v is not None:
                    vals.append(v)
            out[f"{k}_mean"], out[f"{k}_std"] = sw.mean_std(vals)
            out[f"{k}_n"] = len(vals)

        # Original single-seed run, parsed with the same code
        orig_dir = results_dir / orig_name if orig_name else None
        if orig_dir is None or not (orig_dir / "run.log").exists():
            out["orig_run"] = ""
            out["orig_note"] = "no earlier result in the repo"
        else:
            o, owarns = parse_log(orig_dir / "run.log", row)
            ns = (orig_dir / "run.log").read_text().splitlines()
            seed_m = next((re.search(r"\bseed=(\d+)", ln) for ln in ns if pp.NAMESPACE_RE.match(ln)), None)
            out["orig_run"] = f"results/{orig_name}/run.log"
            out["orig_seed"] = seed_m.group(1) if seed_m else "?"
            out["orig_manifest"] = o["manifest"] if o else "?"
            out["orig_note"] = "; ".join(owarns)
            for k, _ in METRICS:
                ov = o[k] if o and o["complete"] else None
                out[f"{k}_orig"] = ov
                m_ = out[f"{k}_mean"]
                out[f"{k}_diff"] = (m_ - ov) if (m_ is not None and ov is not None) else None
            ov = out[f"{PRIMARY}_orig"]
            vals = [out[f"{PRIMARY}_s{s}"] for s in seeds_all if out[f"{PRIMARY}_s{s}"] is not None]
            out["seeds_above_orig_primary"] = (f"{sum(v > ov for v in vals)}/{len(vals)}"
                                               if ov is not None and vals else "")
        rows.append(out)
    return rows, seeds_all


def fmt_ms(row, k, nd=4):
    m, sd = row[f"{k}_mean"], row[f"{k}_std"]
    if m is None:
        return "-"
    return f"{m:.{nd}f} ± {sd:.{nd}f}" if sd is not None else f"{m:.{nd}f} (n=1)"


def fmt_diff(v, nd=4):
    return "-" if v is None else f"{v:+.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Summarise the full-workflow (FW) rerun")
    ap.add_argument("--results_dir", default=str(_REPO_ROOT / "results"))
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "results" / "pp_summary"))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    runs = collect_fw(results_dir)
    if not runs:
        raise SystemExit("No results/FW_<row>_s<seed> folders found")
    rows, seeds = summarise(runs, results_dir)
    shown = [r for r in rows if r[f"{PRIMARY}_n"] > 0]
    pending = [r for r in rows if r[f"{PRIMARY}_n"] == 0]
    incomplete = [r for r in runs if not r["complete"]]
    warn_runs = [r for r in runs if r["complete"] and r["warnings"]]

    # ---- CSV ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = (["row", "backbone", "da", "aug", "seeds_complete", "seeds_incomplete", "manifest"]
              + [f"{k}_{x}" for k, _ in METRICS for x in [f"s{s}" for s in seeds] + ["mean", "std", "n"]]
              + ["orig_run", "orig_seed", "orig_manifest"]
              + [f"{k}_{x}" for k, _ in METRICS for x in ("orig", "diff")]
              + ["seeds_above_orig_primary", "orig_note"])
    with open(out_dir / "fw_by_row.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows({k: round(v, 6) if isinstance(v, float) else ("" if v is None else v)
                     for k, v in r.items()} for r in rows)

    # ---- terminal table ----
    term = []
    for r in shown:
        for i, (k, label) in enumerate(METRICS):
            term.append([r["row"] if i == 0 else "", label.split(" (")[0]]
                        + [pp.fmt(r[f"{k}_s{s}"]) for s in seeds] + [fmt_ms(r, k)]
                        + [pp.fmt(r.get(f"{k}_orig")), fmt_diff(r.get(f"{k}_diff"))])
    main_tbl = pp.text_table(["row", "metric"] + [f"s{s}" for s in seeds]
                             + ["FW mean ± std", "original (s1, P0)", "FW mean - orig"], term)

    not_computed = []
    for r in pending:
        not_computed.append(f"{r['row']}: no complete seed yet (incomplete: seed(s) {r['seeds_incomplete']})")
    for r in incomplete:
        not_computed.append(f"{r['run_dir']}: incomplete, excluded from all aggregates -- "
                            + "; ".join(r["warnings"]))
    for r in shown:
        if r[f"{PRIMARY}_n"] < len(seeds):
            not_computed.append(f"{r['row']}: only {r[f'{PRIMARY}_n']} complete seed(s); "
                                f"std needs >= 2, the 3-seed mean needs 3")
        if r["orig_run"] == "":
            not_computed.append(f"{r['row']}: no earlier single-seed result in the repo, no comparison")
    not_computed.append("Segmentation mIoU: not reported (not comparable across rows).")

    report = "\n".join(
        ["FW rerun (P4a, soil kept). TARGET = Pheno4D held-out, 63 clouds. "
         "Mean ± sample std over complete seeds.",
         "Original = earlier single-seed run, seed 1, old default preprocessing (P0), same script.", "",
         main_tbl, "", "NOT COMPUTED / PENDING:"] + ["  " + x for x in not_computed]
        + ["", "RUN WARNINGS (complete runs):"]
        + (["  " + f"{r['run_dir']}: " + "; ".join(r["warnings"]) for r in warn_runs] or ["  none"]))
    print(report)

    # ---- Markdown ----
    md = ["# Full-workflow rerun (FW): summary of completed rounds", "",
          "Generated by `scripts/15_fw_summarize.py` from `results/FW_<row>_s<seed>/run.log`. "
          "FW runs: soil-kept preprocessing P4a "
          "(`data/preprocessed_variants/P4a/preprocessed_manifest.csv`), seeds "
          + ", ".join(map(str, seeds)) + ", 100 epochs, arguments as in each row's original job "
          "(`jobs/fw/`). Target = Pheno4D held-out set, 63 clouds (40 tomato, 23 maize).", "",
          "Metrics: **full-run mean target accuracy** (mean over the 100 epochs of the per-epoch "
          "target accuracy; primary metric), full-run mean balanced accuracy, and "
          "**selected-checkpoint accuracy** (checkpoint with the lowest source validation loss; for "
          "the Oracle rows A5/B5/C5 the lowest Oracle validation loss on the Pheno4D oracle val "
          "split). Target test labels are never used for selection. Std = sample std over the "
          "complete seeds.", "",
          "## Per row and seed", "",
          "| Row | Backbone / DA / Aug | Metric | " + " | ".join(f"Seed {s}" for s in seeds)
          + " | Mean ± std |",
          "|---|---|---|" + "---|" * len(seeds) + "---|"]
    for r in shown:
        for i, (k, label) in enumerate(METRICS):
            md.append(f"| {r['row'] if i == 0 else ''} | "
                      + (f"{r['backbone']} / {r['da']} / {r['aug']}" if i == 0 else "")
                      + f" | {label} | " + " | ".join(pp.fmt(r[f'{k}_s{s}'], 3) for s in seeds)
                      + f" | {fmt_ms(r, k, 3)} |")
    md += ["", f"Majority-class reference (always 'Tomato'): {pp.MAJORITY_BASELINE:.3f} (40/63).", "",
           "## Comparison with the original single-seed runs", "",
           "Each original is the run in the earlier strategy table (CLAUDE.md, step_notes), read here "
           "from its own `run.log` with the same parser. **Original = seed 1 on the old default "
           "preprocessing (P0, `data/preprocessed/preprocessed_manifest.csv`). FW = seeds 1-3 on P4a "
           "(soil kept).** Both the data and the number of seeds differ, so a difference is not "
           "attributable to the preprocessing alone.", "",
           "| Row | Metric | Original (seed 1, P0) | FW mean ± std (P4a) | FW − original | "
           "FW seeds above original |",
           "|---|---|---|---|---|---|"]
    for r in shown:
        if r["orig_run"] == "":
            md.append(f"| {r['row']} | – | no earlier result in the repo | – | – | – |")
            continue
        for i, (k, label) in enumerate(METRICS):
            above = ""
            if k == PRIMARY:
                above = r["seeds_above_orig_primary"]
            md.append(f"| {r['row'] if i == 0 else ''} | {label} | {pp.fmt(r.get(f'{k}_orig'), 3)} | "
                      f"{fmt_ms(r, k, 3)} | {fmt_diff(r.get(f'{k}_diff'), 3)} | {above or '–'} |")
    md += ["", "Original run files: " + "; ".join(f"{r['row']} `{r['orig_run']}`"
                                                    for r in shown if r["orig_run"]) + "."]
    md += ["", "## Not computed / pending", ""] + [f"- {x}" for x in not_computed]
    md += ["", "## Run warnings (complete runs)", ""]
    md += ([f"- `{r['run_dir']}`: " + "; ".join(r["warnings"]) for r in warn_runs] or ["- none"])
    if any(r["row"].startswith("A") for r in warn_runs):
        md += ["", FPS_NOTE]
    (out_dir / "fw_summary.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_dir}/fw_summary.md, fw_by_row.csv")


if __name__ == "__main__":
    main()
