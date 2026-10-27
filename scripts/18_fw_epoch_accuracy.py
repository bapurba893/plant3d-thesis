"""
18_fw_epoch_accuracy.py

For every FW A-C row and seed (54 runs: results/FW_<row>_s<seed>/run.log; C1 =
results/PP_P4a_s<seed>/run.log, which is FW C1, see step_notes/FW_1_Results.md), reads the
per-epoch target (Pheno4D held-out, 63 clouds) accuracy logged during training and reports four
numbers side by side:

  final_epoch_acc     target accuracy at the last training epoch
  best_epoch_acc      the HIGHEST per-epoch target accuracy, and its epoch. **This picks the epoch
                      using the target test labels, so it is an upper bound (an oracle choice of
                      epoch), not a result any method could report.**
  selected_acc        target accuracy of the validation-selected checkpoint: the "FINAL target"
                      line (lowest source val total loss; Oracle rows A5, B5, C5: lowest Oracle val
                      total loss). Target labels are never used for this selection.
  full_run_mean_acc   mean of the per-epoch target accuracy over all epochs (the FW primary metric)

Parsing reuses scripts/15_fw_summarize.py (its parse_log for the selected checkpoint, full-run mean,
completeness and warnings, and its per-epoch target regex); nothing there is modified. PointNet++
note (15_fw_summarize.FPS_NOTE): the FINAL re-evaluation of the selected checkpoint can differ by a
cloud or two from the same epoch's logged value.

Outputs (in --out_dir, default results/fw_epoch_accuracy/):
  fw_epoch_accuracy_runs.csv  one line per run
  fw_epoch_accuracy_rows.csv  one line per row (3-seed mean and sample SD of each number)
  fw_epoch_accuracy.md        both tables

Usage:
  python scripts/18_fw_epoch_accuracy.py
"""

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent

SEEDS = (1, 2, 3)
METRICS = [
    ("final_epoch_acc", "Final epoch"),
    ("best_epoch_acc", "Best epoch (uses target labels: upper bound)"),
    ("selected_acc", "Validation-selected"),
    ("full_run_mean_acc", "Full-run mean"),
]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fw = _load("fw_summarize", "15_fw_summarize.py")


def run_dir(results_dir: Path, row: str, seed: int) -> Path:
    return results_dir / (f"PP_P4a_s{seed}" if row == "C1" else f"FW_{row}_s{seed}")


def per_epoch_target(log_path: Path) -> dict:
    """{epoch: target acc} from the LAST run in run.log (same rule as 15_fw_summarize.parse_log)."""
    lines = log_path.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if fw.pp.NAMESPACE_RE.match(ln)]
    trgt = {}
    for ln in lines[starts[-1]:]:
        if m := fw.TRGT_EPOCH_RE.match(ln):
            trgt[int(m.group("e"))] = float(m.group("acc"))
    return trgt


def parse_run(results_dir: Path, row: str, seed: int) -> dict:
    d = run_dir(results_dir, row, seed)
    r = {"row": row, "seed": seed, "run_dir": f"results/{d.name}", "oracle": fw.is_oracle(row)}
    base, warnings = fw.parse_log(d / "run.log", row)
    if base is None:
        r.update(complete=False, warnings="; ".join(warnings))
        return r
    trgt = per_epoch_target(d / "run.log")
    last = max(trgt)
    best_e = max(sorted(trgt), key=lambda e: trgt[e])  # first epoch reaching the maximum
    r.update(complete=base["complete"], n_epochs=len(trgt),
             final_epoch=last, final_epoch_acc=trgt[last],
             best_epoch=best_e, best_epoch_acc=trgt[best_e],
             best_epoch_ties=sum(v == trgt[best_e] for v in trgt.values()),
             selected_epoch=base["selected_epoch"], selected_acc=base["selected_acc"],
             full_run_mean_acc=base["full_run_mean_acc"], warnings="; ".join(warnings))
    return r


def fmt(v):
    return "–" if v is None else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results_dir", default=str(_REPO_ROOT / "results"))
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "results" / "fw_epoch_accuracy"))
    args = ap.parse_args()
    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    runs = pd.DataFrame([parse_run(results_dir, row, s) for row in fw.ROW_INFO for s in SEEDS])
    runs.to_csv(out_dir / "fw_epoch_accuracy_runs.csv", index=False)

    rows = []
    for row in fw.ROW_INFO:
        g = runs[(runs.row == row) & (runs.complete == True)]  # noqa: E712
        backbone, da, aug, _ = fw.ROW_INFO[row]
        r = {"row": row, "desc": f"{backbone} / {da} / {aug}", "oracle": fw.is_oracle(row),
             "n_seeds": len(g)}
        for k, _ in METRICS:
            v = list(g[k])
            r[f"{k}_mean"] = statistics.fmean(v) if v else None
            r[f"{k}_sd"] = statistics.stdev(v) if len(v) > 1 else None
        r["best_minus_selected_mean"] = (r["best_epoch_acc_mean"] - r["selected_acc_mean"]
                                         if len(g) else None)
        rows.append(r)
    rows = pd.DataFrame(rows)
    rows.to_csv(out_dir / "fw_epoch_accuracy_rows.csv", index=False)

    L = ["# FW target accuracy: final epoch, best epoch, validation-selected, full-run mean", "",
         "Written by `scripts/18_fw_epoch_accuracy.py` from the 54 FW run.logs (C1 = `PP_P4a_s<seed>`). "
         "Target = Pheno4D held-out, 63 clouds. SD = sample SD over seeds.", "",
         "**Best epoch uses the target test labels to choose the epoch. It is an upper bound (an "
         "oracle choice of epoch), not a result any method could report.** Validation-selected = the "
         "checkpoint at the lowest source (Oracle rows: Oracle) validation loss, chosen without target "
         "labels; this is the honest single-checkpoint number. Full-run mean = mean over all epochs.",
         "", f"PointNet++: {fw.FPS_NOTE}", "",
         "## Per row (mean ± SD over 3 seeds)", "",
         "| Row | Backbone / DA / Aug | " + " | ".join(l for _, l in METRICS)
         + " | Best − selected | Seeds |",
         "|---|---|" + "---|" * (len(METRICS) + 2)]
    for r in rows.itertuples():
        v = r._asdict()
        cells = [f"{fmt(v[f'{k}_mean'])} ± {fmt(v[f'{k}_sd'])}" for k, _ in METRICS]
        L.append(f"| {r.row}{' (Oracle)' if r.oracle else ''} | {r.desc} | " + " | ".join(cells)
                 + f" | {fmt(r.best_minus_selected_mean)} | {r.n_seeds} |")
    L += ["", "## Per run", "",
          "| Run | Final epoch (e) | Best epoch, target labels (e) | Validation-selected (e) "
          "| Full-run mean | Warnings |", "|---|---|---|---|---|---|"]
    for r in runs.itertuples():
        if not r.complete:
            L.append(f"| {r.run_dir} | incomplete | | | | {r.warnings} |")
            continue
        ties = f", {r.best_epoch_ties} epochs tie" if r.best_epoch_ties > 1 else ""
        L.append(f"| {r.run_dir} | {r.final_epoch_acc:.4f} ({r.final_epoch}) "
                 f"| {r.best_epoch_acc:.4f} ({r.best_epoch}{ties}) "
                 f"| {r.selected_acc:.4f} ({r.selected_epoch}) | {r.full_run_mean_acc:.4f} "
                 f"| {r.warnings or ''} |")
    (out_dir / "fw_epoch_accuracy.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    n_bad = int((runs.complete != True).sum())  # noqa: E712
    if n_bad:
        print(f"{n_bad} run(s) incomplete", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
