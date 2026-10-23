"""
16_fw_cross_eval.py

Full-workflow rerun (FW) soil-cue test: evaluate the already-trained FW checkpoints (all trained
on the soil-kept preprocessing P4a) on the Pheno4D held-out clouds of two target caches:

  - P4a (own cache, soil kept) -- the reference, must reproduce each run.log's FINAL target line
  - P4  (soil removed)         -- the cross cache

Same 63 clouds in the same order in both caches (data/pheno4d_heldout_eval.csv). Nothing is
trained. Checkpoint = results/FW_<row>_s<seed>/model.pt (C1: results/PP_P4a_s<seed>/model.pt,
which is FW C1, see step_notes/FW_1_Results.md), i.e. the model each training script saved at
its lowest source (Oracle rows: Oracle) validation loss.

Per backbone, model construction, target Dataset/DataLoader settings and the evaluation function
are taken from that backbone's training script (identical across the rows of one backbone,
checked: `evaluate_cls_only` has the same body in every train_a*/train_b*/train_c* script):
  - A (PointNet++): PointNet2_ClsSeg + train_a1_pointnet2_da0.evaluate_cls_only
  - B (KPConv):     KPConv_ClsSeg + make_kpconv_collate_fn_cls_only + train_b1_kpconv_da0.evaluate_cls_only
  - C (DGCNN):      DGCNN_ClsSeg + train_c1_dgcnn_da0.evaluate_cls_only
Oracle rows (A5, B5, C5) use seg_num_classes=PHENO4D_SEG_NUM_CLASSES. dropout and
test_batch_size are read from each run's own Namespace line in run.log; the target loss is the
unweighted CE, as in training.

KPConv: neighborhood_limits are the training-time limits printed in each run's own run.log; they
are NOT recalibrated on either cache.

PointNet++ (random farthest-point-sampling start) and KPConv (random subsampling-grid rotation)
evaluation is not repeatable, so A/B runs are evaluated `--repeats` times per cache (default 3;
torch/numpy seeded with the repeat index 0, 1, 2 before each repeat); DGCNN once (seed 0, as
scripts/11_cross_eval.py).

Checks (reported, never silently passed):
  - own cache vs run.log FINAL target line (4 decimals). DGCNN: must match. A/B: reported as
    "logged value inside the repeat min-max" (not expected to match exactly).
  - C1 (PP_P4a) vs results/pp_summary/cross_eval_runs.csv, both caches, per seed.

Outputs (in --out_dir, default results/fw_cross_eval/):
  fw_cross_eval_evals.csv  one line per run x cache x repeat
  fw_cross_eval_runs.csv   one line per run (repeat means / min / max)
  fw_cross_eval_rows.csv   one line per row (3-seed mean and sample SD)
  fw_cross_eval.md         tables + header caveats

Usage:
  python scripts/16_fw_cross_eval.py --rows C1 C2 C3 C4 C5 C6 --gpu 0
  python scripts/16_fw_cross_eval.py --gpu 0                    # all 18 rows
  python scripts/16_fw_cross_eval.py --rows B1 B2 B3 B3b B4 B5 --guard B1_s1 --gpu 0

All four output files are rewritten after every run (md header says PARTIAL until the last run),
so a killed job leaves complete files up to its last finished run. An out_dir that already has
outputs is refused unless --overwrite.
"""

import argparse
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "adapters"))

HELDOUT_CSV = _REPO_ROOT / "data" / "pheno4d_heldout_eval.csv"
CACHES = ("P4a", "P4")  # own (soil kept), cross (soil removed)
SEEDS = (1, 2, 3)
N_HELDOUT = 63
MAJORITY = 40 / 63  # always Tomato
PP_CROSS_RUNS = _REPO_ROOT / "results" / "pp_summary" / "cross_eval_runs.csv"

ROWS = {  # row -> (backbone, oracle)
    "A1": ("A", False), "A2": ("A", False), "A3": ("A", False), "A4": ("A", False),
    "A5": ("A", True), "A6": ("A", False),
    "B1": ("B", False), "B2": ("B", False), "B3": ("B", False), "B3b": ("B", False),
    "B4": ("B", False), "B5": ("B", True),
    "C1": ("C", False), "C2": ("C", False), "C3": ("C", False), "C4": ("C", False),
    "C5": ("C", True), "C6": ("C", False),
}
ROW_DESC = {
    "A1": "PointNet++ / DA-0 / ALL", "A2": "PointNet++ / DA-A / ALL", "A3": "PointNet++ / DA-S / ALL",
    "A4": "PointNet++ / DA-A / L-N", "A5": "PointNet++ / DA-O / ALL", "A6": "PointNet++ / DA-D / ALL",
    "B1": "KPConv / DA-0 / ALL", "B2": "KPConv / DA-A / ALL", "B3": "KPConv / DA-D / ALL",
    "B3b": "KPConv / DA-S / ALL", "B4": "KPConv / DA-0 / L-D", "B5": "KPConv / DA-O / ALL",
    "C1": "DGCNN / DA-0 / ALL (= PP_P4a)", "C2": "DGCNN / DA-A / ALL", "C3": "DGCNN / DA-S / ALL",
    "C4": "DGCNN / DA-A / L-N", "C5": "DGCNN / DA-O / ALL", "C6": "DGCNN / DA-D / ALL",
}

FINAL_RE = re.compile(r"FINAL target \(Pheno4D held-out\) test accuracy: ([0-9.]+), avg acc: ([0-9.]+)")
NS_RE = re.compile(r"Namespace\((.*)\)")
LIMITS_RE = re.compile(r"Calibrated neighborhood_limits.*?(?:used\)=|: )\[([0-9, ]+)\]\s*$")


def run_name(row: str, seed: int) -> str:
    return f"PP_P4a_s{seed}" if row == "C1" else f"FW_{row}_s{seed}"


def manifest_for(cache: str) -> Path:
    return _REPO_ROOT / "data" / "preprocessed_variants" / cache / "preprocessed_manifest.csv"


def read_log(run: str) -> dict:
    """Namespace args (dropout, test_batch_size), FINAL target line, KPConv limits."""
    text = (_REPO_ROOT / "results" / run / "run.log").read_text().splitlines()
    ns = next(NS_RE.search(l).group(1) for l in text if NS_RE.search(l))
    def arg(name, cast):
        m = re.search(rf"(?:^|, ){name}=([^,]+)", ns)
        return cast(m.group(1)) if m else None
    final = [FINAL_RE.search(l) for l in text if FINAL_RE.search(l)]
    limits = [LIMITS_RE.search(l) for l in text if LIMITS_RE.search(l)]
    return {
        "dropout": arg("dropout", float),
        "test_batch_size": arg("test_batch_size", int),
        "num_workers": arg("num_workers", int),
        "final": (float(final[-1].group(1)), float(final[-1].group(2))) if final else None,
        "limits": [int(v) for v in limits[-1].group(1).split(",")] if limits else None,
    }


class Backbone:
    """Model + loader + evaluation function of one backbone, imported from its training script."""

    def __init__(self, code: str):
        self.code = code
        ds = importlib.import_module("dataset")
        self.PlantSpeciesDataset = ds.PlantSpeciesDataset
        self.PHENO4D_SEG_NUM_CLASSES = ds.PHENO4D_SEG_NUM_CLASSES
        if code == "A":
            self.tr = importlib.import_module("train_a1_pointnet2_da0")
        elif code == "B":
            self.tr = importlib.import_module("train_b1_kpconv_da0")
            kc = importlib.import_module("kpconv_collate")
            self.make_collate = kc.make_kpconv_collate_fn_cls_only
        else:
            self.tr = importlib.import_module("train_c1_dgcnn_da0")
        self.SEG_NUM_CLASSES = self.tr.SEG_NUM_CLASSES

    def build(self, run: str, oracle: bool, log: dict, device, cuda: bool):
        seg = self.PHENO4D_SEG_NUM_CLASSES if oracle else self.SEG_NUM_CLASSES
        config = None
        if self.code == "A":
            model = self.tr.PointNet2_ClsSeg(argparse.Namespace(dropout=log["dropout"]),
                                             num_class=2, seg_num_classes=seg)
        elif self.code == "B":
            if not log["limits"]:
                raise RuntimeError(f"{run}: no neighborhood_limits line in run.log")
            config = self.tr.PlantKPConvConfig()
            config.dropout = log["dropout"]
            config.neighborhood_limits = list(log["limits"])
            model = self.tr.KPConv_ClsSeg(config, num_class=2, seg_num_classes=seg)
        else:
            model = self.tr.DGCNN_ClsSeg(
                argparse.Namespace(model="dgcnn", cuda=cuda, dropout=log["dropout"]),
                num_class=2, seg_num_classes=seg)
        model = model.to(device)
        state = torch.load(_REPO_ROOT / "results" / run / "model.pt", map_location=device)
        model.load_state_dict(state)  # strict: wrong head sizes fail here
        return model, config

    def loader(self, cache: str, log: dict, config):
        ds = self.PlantSpeciesDataset(HELDOUT_CSV, manifest_for(cache), augment=False)
        if len(ds) != N_HELDOUT:
            raise RuntimeError(f"{cache}: expected {N_HELDOUT} held-out clouds, resolved {len(ds)}")
        if self.code == "B":
            nw = log["num_workers"] if log["num_workers"] is not None else 0
            return ds, DataLoader(ds, batch_size=log["test_batch_size"], shuffle=False,
                                  num_workers=nw, collate_fn=self.make_collate(config))
        return ds, DataLoader(ds, batch_size=log["test_batch_size"], shuffle=False,
                              num_workers=self.tr.NWORKERS)

    def evaluate(self, model, loader, device):
        return self.tr.evaluate_cls_only(model, loader, nn.CrossEntropyLoss(), device)


def sample_keys(ds):
    """Cache-independent identity of each held-out cloud, in loader order."""
    return [(Path(p).parent.name, Path(p).name, int(y)) for p, y in ds.samples]


def evaluate_run(bb: Backbone, row: str, seed: int, oracle: bool, repeats: int, device, cuda):
    run = run_name(row, seed)
    log = read_log(run)
    model, config = bb.build(run, oracle, log, device, cuda)
    keys, out = {}, []
    for cache in CACHES:
        ds, loader = bb.loader(cache, log, config)
        keys[cache] = sample_keys(ds)
        for rep in range(repeats):
            torch.manual_seed(rep)
            np.random.seed(rep)
            acc, bal, loss, conf, _, _ = bb.evaluate(model, loader, device)
            out.append({"row": row, "backbone": bb.code, "oracle": oracle, "seed": seed, "run": run,
                        "cache": cache, "repeat": rep, "acc": acc, "bal_acc": bal, "loss": loss,
                        "tomato_as_maize": int(conf[0, 1]), "maize_as_tomato": int(conf[1, 0]),
                        "conf_TT": int(conf[0, 0]), "conf_MM": int(conf[1, 1]),
                        "test_batch_size": log["test_batch_size"],
                        "kpconv_limits": "" if config is None else str(config.neighborhood_limits)})
            print(f"{run} on {cache} rep {rep}: acc {acc:.4f}, bal {bal:.4f}, "
                  f"T->M {conf[0, 1]}, M->T {conf[1, 0]}", flush=True)
    if keys["P4a"] != keys["P4"]:
        raise RuntimeError(f"{run}: P4a and P4 held-out clouds differ in identity or order")
    return out, log


def summarise_runs(ev: pd.DataFrame, logs: dict) -> pd.DataFrame:
    rows = []
    for (row, seed), g in ev.groupby(["row", "seed"], sort=False):
        r = {"row": row, "backbone": g.backbone.iloc[0], "oracle": g.oracle.iloc[0], "seed": seed,
             "run": g.run.iloc[0], "repeats": g.repeat.nunique()}
        for cache in CACHES:
            c = g[g.cache == cache]
            for m in ("acc", "bal_acc"):
                r[f"{m}_{cache}"] = c[m].mean()
                r[f"{m}_{cache}_min"] = c[m].min()
                r[f"{m}_{cache}_max"] = c[m].max()
            r[f"tomato_as_maize_{cache}"] = c.tomato_as_maize.mean()
            r[f"maize_as_tomato_{cache}"] = c.maize_as_tomato.mean()
        r["d_acc"] = r["acc_P4"] - r["acc_P4a"]
        r["d_bal_acc"] = r["bal_acc_P4"] - r["bal_acc_P4a"]
        final = logs[r["run"]]["final"]
        r["logged_acc"], r["logged_bal_acc"] = final if final else (np.nan, np.nan)
        own = g[g.cache == "P4a"]
        if r["backbone"] == "C":
            r["own_check"] = ("MATCH" if final and round(own.acc.iloc[0], 4) == final[0]
                              and round(own.bal_acc.iloc[0], 4) == final[1] else "MISMATCH")
        else:
            inside = final and round(own.acc.min(), 4) <= final[0] <= round(own.acc.max(), 4)
            r["own_check"] = "logged inside repeat range" if inside else "logged OUTSIDE repeat range"
        rows.append(r)
    return pd.DataFrame(rows)


def pp_check(runs: pd.DataFrame):
    """C1 (PP_P4a) vs the Block PP cross-evaluation, per seed and cache."""
    if not (runs.row == "C1").any():
        return []
    pp = pd.read_csv(PP_CROSS_RUNS)
    pp = pp[pp.model_pipeline == "P4a"].set_index(["seed", "eval_cache"])
    lines = []
    for r in runs[runs.row == "C1"].itertuples():
        for cache in CACHES:
            ref = pp.loc[(r.seed, cache)]
            new_acc, new_bal = getattr(r, f"acc_{cache}"), getattr(r, f"bal_acc_{cache}")
            ok = (new_acc == ref.acc and new_bal == ref.bal_acc
                  and getattr(r, f"tomato_as_maize_{cache}") == ref.conf_TM
                  and getattr(r, f"maize_as_tomato_{cache}") == ref.conf_MT)
            lines.append(f"{r.run} on {cache}: acc {new_acc:.6f} / bal {new_bal:.6f} vs Block PP "
                         f"{ref.acc:.6f} / {ref.bal_acc:.6f} -> {'MATCH' if ok else 'MISMATCH'}")
    return lines


def summarise_rows(runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for row, g in runs.groupby("row", sort=False):
        r = {"row": row, "desc": ROW_DESC[row], "oracle": g.oracle.iloc[0],
             "seeds": ",".join(str(s) for s in sorted(g.seed))}
        for col in ("acc_P4a", "acc_P4", "d_acc", "bal_acc_P4a", "bal_acc_P4", "d_bal_acc"):
            r[f"{col}_mean"] = g[col].mean()
            r[f"{col}_sd"] = g[col].std(ddof=1) if len(g) > 1 else np.nan
        r["acc_P4_per_seed"] = " / ".join(f"{v:.4f}" for v in g.sort_values("seed").acc_P4)
        r["d_acc_per_seed"] = " / ".join(f"{v:+.4f}" for v in g.sort_values("seed").d_acc)
        r["seeds_below_majority_P4"] = int((g.acc_P4 < MAJORITY - 1e-12).sum())
        for cache in CACHES:
            r[f"tomato_as_maize_{cache}_mean"] = g[f"tomato_as_maize_{cache}"].mean()
            r[f"maize_as_tomato_{cache}_mean"] = g[f"maize_as_tomato_{cache}"].mean()
        out.append(r)
    return pd.DataFrame(out)


def fmt_pm(m, s):
    return f"{m:.4f} ± {s:.4f}" if not np.isnan(s) else f"{m:.4f}"


def rng(r, m, cache):
    lo, hi = getattr(r, f"{m}_{cache}_min"), getattr(r, f"{m}_{cache}_max")
    v = getattr(r, f"{m}_{cache}")
    return f"{v:.4f}" if r.repeats == 1 else f"{v:.4f} [{lo:.4f}–{hi:.4f}]"


def write_md(runs, rows, pp_lines, out_dir: Path, status: str = "complete"):
    L = [
        "# FW soil-cue cross-evaluation (P4a-trained models on the soil-removed P4 target)",
        "",
        f"**Status: {status}.**",
        "",
        "Written by `scripts/16_fw_cross_eval.py`. No retraining: each run's selected checkpoint "
        "(`results/FW_<row>_s<seed>/model.pt`; C1 = `results/PP_P4a_s<seed>/model.pt`) is evaluated "
        f"on the {N_HELDOUT} Pheno4D held-out clouds of the P4a cache (own, soil kept) and of the P4 "
        "cache (soil removed): same clouds, same order (checked per run).",
        "",
        "**Caveat: P4 is normalised after the soil cut.** The unit-sphere centre and scale are "
        "computed on the plant only, the voxel size is height-relative, and all 4096 FPS points "
        "fall on the plant. So on P4 the plant's scale, centre and point density also differ from "
        "P4a (P4/P4a norm_scale median 0.78 over the 63 clouds). A drop from P4a to P4 is therefore "
        "**not only the loss of the soil cue**.",
        "",
        "**KPConv:** neighborhood_limits are the training-time limits printed in each run's own "
        "`run.log`; they were not recalibrated on P4 or P4a.",
        "",
        "**Repeats:** PointNet++ and KPConv evaluation is not repeatable (random FPS start; random "
        "subsampling-grid rotation), so A/B runs are evaluated 3 times per cache (seeds 0, 1, 2): "
        "value = mean, [min–max] of the 3 repeats. DGCNN: one evaluation.",
        "",
        f"Majority baseline (always Tomato) = 40/63 = {MAJORITY:.3f}. SD = sample SD over seeds. "
        "T→M = tomato clouds predicted maize, M→T = maize clouds predicted tomato (of 40 tomato, "
        "23 maize; repeat mean for A/B).",
        "",
    ]
    for title, sel in (("Adaptation rows (non-Oracle)", ~rows.oracle),
                       ("Oracle rows (A5, B5, C5; upper bound, not pooled)", rows.oracle)):
        sub = rows[sel]
        if sub.empty:
            continue
        L += [f"## {title}: per row (3 seeds)", "",
              "| Row | Backbone / DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 "
              "| Δ bal. acc | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub.itertuples():
            L.append(
                f"| {r.row} | {r.desc} | {fmt_pm(r.acc_P4a_mean, r.acc_P4a_sd)} "
                f"| {fmt_pm(r.acc_P4_mean, r.acc_P4_sd)} | {r.d_acc_mean:+.4f} "
                f"| {fmt_pm(r.bal_acc_P4a_mean, r.bal_acc_P4a_sd)} "
                f"| {fmt_pm(r.bal_acc_P4_mean, r.bal_acc_P4_sd)} | {r.d_bal_acc_mean:+.4f} "
                f"| {r.acc_P4_per_seed} | {r.seeds_below_majority_P4}/3 "
                f"| {r.tomato_as_maize_P4a_mean:.1f} / {r.tomato_as_maize_P4_mean:.1f} "
                f"| {r.maize_as_tomato_P4a_mean:.1f} / {r.maize_as_tomato_P4_mean:.1f} |")
        L.append("")
    L += ["## Per run", "",
          "| Run | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | T→M P4a / P4 "
          "| M→T P4a / P4 | Logged FINAL acc / bal | Own-cache check |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs.itertuples():
        L.append(f"| {r.run}{' (Oracle)' if r.oracle else ''} | {rng(r, 'acc', 'P4a')} "
                 f"| {rng(r, 'acc', 'P4')} | {r.d_acc:+.4f} | {rng(r, 'bal_acc', 'P4a')} "
                 f"| {rng(r, 'bal_acc', 'P4')} "
                 f"| {r.tomato_as_maize_P4a:.1f} / {r.tomato_as_maize_P4:.1f} "
                 f"| {r.maize_as_tomato_P4a:.1f} / {r.maize_as_tomato_P4:.1f} "
                 f"| {r.logged_acc:.4f} / {r.logged_bal_acc:.4f} | {r.own_check} |")
    if pp_lines:
        L += ["", "## C1 vs Block PP cross-evaluation (`results/pp_summary/cross_eval_runs.csv`)", ""]
        L += [f"- {l}" for l in pp_lines]
    (out_dir / "fw_cross_eval.md").write_text("\n".join(L) + "\n")


def write_all(evals, logs, out_dir: Path, status: str):
    """(Re)write every output file from the evaluations so far (called after each run, so the
    files are complete up to the last finished run even if the job is killed)."""
    ev = pd.DataFrame(evals)
    ev.to_csv(out_dir / "fw_cross_eval_evals.csv", index=False)
    runs = summarise_runs(ev, logs)
    runs.to_csv(out_dir / "fw_cross_eval_runs.csv", index=False)
    rows = summarise_rows(runs)
    rows.to_csv(out_dir / "fw_cross_eval_rows.csv", index=False)
    pp_lines = pp_check(runs)
    write_md(runs, rows, pp_lines, out_dir, status)
    return runs, pp_lines


def guard(bb: Backbone, row: str, seed: int, oracle: bool, repeats: int, device, cuda) -> bool:
    """Own-cache (P4a) check of one run before anything else: the run.log FINAL target accuracy
    must lie within [min - 1 cloud, max + 1 cloud] of `repeats` evaluations."""
    run = run_name(row, seed)
    log = read_log(run)
    model, config = bb.build(run, oracle, log, device, cuda)
    _, loader = bb.loader("P4a", log, config)
    accs = []
    for rep in range(repeats):
        torch.manual_seed(rep)
        np.random.seed(rep)
        accs.append(bb.evaluate(model, loader, device)[0])
    lo, hi, tol = min(accs), max(accs), 1 / N_HELDOUT
    logged = log["final"][0] if log["final"] else None
    ok = logged is not None and lo - tol - 1e-9 <= logged <= hi + tol + 1e-9
    print(f"GUARD {run} on P4a: repeats {' / '.join(f'{a:.4f}' for a in accs)} "
          f"(min {lo:.4f}, max {hi:.4f}, tolerance 1 cloud = {tol:.4f}); logged FINAL {logged} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS),
                    help="evaluated in the order given")
    ap.add_argument("--repeats", type=int, default=3, help="per cache, PointNet++/KPConv rows only")
    ap.add_argument("--gpu", type=int, default=0, help="-1 for cpu")
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "results" / "fw_cross_eval"))
    ap.add_argument("--guard", default=None, metavar="ROW_sSEED",
                    help="e.g. B1_s1: check this run's own-cache value against run.log first "
                         "(1-cloud tolerance) and stop without evaluating anything if it fails")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into an out_dir that already has fw_cross_eval.md")
    args = ap.parse_args()

    cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print(f"device: {device}; rows: {' '.join(args.rows)}", flush=True)
    out_dir = Path(args.out_dir)
    if (out_dir / "fw_cross_eval.md").exists() and not args.overwrite:
        sys.exit(f"{out_dir} already has outputs; refusing to overwrite (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    backbones = {}
    def backbone(code):
        if code not in backbones:
            backbones[code] = Backbone(code)
        return backbones[code]

    if args.guard:
        g_row, g_seed = args.guard.rsplit("_s", 1)
        g_code, g_oracle = ROWS[g_row]
        if not guard(backbone(g_code), g_row, int(g_seed), g_oracle,
                     1 if g_code == "C" else args.repeats, device, cuda):
            print(f"GUARD FAILED: {run_name(g_row, int(g_seed))} own-cache evaluation does not "
                  "reproduce its run.log FINAL target accuracy within 1 cloud; stopping, nothing "
                  "else evaluated or written.", flush=True)
            sys.exit(2)

    n_total = len(args.rows) * len(SEEDS)
    evals, logs = [], {}
    for row in args.rows:
        code, oracle = ROWS[row]
        repeats = 1 if code == "C" else args.repeats
        for seed in SEEDS:
            out, log = evaluate_run(backbone(code), row, seed, oracle, repeats, device, cuda)
            evals += out
            logs[run_name(row, seed)] = log
            n_done = len(logs)
            status = ("complete" if n_done == n_total else
                      f"PARTIAL: {n_done} of {n_total} runs done (last: {run_name(row, seed)})")
            runs, pp_lines = write_all(evals, logs, out_dir, status)
            print(f"[written] {status}", flush=True)

    print("\n" + (out_dir / "fw_cross_eval.md").read_text(), flush=True)
    bad = [f"{r.run}: {r.own_check}" for r in runs.itertuples()
           if r.own_check in ("MISMATCH", "logged OUTSIDE repeat range")]
    bad += [l for l in pp_lines if l.endswith("MISMATCH")]
    if bad:
        print("CHECK FAILURES:\n" + "\n".join(bad), flush=True)
        sys.exit(1)
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
