"""
11_cross_eval.py

Block PP soil-cue test (docs/PREPROCESSING_COMPARISON_PLAN.md, follow-up to
step_notes/PP_2_Results.md caveat 3.4): evaluate already-trained PP checkpoints on Pheno4D
held-out clouds taken from a DIFFERENT preprocessing cache than they were trained on.

  - soil-KEPT models (P0, P1, P4a) evaluated on the SOIL-REMOVED target cache (P4)
  - soil-REMOVED models (P3, P4) evaluated on the SOIL-KEPT target cache (P4a)
  - every model also on its own cache, as the reference

Nothing is trained. Each run's results/PP_<ID>_s<seed>/model.pt is the checkpoint the C1 script
saved at its lowest source validation loss (the protocol-selected one). Model construction,
data loading and the metric code are imported from adapters/train_c1_dgcnn_da0.py (the same
DGCNN_ClsSeg, PlantSpeciesDataset, DataLoader settings and evaluate_cls_only with the same
unweighted target CE), so the own-cache numbers must reproduce the "FINAL target" line of each
run.log. Every own-cache evaluation is checked against that line; --mode check runs only that
check on one run (default PP_P4a_s1 on P4a) and exits non-zero on a mismatch.

Usage:
  python scripts/11_cross_eval.py --mode check                 # PP_P4a_s1 on P4a == 0.8413?
  python scripts/11_cross_eval.py --mode cross                 # all pairs, writes results/pp_summary/
  python scripts/11_cross_eval.py --mode check --run PP_P0_s2 --cache P0 --gpu -1
"""

import argparse
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
import train_c1_dgcnn_da0 as c1  # noqa: E402  (sets up DefRec_and_PCM on sys.path; main() not run)

HELDOUT_CSV = _REPO_ROOT / "data" / "pheno4d_heldout_eval.csv"
SEEDS = (1, 2, 3)
SOIL_KEPT = ("P0", "P1", "P4a")
SOIL_REMOVED = ("P3", "P4")
# model pipeline -> cross cache (the opposite soil treatment)
CROSS_CACHE = {**{p: "P4" for p in SOIL_KEPT}, **{p: "P4a" for p in SOIL_REMOVED}}
DROPOUT = 0.5  # train_c1_dgcnn_da0.py default; used for every PP run (jobs/pp_compare.sbatch)

FINAL_RE = re.compile(r"FINAL target \(Pheno4D held-out\) test accuracy: ([0-9.]+), avg acc: ([0-9.]+)")


def manifest_for(cache: str) -> Path:
    if cache == "P0":
        return _REPO_ROOT / "data" / "preprocessed" / "preprocessed_manifest.csv"
    return _REPO_ROOT / "data" / "preprocessed_variants" / cache / "preprocessed_manifest.csv"


def logged_final(run: str):
    """(acc, bal_acc) from the run.log FINAL target line, or None."""
    log = _REPO_ROOT / "results" / run / "run.log"
    m = None
    for line in log.read_text().splitlines():
        mm = FINAL_RE.search(line)
        if mm:
            m = mm
    return (float(m.group(1)), float(m.group(2))) if m else None


def load_model(run: str, device, cuda: bool):
    model_args = argparse.Namespace(model="dgcnn", cuda=cuda, dropout=DROPOUT)
    model = c1.DGCNN_ClsSeg(model_args, num_class=2, seg_num_classes=c1.SEG_NUM_CLASSES).to(device)
    state = torch.load(_REPO_ROOT / "results" / run / "model.pt", map_location=device)
    model.load_state_dict(state)
    return model


def evaluate(run: str, cache: str, device, cuda: bool):
    model = load_model(run, device, cuda)
    ds = c1.PlantSpeciesDataset(HELDOUT_CSV, manifest_for(cache), augment=False)
    if len(ds) != 63:
        raise RuntimeError(f"{cache}: expected 63 held-out clouds, resolved {len(ds)}")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=c1.NWORKERS)
    acc, bal_acc, loss, conf, _, _ = c1.evaluate_cls_only(model, loader, nn.CrossEntropyLoss(), device)
    return acc, bal_acc, loss, conf


def check_own(run, cache, acc, bal_acc):
    """Own-cache result vs the run.log FINAL line, compared at the log's 4 decimals."""
    logged = logged_final(run)
    if logged is None:
        return None, f"{run}: no FINAL target line in run.log"
    ok = round(acc, 4) == logged[0] and round(bal_acc, 4) == logged[1]
    return ok, (f"{run} on {cache}: evaluated acc {acc:.4f} / bal {bal_acc:.4f}, "
                f"logged {logged[0]:.4f} / {logged[1]:.4f} -> {'MATCH' if ok else 'MISMATCH'}")


def write_outputs(rows, out_dir: Path):
    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "cross_eval_runs.csv", index=False)

    own = runs[runs.eval_cache == runs.model_pipeline].set_index(["model_pipeline", "seed"])
    runs["d_acc"] = [r.acc - own.loc[(r.model_pipeline, r.seed), "acc"] for r in runs.itertuples()]
    runs["d_bal_acc"] = [r.bal_acc - own.loc[(r.model_pipeline, r.seed), "bal_acc"]
                         for r in runs.itertuples()]

    agg = []
    for p in SOIL_KEPT + SOIL_REMOVED:
        for cache in (p, CROSS_CACHE[p]):
            g = runs[(runs.model_pipeline == p) & (runs.eval_cache == cache)]
            agg.append({
                "model_pipeline": p,
                "model_soil": "kept" if p in SOIL_KEPT else "removed",
                "eval_cache": cache,
                "eval_kind": "own" if cache == p else "cross",
                "seeds": ",".join(str(s) for s in sorted(g.seed)),
                "acc_mean": g.acc.mean(), "acc_std": g.acc.std(ddof=1),
                "bal_acc_mean": g.bal_acc.mean(), "bal_acc_std": g.bal_acc.std(ddof=1),
                "d_acc_mean": g.d_acc.mean(), "d_bal_acc_mean": g.d_bal_acc.mean(),
                "d_acc_per_seed": " / ".join(f"{v:+.3f}" for v in g.sort_values("seed").d_acc),
            })
    agg = pd.DataFrame(agg)
    agg.to_csv(out_dir / "cross_eval.csv", index=False)

    lines = [
        "**Block PP soil-cue test: cross-evaluation** (no retraining; protocol-selected "
        "checkpoints `results/PP_<ID>_s<seed>/model.pt`; target = Pheno4D held-out, 63 clouds; "
        "mean ± std over seeds 1,2,3)",
        "",
        "Soil-kept models (P0, P1, P4a) evaluated on the soil-removed target cache (P4); "
        "soil-removed models (P3, P4) on the soil-kept target cache (P4a). Change = cross minus "
        "the same model's own-cache result, averaged over seeds (per-seed values in the last "
        "column).",
        "",
        "| Model pipeline | Model soil | Evaluated on cache | Target acc | Target bal. acc "
        "| Δ acc vs own | Δ bal. acc vs own | Δ acc per seed (s1 / s2 / s3) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in agg.itertuples():
        own_row = r.eval_kind == "own"
        lines.append(
            f"| {r.model_pipeline} | {r.model_soil} | {r.eval_cache}{' (own)' if own_row else ''} "
            f"| {r.acc_mean:.3f} ± {r.acc_std:.3f} | {r.bal_acc_mean:.3f} ± {r.bal_acc_std:.3f} "
            f"| {'reference' if own_row else f'{r.d_acc_mean:+.3f}'} "
            f"| {'reference' if own_row else f'{r.d_bal_acc_mean:+.3f}'} "
            f"| {'–' if own_row else r.d_acc_per_seed} |")
    lines += ["", "Majority baseline (always Tomato): 40/63 = 0.635. Own-cache rows reproduce "
              "each run.log's FINAL target line (checked per run; see cross_eval_runs.csv, "
              "column `own_matches_log`). P3, P4 and P4c caches are identical in data "
              "(step_notes/PP_2_Results.md caveat 3.1), so P3-model-on-P4-cache equals "
              "P3 on its own cache and is not listed as a cross run."]
    (out_dir / "cross_eval.md").write_text("\n".join(lines) + "\n")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["check", "cross"], default="check")
    ap.add_argument("--run", default="PP_P4a_s1", help="check mode: run folder under results/")
    ap.add_argument("--cache", default="P4a", help="check mode: cache to evaluate on")
    ap.add_argument("--gpu", type=int, default=0, help="-1 for cpu")
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "results" / "pp_summary"))
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print(f"device: {device}", flush=True)

    if args.mode == "check":
        acc, bal, loss, conf = evaluate(args.run, args.cache, device, cuda)
        print(f"{args.run} on {args.cache}: acc {acc:.4f}, bal acc {bal:.4f}, loss {loss:.4f}\n{conf}")
        ok, msg = check_own(args.run, args.cache, acc, bal)
        print(msg, flush=True)
        sys.exit(0 if ok else 1)

    # cross mode: guard first -- the reference run must reproduce its log before anything else
    acc, bal, _, _ = evaluate("PP_P4a_s1", "P4a", device, cuda)
    ok, msg = check_own("PP_P4a_s1", "P4a", acc, bal)
    print("GUARD " + msg, flush=True)
    if not ok:
        sys.exit("guard failed: own-cache evaluation does not reproduce run.log; nothing written")

    rows, mismatches = [], []
    for p in SOIL_KEPT + SOIL_REMOVED:
        for seed in SEEDS:
            run = f"PP_{p}_s{seed}"
            for cache in (p, CROSS_CACHE[p]):
                acc, bal, loss, conf = evaluate(run, cache, device, cuda)
                row = {"model_pipeline": p, "seed": seed, "run": run, "eval_cache": cache,
                       "acc": acc, "bal_acc": bal, "loss": loss,
                       "conf_TT": conf[0, 0], "conf_TM": conf[0, 1],
                       "conf_MT": conf[1, 0], "conf_MM": conf[1, 1],
                       "own_matches_log": ""}
                if cache == p:
                    ok, msg = check_own(run, cache, acc, bal)
                    row["own_matches_log"] = ok
                    if not ok:
                        mismatches.append(msg)
                print(f"{run} on {cache}: acc {acc:.4f}, bal acc {bal:.4f}"
                      f"{'  [own]' if cache == p else ''}", flush=True)
                rows.append(row)

    agg = write_outputs(rows, Path(args.out_dir))
    print("\n" + (Path(args.out_dir) / "cross_eval.md").read_text())
    if mismatches:
        print("OWN-CACHE MISMATCHES:\n" + "\n".join(mismatches), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
