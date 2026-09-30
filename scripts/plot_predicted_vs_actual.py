"""
plot_predicted_vs_actual.py

Report-figure generator (same category as tsne_feature_plot.py/plot_confusion_matrices.py):
loads a trained Block D fusion checkpoint, runs real inference on the real train/val/test scan
tables (no synthetic data), and renders a predicted-vs-actual scatter for one trait, split into
a "train+val" panel and a "held-out test" panel so a train/test generalization gap (if any) is
visible directly, not just reported as two R^2 numbers in a table.

Built specifically to make D1's documented Maize-volume generalization failure visible
(step_notes/D1_Fusion_Baseline.md: Maize train R^2=0.683, healthy; Maize test R^2=-0.270, worse
than predicting the mean; concentrated in the 2 held-out Maize plants, 88.6% of test squared
error; several individual predictions physically impossible (negative volume) or 4-6x overshoot)
-- currently only usable for D1 (the only row whose checkpoint is a bare TraitFusionHead over
B3b's frozen features with no other fusion inputs); extending to D2-D6 needs each row's own
table-building/model-construction functions, not implemented here yet.

Black-and-white rendering (CLAUDE.md style rule): species -> marker shape (circle=Tomato,
triangle=Maize), matching the convention already established for t-SNE plots in
step_notes/Report_Figures.md. No further per-plot design review needed for this shape encoding;
this file introduces one new design element not yet used elsewhere -- a two-panel train+val vs.
test split (own panels, not a third shape/fill category) -- to keep the species encoding
consistent with existing figures rather than overloading it with a third dimension.

Usage:
    python plot_predicted_vs_actual.py --term volume --out ../data/predicted_vs_actual_volume_d1.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "adapters"))

from train_d1_fusion_baseline import (  # noqa: E402
    ALL_TERMS, TraitFusionHead, build_scan_table, compute_norm_stats,
)

_DATA_DIR = _REPO_ROOT / "data"
_TRAITS_CSV = _DATA_DIR / "traits" / "pheno4d_traits_per_scan.csv"
_GROWTH_CSV = _DATA_DIR / "traits" / "pheno4d_growth_rate_targets.csv"
_FEATURES_DIR = _DATA_DIR / "features" / "Pheno4D"


def predict_table(model, table, norm_stats, term, device):
    """Runs real inference (no retraining) over every row in `table`, returns
    (actual, predicted, species) arrays in RAW (un-standardized) units for one term, restricted
    to rows where that term is valid (matches compute_metrics' own masking convention)."""
    idx = ALL_TERMS.index(term)
    mean, std = norm_stats[term]
    actual, predicted, species = [], [], []
    model.eval()
    with torch.no_grad():
        for _, row in table.iterrows():
            if not row[f"{term}_valid"]:
                continue
            with np.load(row["feat_path"]) as d:
                feat = torch.tensor(d["feat"].astype(np.float32)).unsqueeze(0).to(device)
            pred_z = model(feat)[0, idx].item()
            actual.append(row[term])
            predicted.append(pred_z * std + mean)
            species.append(row["species"])
    return np.array(actual), np.array(predicted), np.array(species)


def r2_score(actual, predicted):
    err = predicted - actual
    ss_res = (err ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")


def plot_panel(ax, actual, predicted, species, title):
    marker_by_species = {"Tomato": "o", "Maize": "^"}
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="black",
            linewidth=1, alpha=0.5, label="perfect prediction (y=x)")

    lines = []
    for sp in marker_by_species:
        mask = species == sp
        if not mask.any():
            continue
        ax.scatter(actual[mask], predicted[mask], marker=marker_by_species[sp],
                   s=60, facecolor="none", edgecolor="black", linewidths=1.3, label=sp)
        r2 = r2_score(actual[mask], predicted[mask])
        lines.append(f"{sp} R²={r2:.3f} (n={mask.sum()})")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.text(0.98, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom", family="monospace")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", required=True, choices=ALL_TERMS)
    ap.add_argument("--checkpoint", default=str(_REPO_ROOT / "results" / "D1_fusion_baseline" / "model.pt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cpu")

    train_table = build_scan_table(_DATA_DIR / "pheno4d_oracle_train.csv", _TRAITS_CSV, _GROWTH_CSV, _FEATURES_DIR)
    val_table = build_scan_table(_DATA_DIR / "pheno4d_oracle_val.csv", _TRAITS_CSV, _GROWTH_CSV, _FEATURES_DIR)
    test_table = build_scan_table(_DATA_DIR / "pheno4d_heldout_eval.csv", _TRAITS_CSV, _GROWTH_CSV, _FEATURES_DIR)
    norm_stats = compute_norm_stats(train_table)  # train-split stats only, matches training convention

    model = TraitFusionHead(in_dim=256, hidden_dim=0, dropout=0.3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    train_val_table = pd.concat([train_table, val_table], ignore_index=True)

    tv_actual, tv_pred, tv_species = predict_table(model, train_val_table, norm_stats, args.term, device)
    test_actual, test_pred, test_species = predict_table(model, test_table, norm_stats, args.term, device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_panel(axes[0], tv_actual, tv_pred, tv_species, f"{args.term}: train+val (seen during training)")
    plot_panel(axes[1], test_actual, test_pred, test_species, f"{args.term}: held-out test (never trained on)")
    fig.suptitle(f"D1 (fusion baseline, B3b features only): predicted vs. actual {args.term}", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
