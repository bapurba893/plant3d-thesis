"""
plot_confusion_matrices.py

Report-figure generator (same category as tsne_feature_plot.py) -- recomputes each curated row's
FINAL target (Pheno4D held-out) confusion matrix directly from its checkpoint (all still on disk,
gitignored but not deleted) rather than parsing the printed text block already in each run.log,
since re-running inference is more reliable than parsing a matrix that was printed as raw text.

Curated set (not all 22 trained rows, to keep the report focused): DA-0 baseline, Oracle ceiling,
and -- where one exists -- the backbone's best non-Oracle adaptation method, per backbone. DGCNN
(Block C) has no winning adaptation method (every method tried underperformed C1 itself, see
CLAUDE.md's FINAL SUMMARY), so only DA-0/Oracle are shown for that backbone.

Black-and-white rendering (CLAUDE.md style rule): cell shading uses matplotlib's "Greys"
grayscale colormap (not a hue-based color) plus printed count in every cell -- the standard
convention for B&W confusion-matrix figures, distinct in kind from the t-SNE scatter design
(which uses pure black + marker shape/fill, no shading at all, per the confirmed design in
step_notes/Report_Figures.md) -- a matrix/heatmap is a different chart type with its own
established B&W convention, not a departure from that design decision.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFREC_ROOT = os.environ.get("DEFREC_ROOT", str(_REPO_ROOT.parent / "DefRec_and_PCM"))
if _DEFREC_ROOT not in sys.path:
    sys.path.insert(0, _DEFREC_ROOT)
os.environ.setdefault("POINTNET2_ROOT", str(_REPO_ROOT.parent / "Pointnet_Pointnet2_pytorch"))
sys.path.insert(0, str(_REPO_ROOT / "adapters"))

from dataset import PlantSpeciesDataset, SEG_NUM_CLASSES, IDX_TO_SPECIES  # noqa: E402
from tsne_feature_plot import _load_state_dict_lenient  # noqa: E402

_MANIFEST_CSV = _REPO_ROOT / "data" / "preprocessed" / "preprocessed_manifest.csv"


def _eval_target_confusion_plain_tensor(model, device, batch_size=32):
    """DGCNN/PointNet2 shared eval path -- both use plain (B, 3, N) tensors and the same
    {"cls", ...} forward interface."""
    target_csv = _REPO_ROOT / "data" / "pheno4d_heldout_eval.csv"
    tgt_set = PlantSpeciesDataset(target_csv, _MANIFEST_CSV, augment=False)
    loader = DataLoader(tgt_set, batch_size=batch_size, shuffle=False)
    all_true, all_pred = [], []
    with torch.no_grad():
        for pts, labels in loader:
            pts = pts.permute(0, 2, 1).to(device)
            logits = model(pts, activate_DefRec=False)
            preds = logits["cls"].max(dim=1)[1].cpu().numpy()
            all_true.append(labels.numpy()); all_pred.append(preds)
    return np.concatenate(all_true), np.concatenate(all_pred)


def confusion_dgcnn(checkpoint_path, device):
    from models import DGCNN_ClsSeg
    model_args = argparse.Namespace(model="dgcnn", cuda=False, dropout=0.5)
    model = DGCNN_ClsSeg(model_args, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    _load_state_dict_lenient(model, checkpoint_path, device)
    model.eval()
    return _eval_target_confusion_plain_tensor(model, device)


def confusion_pointnet2(checkpoint_path, device):
    from models_pointnet2 import PointNet2_ClsSeg
    model_args = argparse.Namespace(dropout=0.5)
    model = PointNet2_ClsSeg(model_args, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    _load_state_dict_lenient(model, checkpoint_path, device)
    model.eval()
    return _eval_target_confusion_plain_tensor(model, device)


def confusion_kpconv(checkpoint_path, device, batch_size=16):
    from models_kpconv import KPConv_ClsSeg, PlantKPConvConfig
    from kpconv_collate import make_kpconv_collate_fn_cls_only, calibrate_neighborhood_limits

    target_csv = _REPO_ROOT / "data" / "pheno4d_heldout_eval.csv"
    tgt_set = PlantSpeciesDataset(target_csv, _MANIFEST_CSV, augment=False)

    config = PlantKPConvConfig()
    config.neighborhood_limits = calibrate_neighborhood_limits(
        config, tgt_set, batch_size, num_workers=0, collate_fn_factory=make_kpconv_collate_fn_cls_only)

    model = KPConv_ClsSeg(config, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    _load_state_dict_lenient(model, checkpoint_path, device)
    model.eval()

    loader = DataLoader(tgt_set, batch_size=batch_size, shuffle=False,
                         collate_fn=make_kpconv_collate_fn_cls_only(config))
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits["cls"].max(dim=1)[1].cpu().numpy()
            all_true.append(batch.species_labels.cpu().numpy()); all_pred.append(preds)
    return np.concatenate(all_true), np.concatenate(all_pred)


def plot_confusion_bw(y_true, y_pred, title, out_path):
    labels = list(IDX_TO_SPECIES.keys())
    names = [IDX_TO_SPECIES[l] for l in labels]
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max())
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            frac = val / cm.max() if cm.max() > 0 else 0
            text_color = "white" if frac > 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontsize=13)

    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


ROWS = [
    ("A1 (PointNet++, DA-0)", confusion_pointnet2, "A1_pointnet2_da0_clsseg"),
    ("A3 (PointNet++, DA-S, best)", confusion_pointnet2, "A3_pointnet2_da_s"),
    ("A5 (PointNet++, Oracle)", confusion_pointnet2, "A5_pointnet2_da_o"),
    ("B1 (KPConv, DA-0)", confusion_kpconv, "B1_kpconv_da0"),
    ("B3b (KPConv, DA-S, best)", confusion_kpconv, "B3b_kpconv_da_s"),
    ("B5 (KPConv, Oracle)", confusion_kpconv, "B5_kpconv_da_o"),
    ("C1 (DGCNN, DA-0)", confusion_dgcnn, "C1_dgcnn_da0_clsseg"),
    ("C5 (DGCNN, Oracle)", confusion_dgcnn, "C5_dgcnn_da_o"),
]


def main():
    device = torch.device("cpu")
    out_dir = _REPO_ROOT / "results" / "report_figures"
    out_dir.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    for label, confusion_fn, dirname in ROWS:
        if args.only and args.only not in dirname:
            continue
        checkpoint = _REPO_ROOT / "results" / dirname / "model.pt"
        print(f"[{label}] loading checkpoint {checkpoint}")
        y_true, y_pred = confusion_fn(checkpoint, device)
        acc = (y_true == y_pred).mean()
        print(f"  n={len(y_true)}, acc={acc:.4f}")
        out_path = out_dir / f"confusion_{dirname}.png"
        plot_confusion_bw(y_true, y_pred, f"{label}\ntarget held-out, acc={acc:.4f}", out_path)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
