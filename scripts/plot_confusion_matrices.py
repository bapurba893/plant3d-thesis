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

from dataset import PlantSpeciesDataset, SEG_NUM_CLASSES, PHENO4D_SEG_NUM_CLASSES, IDX_TO_SPECIES  # noqa: E402
from tsne_feature_plot import _load_state_dict_lenient  # noqa: E402

_MANIFEST_CSV = _REPO_ROOT / "data" / "preprocessed" / "preprocessed_manifest.csv"

# Fixed seed set immediately before each row's evaluation pass (not a training seed -- these
# checkpoints are already trained). Needed because the vendored PointNet++ backbone's
# farthest_point_sample draws an unseeded random starting centroid on every forward call, so
# eval-mode inference is NOT deterministic even with frozen weights -- confirmed and quantified
# in step_notes/PointNet2_Eval_Nondeterminism.md (std 0.010-0.027, range 0.032-0.079 across 10
# unseeded repeats on 3 fixed checkpoints). Applied uniformly to all three backbones here (not
# just PointNet2) for figure-to-figure reproducibility even though DGCNN/KPConv are believed
# deterministic already (no FPS in either architecture) -- harmless either way, and removes any
# doubt for a reader comparing this script's numbers against another run of the same script.
EVAL_SEED = 0

# Oracle rows (A5/B5/C5) train directly on Pheno4D's OWN per-point organ labels (3-class
# soil/stem/leaf scheme for BOTH species, see PHENO4D_SEG_NUM_CLASSES docstring in dataset.py) --
# a different label space than every other row's Crops3D-shaped SEG_NUM_CLASSES (Tomato=3,
# Maize=6). Model construction must match whichever scheme a given checkpoint was actually
# trained with, or state_dict loading fails on the segmentation head's shape (found the hard way:
# A5 raised a size-mismatch error on seg_heads.Maize before this was added).
_ORACLE_DIRNAME_PREFIXES = ("A5_", "B5_", "C5_")


def _seg_num_classes_for(dirname):
    if dirname.startswith(_ORACLE_DIRNAME_PREFIXES):
        return PHENO4D_SEG_NUM_CLASSES
    return SEG_NUM_CLASSES


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


def confusion_dgcnn(checkpoint_path, device, dirname=None):
    from models import DGCNN_ClsSeg
    model_args = argparse.Namespace(model="dgcnn", cuda=False, dropout=0.5)
    model = DGCNN_ClsSeg(model_args, num_class=2,
                          seg_num_classes=_seg_num_classes_for(dirname or "")).to(device)
    _load_state_dict_lenient(model, checkpoint_path, device)
    model.eval()
    torch.manual_seed(EVAL_SEED)
    return _eval_target_confusion_plain_tensor(model, device)


def confusion_pointnet2(checkpoint_path, device, dirname=None):
    from models_pointnet2 import PointNet2_ClsSeg
    model_args = argparse.Namespace(dropout=0.5)
    model = PointNet2_ClsSeg(model_args, num_class=2,
                              seg_num_classes=_seg_num_classes_for(dirname or "")).to(device)
    _load_state_dict_lenient(model, checkpoint_path, device)
    model.eval()
    torch.manual_seed(EVAL_SEED)  # see EVAL_SEED docstring -- this is the backbone that actually
    # needs it (unseeded farthest_point_sample); set uniformly across all three regardless.
    return _eval_target_confusion_plain_tensor(model, device)


def confusion_kpconv(checkpoint_path, device, dirname=None, batch_size=16):
    from models_kpconv import KPConv_ClsSeg, PlantKPConvConfig
    from kpconv_collate import make_kpconv_collate_fn_cls_only, calibrate_neighborhood_limits

    # Both seeds MUST be set before calibrate_neighborhood_limits AND before KPConv_ClsSeg(...)
    # construction below -- TWO separate, independently-confirmed nondeterminism sources, not
    # one: (1) calibrate_neighborhood_limits' own calibration DataLoader uses shuffle=True with
    # no seed (adapters/kpconv_collate.py), consuming torch's RNG; (2) the vendored KPConv-PyTorch
    # kernel-point layer (kernels/kernel_points.py::load_kernels) applies a FRESH RANDOM ROTATION
    # to the convolution kernel's geometry on every model construction via np.random.rand/normal
    # -- NumPy's global RNG, not torch's, and NOT part of the saved state_dict (recomputed at
    # __init__ every time, even loading the same trained weights). Seeding torch alone (the first,
    # incomplete fix) did NOT resolve the measured variance -- confirmed by a second repeated-call
    # test after that fix still showing 0.4286-0.5079. Both together are needed; see step_notes/
    # KPConv_Eval_Nondeterminism.md for the full investigation and final verification.
    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)

    target_csv = _REPO_ROOT / "data" / "pheno4d_heldout_eval.csv"
    tgt_set = PlantSpeciesDataset(target_csv, _MANIFEST_CSV, augment=False)

    config = PlantKPConvConfig()
    config.neighborhood_limits = calibrate_neighborhood_limits(
        config, tgt_set, batch_size, num_workers=0, collate_fn_factory=make_kpconv_collate_fn_cls_only)

    model = KPConv_ClsSeg(config, num_class=2,
                           seg_num_classes=_seg_num_classes_for(dirname or "")).to(device)
    _load_state_dict_lenient(model, checkpoint_path, device)
    model.eval()
    # NOT re-seeding here: doing so would silently discard whatever RNG state the eval loader
    # (shuffle=False, but any last calibration draws still matter for reproducibility of THIS
    # specific run) is at -- the single seed call above, before calibration, is what makes the
    # whole call (calibration + eval) reproducible end to end.

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
        y_true, y_pred = confusion_fn(checkpoint, device, dirname=dirname)
        acc = (y_true == y_pred).mean()
        print(f"  n={len(y_true)}, acc={acc:.4f}")
        out_path = out_dir / f"confusion_{dirname}.png"
        plot_confusion_bw(y_true, y_pred, f"{label}\ntarget held-out, acc={acc:.4f}", out_path)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
