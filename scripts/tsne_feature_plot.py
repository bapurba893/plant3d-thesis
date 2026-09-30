"""
tsne_feature_plot.py

Report-figure generator (same category as scripts/02_visualize_pointclouds.py) -- extracts a
trained backbone's pooled global feature for a sample of source (Crops3D) and target (Pheno4D)
scans, runs t-SNE, and renders a BLACK-AND-WHITE plot (CLAUDE.md style rule: "Documents/reports:
black and white only, no color" -- so species/domain are distinguished by marker SHAPE and FILL,
never color).

Design confirmed by the user (2026-09-29) on a single DGCNN/C1 example before generating the
full curated set -- see step_notes/Report_Figures.md for the design decision writeup and the
Maize-overlap/Tomato-separation finding that example surfaced. No further per-plot design review
needed; this file now supports all three backbones for the curated set:
  - DGCNN (extract_features_dgcnn): plain (B, 3, N) tensor forward, one sample at a time.
  - PointNet++ (extract_features_pointnet2): identical plain-tensor pattern, same interface
    (logits["feat"]).
  - KPConv (extract_features_kpconv): needs the stacked-batch collate machinery
    (adapters/kpconv_collate.py) -- calibrates neighborhood_limits over BOTH domains (same
    reasoning as B2/B3/B3b: this extraction routes target points through the encoder too), then
    batches properly (collate calls are the expensive part, so batched not per-sample).

Uses currently-uncommitted checkpoints: results/*/model.pt is gitignored (results/**/*.pt) but
still physically present on disk from the original training runs -- no retraining needed.
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
from sklearn.manifold import TSNE

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFREC_ROOT = os.environ.get("DEFREC_ROOT", str(_REPO_ROOT.parent / "DefRec_and_PCM"))
if _DEFREC_ROOT not in sys.path:
    sys.path.insert(0, _DEFREC_ROOT)
_POINTNET2_ROOT = os.environ.get("POINTNET2_ROOT", str(_REPO_ROOT.parent / "Pointnet_Pointnet2_pytorch"))
os.environ.setdefault("POINTNET2_ROOT", _POINTNET2_ROOT)
sys.path.insert(0, str(_REPO_ROOT / "adapters"))

from dataset import PlantClsSegDataset, PlantSpeciesDataset, SEG_NUM_CLASSES, IDX_TO_SPECIES  # noqa: E402

_MANIFEST_CSV = _REPO_ROOT / "data" / "preprocessed" / "preprocessed_manifest.csv"
_SOURCE_CSV = _REPO_ROOT / "data" / "crops3d_val.csv"
_TARGET_CSV = _REPO_ROOT / "data" / "pheno4d_heldout_eval.csv"


def _subsample_indices(n, cap, rng):
    idx = np.arange(n)
    if n > cap:
        idx = rng.choice(idx, size=cap, replace=False)
    return idx


def _load_state_dict_lenient(model, checkpoint_path, device):
    """strict=False: some early checkpoints (e.g. A1, trained before the DefRec head was added
    to PointNet2_ClsSeg/DGCNN_ClsSeg for later DA-S rows) predate optional sub-modules the model
    class now always constructs. Feature extraction only ever reads logits["feat"], never
    DefRec's output, so missing DefRec.* weights are harmless here -- reported, not silently
    swallowed, so an unexpected/unrelated missing key would still be visible."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    result = model.load_state_dict(state_dict, strict=False)
    unexpected_missing = [k for k in result.missing_keys if not k.startswith("DefRec.")]
    if unexpected_missing or result.unexpected_keys:
        print(f"  WARNING: unexpected state_dict mismatch -- missing (non-DefRec): "
              f"{unexpected_missing}, unexpected: {result.unexpected_keys}")


# ---------------------------------------------------------------------------
# DGCNN
# ---------------------------------------------------------------------------

def extract_features_dgcnn(checkpoint_path, device, max_per_domain=150, seed=0):
    from models import DGCNN_ClsSeg

    model_args = argparse.Namespace(model="dgcnn", cuda=False, dropout=0.5)
    model = DGCNN_ClsSeg(model_args, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    _load_state_dict_lenient(model, checkpoint_path, device)
    model.eval()
    torch.manual_seed(seed)  # reproducibility across repeated runs of this script -- see
    # step_notes/PointNet2_Eval_Nondeterminism.md (applies to PointNet2 specifically; set here
    # too for consistency/figure-to-figure reproducibility, harmless for DGCNN either way)

    rng = np.random.default_rng(seed)
    src_set = PlantClsSegDataset(_SOURCE_CSV, _MANIFEST_CSV, augment=False)
    tgt_set = PlantSpeciesDataset(_TARGET_CSV, _MANIFEST_CSV, augment=False)

    feats, domains, species = [], [], []
    with torch.no_grad():
        for i in _subsample_indices(len(src_set), max_per_domain, rng):
            pts, species_label, _seg_label = src_set[i]
            pts_t = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
            logits = model(pts_t, activate_DefRec=False)
            feats.append(logits["feat"].squeeze(0).cpu().numpy())
            domains.append("source"); species.append(IDX_TO_SPECIES[int(species_label)])
        for i in _subsample_indices(len(tgt_set), max_per_domain, rng):
            pts, species_label = tgt_set[i]
            pts_t = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
            logits = model(pts_t, activate_DefRec=False)
            feats.append(logits["feat"].squeeze(0).cpu().numpy())
            domains.append("target"); species.append(IDX_TO_SPECIES[int(species_label)])

    return np.stack(feats), np.array(domains), np.array(species)


# ---------------------------------------------------------------------------
# PointNet++
# ---------------------------------------------------------------------------

def extract_features_pointnet2(checkpoint_path, device, max_per_domain=150, seed=0):
    from models_pointnet2 import PointNet2_ClsSeg

    model_args = argparse.Namespace(dropout=0.5)
    model = PointNet2_ClsSeg(model_args, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    _load_state_dict_lenient(model, checkpoint_path, device)
    model.eval()
    torch.manual_seed(seed)  # PointNet2's farthest_point_sample is unseeded/nondeterministic at
    # eval time otherwise -- see step_notes/PointNet2_Eval_Nondeterminism.md

    rng = np.random.default_rng(seed)
    src_set = PlantClsSegDataset(_SOURCE_CSV, _MANIFEST_CSV, augment=False)
    tgt_set = PlantSpeciesDataset(_TARGET_CSV, _MANIFEST_CSV, augment=False)

    feats, domains, species = [], [], []
    with torch.no_grad():
        for i in _subsample_indices(len(src_set), max_per_domain, rng):
            pts, species_label, _seg_label = src_set[i]
            pts_t = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
            logits = model(pts_t, activate_DefRec=False)
            feats.append(logits["feat"].squeeze(0).cpu().numpy())
            domains.append("source"); species.append(IDX_TO_SPECIES[int(species_label)])
        for i in _subsample_indices(len(tgt_set), max_per_domain, rng):
            pts, species_label = tgt_set[i]
            pts_t = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
            logits = model(pts_t, activate_DefRec=False)
            feats.append(logits["feat"].squeeze(0).cpu().numpy())
            domains.append("target"); species.append(IDX_TO_SPECIES[int(species_label)])

    return np.stack(feats), np.array(domains), np.array(species)


# ---------------------------------------------------------------------------
# KPConv
# ---------------------------------------------------------------------------

def extract_features_kpconv(checkpoint_path, device, max_per_domain=150, seed=0, batch_size=16):
    from models_kpconv import KPConv_ClsSeg, PlantKPConvConfig
    from kpconv_collate import (
        make_kpconv_collate_fn, make_kpconv_collate_fn_cls_only,
        calibrate_neighborhood_limits, combine_neighborhood_limits,
    )

    rng = np.random.default_rng(seed)
    src_full = PlantClsSegDataset(_SOURCE_CSV, _MANIFEST_CSV, augment=False)
    tgt_full = PlantSpeciesDataset(_TARGET_CSV, _MANIFEST_CSV, augment=False)

    config = PlantKPConvConfig()
    src_limits = calibrate_neighborhood_limits(config, src_full, batch_size, num_workers=0)
    tgt_limits = calibrate_neighborhood_limits(config, tgt_full, batch_size, num_workers=0,
                                                collate_fn_factory=make_kpconv_collate_fn_cls_only)
    config.neighborhood_limits = combine_neighborhood_limits(src_limits, tgt_limits)

    model = KPConv_ClsSeg(config, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    _load_state_dict_lenient(model, checkpoint_path, device)
    model.eval()
    torch.manual_seed(seed)  # see the two functions above -- set uniformly across all three
    # backbones for consistency, though KPConv has no known eval-time randomness source

    src_idx = _subsample_indices(len(src_full), max_per_domain, rng)
    tgt_idx = _subsample_indices(len(tgt_full), max_per_domain, rng)
    src_subset = torch.utils.data.Subset(src_full, src_idx)
    tgt_subset = torch.utils.data.Subset(tgt_full, tgt_idx)

    src_loader = DataLoader(src_subset, batch_size=batch_size, shuffle=False,
                             collate_fn=make_kpconv_collate_fn(config))
    tgt_loader = DataLoader(tgt_subset, batch_size=batch_size, shuffle=False,
                             collate_fn=make_kpconv_collate_fn_cls_only(config))

    feats, domains, species = [], [], []
    with torch.no_grad():
        for batch in src_loader:
            batch = batch.to(device)
            logits = model(batch)
            for f, sp in zip(logits["feat"].cpu().numpy(), batch.species_labels.cpu().numpy()):
                feats.append(f); domains.append("source"); species.append(IDX_TO_SPECIES[int(sp)])
        for batch in tgt_loader:
            batch = batch.to(device)
            logits = model(batch)
            for f, sp in zip(logits["feat"].cpu().numpy(), batch.species_labels.cpu().numpy()):
                feats.append(f); domains.append("target"); species.append(IDX_TO_SPECIES[int(sp)])

    return np.stack(feats), np.array(domains), np.array(species)


# ---------------------------------------------------------------------------
# Plotting (backbone-agnostic)
# ---------------------------------------------------------------------------

def plot_tsne_bw(feats, domains, species, title, out_path):
    """Grayscale-only, per CLAUDE.md's "Documents/reports: black and white only, no color" style
    rule. species -> marker SHAPE ('o' Tomato, '^' Maize); domain -> marker FILL (solid black =
    source, hollow/white edge-black = target). User-confirmed design, see
    step_notes/Report_Figures.md."""
    emb = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(feats)

    marker_by_species = {"Tomato": "o", "Maize": "^"}
    fill_by_domain = {"source": dict(facecolor="black", edgecolor="black"),
                       "target": dict(facecolor="none", edgecolor="black")}

    fig, ax = plt.subplots(figsize=(6, 6))
    for sp in marker_by_species:
        for dom in fill_by_domain:
            mask = (species == sp) & (domains == dom)
            if not mask.any():
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       marker=marker_by_species[sp], s=70, linewidths=1.2,
                       label=f"{sp} ({dom})", **fill_by_domain[dom])

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Curated set
# ---------------------------------------------------------------------------

ROWS = [
    ("C1 (DGCNN, DA-0)", extract_features_dgcnn, "C1_dgcnn_da0_clsseg"),
    ("A1 (PointNet++, DA-0)", extract_features_pointnet2, "A1_pointnet2_da0_clsseg"),
    ("B1 (KPConv, DA-0)", extract_features_kpconv, "B1_kpconv_da0"),
    ("B3b (KPConv, DA-S)", extract_features_kpconv, "B3b_kpconv_da_s"),
]


def main():
    device = torch.device("cpu")
    out_dir = _REPO_ROOT / "results" / "report_figures"
    out_dir.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                         help="substring to filter which rows to run (default: all)")
    args = parser.parse_args()

    for label, extract_fn, dirname in ROWS:
        if args.only and args.only not in dirname:
            continue
        checkpoint = _REPO_ROOT / "results" / dirname / "model.pt"
        print(f"[{label}] loading checkpoint {checkpoint}")
        feats, domains, species = extract_fn(checkpoint, device)
        print(f"  {len(feats)} total scans: "
              f"{(domains == 'source').sum()} source, {(domains == 'target').sum()} target")
        out_path = out_dir / f"tsne_{dirname}.png"
        plot_tsne_bw(feats, domains, species,
                     title=f"{label}: source vs. target feature space", out_path=out_path)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
