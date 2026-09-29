"""
tsne_feature_plot.py

Diagnostic/report-figure generator (same category as scripts/02_visualize_pointclouds.py) --
extracts a trained backbone's pooled global feature for a sample of source (Crops3D) and target
(Pheno4D) scans, runs t-SNE, and renders a BLACK-AND-WHITE plot (CLAUDE.md style rule:
"Documents/reports: black and white only, no color" -- so species/domain are distinguished by
marker SHAPE and FILL, never color).

Currently wired for DGCNN only (row C1, results/C1_dgcnn_da0_clsseg/model.pt) -- chosen as the
first proof-of-concept row because it has the simplest, most standard (B, 3, N) forward pass of
the three backbones (no KPConv-style stacked-batch collate, no PointNet++ sys.path juggling).
Per explicit user instruction: this script renders ONE example plot for design review (is the
grayscale marker/fill scheme actually legible?) BEFORE any decision to generate the full set of
report figures across all rows/backbones. Extending to KPConv/PointNet++ rows is deferred until
the design itself is confirmed.

Uses currently-uncommitted checkpoints: results/*/model.pt is gitignored (results/**/*.pt) but
still physically present on disk from the original training runs -- no retraining needed.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFREC_ROOT = os.environ.get("DEFREC_ROOT", str(_REPO_ROOT.parent / "DefRec_and_PCM"))
if _DEFREC_ROOT not in sys.path:
    sys.path.insert(0, _DEFREC_ROOT)
sys.path.insert(0, str(_REPO_ROOT / "adapters"))

from models import DGCNN_ClsSeg  # noqa: E402
from dataset import PlantClsSegDataset, PlantSpeciesDataset, SEG_NUM_CLASSES, IDX_TO_SPECIES  # noqa: E402


def load_c1_model(checkpoint_path, device):
    model_args = argparse.Namespace(model="dgcnn", cuda=False, dropout=0.5)
    model = DGCNN_ClsSeg(model_args, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def extract_features(model, source_csv, target_csv, manifest_csv, device, max_per_domain=150, seed=0):
    """Returns (feats [N,1024] np.ndarray, domain_labels [N] 'source'/'target', species_labels
    [N] 'Tomato'/'Maize'). Subsamples each domain to at most max_per_domain scans (t-SNE doesn't
    need every scan, and this project's datasets are small-N throughout anyway)."""
    rng = np.random.default_rng(seed)

    src_set = PlantClsSegDataset(source_csv, manifest_csv, augment=False)
    tgt_set = PlantSpeciesDataset(target_csv, manifest_csv, augment=False)

    def subsample_indices(n, cap):
        idx = np.arange(n)
        if n > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        return idx

    feats, domains, species = [], [], []
    with torch.no_grad():
        for i in subsample_indices(len(src_set), max_per_domain):
            pts, species_label, _seg_label = src_set[i]
            pts_t = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
            logits = model(pts_t, activate_DefRec=False)
            feats.append(logits["feat"].squeeze(0).cpu().numpy())
            domains.append("source")
            species.append(IDX_TO_SPECIES[int(species_label)])

        for i in subsample_indices(len(tgt_set), max_per_domain):
            pts, species_label = tgt_set[i]
            pts_t = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
            logits = model(pts_t, activate_DefRec=False)
            feats.append(logits["feat"].squeeze(0).cpu().numpy())
            domains.append("target")
            species.append(IDX_TO_SPECIES[int(species_label)])

    return np.stack(feats), np.array(domains), np.array(species)


def plot_tsne_bw(feats, domains, species, title, out_path):
    """Grayscale-only, per CLAUDE.md's "Documents/reports: black and white only, no color" style
    rule. Two categorical dimensions (species, domain) encoded WITHOUT color:
      - species -> marker SHAPE ('o' Tomato, '^' Maize)
      - domain  -> marker FILL ('source' = solid black fill, 'target' = hollow/white fill with
        black edge) -- so all four (species x domain) combinations stay visually distinct at a
        glance, not just in a legend.
    """
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
                       marker=marker_by_species[sp],
                       s=70, linewidths=1.2,
                       label=f"{sp} ({dom})",
                       **fill_by_domain[dom])

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


def main():
    device = torch.device("cpu")
    checkpoint = _REPO_ROOT / "results" / "C1_dgcnn_da0_clsseg" / "model.pt"
    manifest_csv = _REPO_ROOT / "data" / "preprocessed" / "preprocessed_manifest.csv"
    source_csv = _REPO_ROOT / "data" / "crops3d_val.csv"
    target_csv = _REPO_ROOT / "data" / "pheno4d_heldout_eval.csv"

    print(f"Loading C1 (DGCNN, DA-0) checkpoint from {checkpoint}")
    model = load_c1_model(checkpoint, device)

    print("Extracting pooled features for source (Crops3D val) + target (Pheno4D heldout)...")
    feats, domains, species = extract_features(model, source_csv, target_csv, manifest_csv, device)
    print(f"  {len(feats)} total scans: "
          f"{(domains == 'source').sum()} source, {(domains == 'target').sum()} target")

    out_path = _REPO_ROOT / "results" / "C1_dgcnn_da0_clsseg" / "tsne_example.png"
    plot_tsne_bw(feats, domains, species,
                 title="C1 (DGCNN, DA-0): source vs. target feature space",
                 out_path=out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
