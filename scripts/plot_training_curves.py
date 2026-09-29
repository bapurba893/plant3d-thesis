"""
plot_training_curves.py

Report-figure generator (same category as scripts/tsne_feature_plot.py): parses the
per-epoch "Eval(no-train) - Target N, acc: ..." lines every Block A/B/C run.log already
contains (100 lines per row, one per epoch -- the exact line every step_notes "full-trajectory"
analysis in this project has been hand-parsing all along) and renders one small-multiples
figure per block (all of that block's rows overlaid), target held-out classification accuracy
vs. epoch.

Black-and-white only (CLAUDE.md style rule: "Documents/reports: black and white only, no
color") -- lines are pure black, distinguished by LINESTYLE + sparse MARKERS (every 10th epoch),
not by color/grayscale shading. No further per-plot design review needed for this figure type
per explicit user instruction ("zero remaining design decisions").
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _REPO_ROOT / "results"

EVAL_LINE_RE = re.compile(
    r"Eval\(no-train\) - Target (?:'test' )?(\d+), acc: ([\d.]+), avg acc: ([\d.]+), cls loss: ([\d.]+)")

# (label, results dir) per block -- canonical dirs only (C1 has two superseded interim dirs,
# excluded; C1_dgcnn_da0_clsseg is the final full-spec run, see CLAUDE.md).
BLOCKS = {
    "A (PointNet++)": [
        ("A1 DA-0", "A1_pointnet2_da0_clsseg"),
        ("A2 DA-A", "A2_pointnet2_da_a"),
        ("A3 DA-S", "A3_pointnet2_da_s"),
        ("A4 DA-A/L-N", "A4_pointnet2_da_a_ln"),
        ("A5 DA-O", "A5_pointnet2_da_o"),
    ],
    "B (KPConv)": [
        ("B1 DA-0", "B1_kpconv_da0"),
        ("B2 DA-A", "B2_kpconv_da_a"),
        ("B3 DA-D", "B3_kpconv_da_d"),
        ("B3b DA-S", "B3b_kpconv_da_s"),
        ("B4 DA-0/L-D", "B4_kpconv_da0_ld"),
        ("B5 DA-O", "B5_kpconv_da_o"),
    ],
    "C (DGCNN)": [
        ("C1 DA-0", "C1_dgcnn_da0_clsseg"),
        ("C2 DA-A", "C2_dgcnn_da_a"),
        ("C3 DA-S", "C3_dgcnn_da_s"),
        ("C4 DA-A/L-N", "C4_dgcnn_da_a_ln"),
        ("C5 DA-O", "C5_dgcnn_da_o"),
    ],
}

# Distinct (linestyle, marker) pairs, pure black -- enough combinations for the largest block
# (B, 6 rows) with no repeats and no reliance on color/grayscale shading.
STYLES = [
    ("-", "o"),
    ("--", "s"),
    (":", "^"),
    ("-.", "D"),
    ((0, (3, 1, 1, 1)), "v"),
    ((0, (5, 1)), "x"),
]


def parse_target_acc_trajectory(run_log_path):
    """Returns (epochs [list[int]], acc [list[float]]) from a row's run.log, one point per
    epoch, in the same "Target N, acc: X" line every prior step_notes trajectory analysis in
    this project has parsed by hand."""
    epochs, accs = [], []
    with open(run_log_path) as f:
        for line in f:
            m = EVAL_LINE_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                accs.append(float(m.group(2)))
    return epochs, accs


def plot_block(block_name, rows, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (label, dirname), (linestyle, marker) in zip(rows, STYLES):
        run_log = _RESULTS / dirname / "run.log"
        if not run_log.exists():
            print(f"  WARNING: missing {run_log}, skipping {label}")
            continue
        epochs, accs = parse_target_acc_trajectory(run_log)
        if not epochs:
            print(f"  WARNING: no Eval(no-train) lines found in {run_log}, skipping {label}")
            continue
        ax.plot(epochs, accs, linestyle=linestyle, color="black", linewidth=1.1,
                 marker=marker, markevery=10, markersize=5, markerfacecolor="white",
                 markeredgecolor="black", label=label)

    ax.set_title(f"Block {block_name}: target held-out classification accuracy vs. epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Target (Pheno4D) held-out cls accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


def main():
    out_dir = _REPO_ROOT / "results" / "report_figures"
    out_dir.mkdir(exist_ok=True)
    for block_name, rows in BLOCKS.items():
        block_letter = block_name.split()[0]
        out_path = out_dir / f"training_curves_block_{block_letter}.png"
        print(f"Block {block_name} -> {out_path}")
        plot_block(block_name, rows, out_path)


if __name__ == "__main__":
    main()
