"""
plot_physics_ablation_bars.py

Report-figure generator (same category as plot_training_curves.py): renders the D5/D6
physics-loss-ablation comparison (step_notes/D5_D6_Physics_Loss.md's "Three-way per-term test R^2"
tables) as a grouped bar chart instead of a table, per explicit user instruction -- this is one of
the project's most rigorously-verified findings (a real confound found and isolated via a proper
control ablation, see step_notes/D5_D6_Physics_Loss.md) and currently only existed as text tables.

Parses each row's FINAL test metrics DIRECTLY from its already-committed run.log (same per-term
R^2 lines train_d1_fusion_baseline.py::compute_metrics prints) -- not hand-copied from the
step_notes tables, to avoid any transcription risk between what was verified there and what this
figure shows.

Four bars per term: D4 (baseline, no physics loss at all), Ablation (lambda_phys=lambda_mono=0,
same warm-start/schedule as D5/D6 -- D5's and D6's ablation runs are numerically IDENTICAL, see
step_notes/D5_D6_Physics_Loss.md, "expected... the ODE family choice is irrelevant once its loss
weight is zero" -- shown once here, not twice), D5 (Logistic ODE physics loss), D6 (Gompertz ODE
physics loss).

Black-and-white rendering (CLAUDE.md style rule): all bars white/black-edge, distinguished by
HATCH PATTERN, not color or grayscale shading -- the standard academic B&W bar-chart convention,
distinct from both the pure-black-marker t-SNE design and the Greys-colormap confusion-matrix
design already established in step_notes/Report_Figures.md (a third chart type, own convention,
not a departure from either).
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _REPO_ROOT / "results"

TERM_LINE_RE = re.compile(
    r"(\w+)\s*:\s*n=\s*\d+,\s*RMSE=[\d.]+,\s*MAE=[\d.]+,\s*R2=([\d.\-]+)")

TERMS = ["height", "stem_diameter", "leaf_area", "leaf_count", "volume", "height_rate", "stem_diameter_rate"]

ROWS = [
    ("D4\n(baseline)", "D4_fusion_previous_stage", dict(facecolor="black", hatch=None)),
    ("Ablation\n(λ=0)", "D5_ablation_no_physics", dict(facecolor="white", hatch="///")),
    ("D5\n(Logistic)", "D5_fusion_physics_logistic", dict(facecolor="white", hatch="...")),
    ("D6\n(Gompertz)", "D6_fusion_physics_gompertz", dict(facecolor="white", hatch="xxx")),
]


def parse_final_test_r2(run_log_path):
    """Parses the FINAL test metrics block's per-term R^2 values directly from a row's run.log
    (the same lines train_d1_fusion_baseline.py::compute_metrics prints) -- returns {term: r2}."""
    r2 = {}
    in_test_block = False
    with open(run_log_path) as f:
        for line in f:
            if "FINAL test metrics" in line:
                in_test_block = True
                continue
            if in_test_block:
                m = TERM_LINE_RE.search(line)
                if m and m.group(1) in TERMS:
                    r2[m.group(1)] = float(m.group(2))
                    if len(r2) == len(TERMS):
                        break
    return r2


def main():
    data = {}
    for label, dirname, style in ROWS:
        run_log = _RESULTS / dirname / "run.log"
        r2 = parse_final_test_r2(run_log)
        missing = [t for t in TERMS if t not in r2]
        if missing:
            raise RuntimeError(f"{dirname}: missing R2 for terms {missing} -- check run.log parsing")
        data[label] = r2
        print(f"{label.replace(chr(10), ' ')}: {r2}")

    n_terms = len(TERMS)
    n_rows = len(ROWS)
    bar_width = 0.8 / n_rows
    x = range(n_terms)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, (label, _dirname, style) in enumerate(ROWS):
        offsets = [xi + (i - (n_rows - 1) / 2) * bar_width for xi in x]
        heights = [data[label][t] for t in TERMS]
        ax.bar(offsets, heights, width=bar_width, edgecolor="black", linewidth=1,
               label=label.replace("\n", " "), **style)

    ax.set_xticks(list(x))
    ax.set_xticklabels(TERMS, rotation=20, ha="right")
    ax.set_ylabel("Test R²")
    ax.set_title("D5/D6 physics-loss ablation: test R² per term\n"
                  "D4 baseline vs. control ablation (λ=0) vs. Logistic (D5) vs. Gompertz (D6) physics loss",
                  fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True, ncol=4, fontsize=9)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.grid(axis="y", color="0.85", linewidth=0.5)
    fig.tight_layout()

    out_path = _REPO_ROOT / "data" / "physics_ablation_bars.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
