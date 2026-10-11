"""
09_shortcut_probe.py

Block PP: the shortcut-score probe from docs/PREPROCESSING_COMPARISON_PLAN.md
section 7, item 5 -- "logistic regression on non-shape features (raw point
count, mean nearest-neighbour spacing of the 4096 points, flatness of the
lowest 10% of points), trained on Crops3D species, tested on Pheno4D. Report
source accuracy and target accuracy."

For each pipeline -- P0 (the original, committed data/preprocessed/ cache,
read-only here, never regenerated) and P1-P4d (data/preprocessed_variants/
<ID>/, generated in this block's Stage 2) -- fits two feature sets:
  - "all3": n_raw (from the manifest) + nn_spacing + flatness
  - "cache_only": nn_spacing + flatness only -- the two features derivable
    purely from the cached 4096-point cloud itself, without the manifest's
    raw-cloud stat. Isolates whether a shortcut survives once raw point
    count (the single most obviously dataset-correlated nuisance variable,
    per plan section 3: "Raw point count alone separates Crops3D species
    with AUC 0.988") is no longer available to the probe at all.

Trained on data/crops3d_train.csv (species label, 0=Tomato/1=Maize, matching
adapters/dataset.py::SPECIES_TO_IDX), evaluated on data/crops3d_val.csv
(source) and data/pheno4d_heldout_eval.csv (target) -- target species labels
are used ONLY to score this probe, never to train it, the same convention
every other row in this project follows for target-domain evaluation.

No sklearn in this environment: logistic regression is fit via Newton's
method / IRLS (exact and stable for this few-feature, few-hundred-sample
problem; a small ridge term guards against the near-perfect separation the
plan's own section 3 already documents for raw point count) and AUC via the
closed-form Mann-Whitney-U / rank-sum identity -- both standard, no
approximation beyond float arithmetic.

Manifest path resolution duplicates (does not import, to avoid a torch
dependency) the same key-matching logic as adapters/dataset.py's
_strip_leading_dotdot/_load_manifest_lookup, so a cloud is matched to its
split-CSV row exactly the way training will actually resolve it.

Usage:
    python scripts/09_shortcut_probe.py
Writes:
    results/pp_summary/shortcut_probe.csv
    results/pp_summary/shortcut_probe.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

SPECIES_TO_IDX = {"Tomato": 0, "Maize": 1}
UP_AXIS = 2  # x,y,z column order is never altered by any pipeline (alignment is identity
             # for every dataset currently in scope -- see preprocessing_variants.py).

PIPELINES = {
    "P0": _REPO_ROOT / "data" / "preprocessed" / "preprocessed_manifest.csv",
    "P1": _REPO_ROOT / "data" / "preprocessed_variants" / "P1" / "preprocessed_manifest.csv",
    "P2": _REPO_ROOT / "data" / "preprocessed_variants" / "P2" / "preprocessed_manifest.csv",
    "P3": _REPO_ROOT / "data" / "preprocessed_variants" / "P3" / "preprocessed_manifest.csv",
    "P4": _REPO_ROOT / "data" / "preprocessed_variants" / "P4" / "preprocessed_manifest.csv",
    "P4a": _REPO_ROOT / "data" / "preprocessed_variants" / "P4a" / "preprocessed_manifest.csv",
    "P4b": _REPO_ROOT / "data" / "preprocessed_variants" / "P4b" / "preprocessed_manifest.csv",
    "P4c": _REPO_ROOT / "data" / "preprocessed_variants" / "P4c" / "preprocessed_manifest.csv",
    "P4d": _REPO_ROOT / "data" / "preprocessed_variants" / "P4d" / "preprocessed_manifest.csv",
}

SPLITS = {
    "crops3d_train": _REPO_ROOT / "data" / "crops3d_train.csv",
    "crops3d_val": _REPO_ROOT / "data" / "crops3d_val.csv",
    "pheno4d_heldout": _REPO_ROOT / "data" / "pheno4d_heldout_eval.csv",
}


def _strip_leading_dotdot(rel_path: str) -> str:
    """Identical logic to adapters/dataset.py::_strip_leading_dotdot -- duplicated
    rather than imported, since dataset.py imports torch (not installed here)."""
    rel_path = rel_path.replace("\\", "/")
    while rel_path.startswith("../"):
        rel_path = rel_path[3:]
    return rel_path


def _load_manifest_lookup(manifest_csv: Path) -> dict:
    """source_filepath (normalized) -> (cache_path (absolute), n_raw)."""
    df = pd.read_csv(manifest_csv)
    lookup = {}
    for _, row in df.iterrows():
        key = _strip_leading_dotdot(str(row["source_filepath"]))
        cache_path = _REPO_ROOT / _strip_leading_dotdot(str(row["cache_path"]))
        n_raw = row["n_raw"] if "n_raw" in df.columns and pd.notna(row.get("n_raw")) else np.nan
        lookup[key] = (cache_path, n_raw)
    return lookup


def compute_features(points: np.ndarray):
    """Returns (nn_spacing, flatness_lowest10pct) for a (4096, 3) cached cloud."""
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=2, workers=-1)
    nn_spacing = float(dist[:, 1].mean())

    n = len(points)
    n_bottom = max(10, int(np.ceil(0.10 * n)))
    order = np.argsort(points[:, UP_AXIS])
    bottom = points[order[:n_bottom]]
    c = bottom - bottom.mean(axis=0)
    _, s, _ = np.linalg.svd(c, full_matrices=False)
    eig = s ** 2
    flatness = float(eig[-1] / max(eig.sum(), 1e-12))  # small = flat (same convention as
                                                        # step_notes/PP_0's planarity metric)
    return nn_spacing, flatness


def build_feature_table(pipeline_id: str, manifest_path: Path) -> pd.DataFrame:
    """One row per cloud across all 3 splits, with n_raw/nn_spacing/flatness/species/split."""
    lookup = _load_manifest_lookup(manifest_path)
    rows = []
    n_missing = 0
    for split_name, split_path in SPLITS.items():
        split_df = pd.read_csv(split_path)
        for _, srow in split_df.iterrows():
            key = _strip_leading_dotdot(str(srow["filepath"]))
            entry = lookup.get(key)
            if entry is None:
                n_missing += 1
                continue
            cache_path, n_raw = entry
            if not cache_path.exists():
                n_missing += 1
                continue
            with np.load(cache_path) as d:
                pts = d["points"]
            nn_spacing, flatness = compute_features(pts)
            rows.append(dict(
                pipeline=pipeline_id, split=split_name, species=srow["species"],
                label=SPECIES_TO_IDX[srow["species"]],
                n_raw=float(n_raw), nn_spacing=nn_spacing, flatness=flatness,
            ))
    if n_missing:
        print(f"  [warn] {pipeline_id}: {n_missing} split rows had no matching/existing cache entry")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Logistic regression (IRLS / Newton's method) -- no sklearn in this environment
# ---------------------------------------------------------------------------

def fit_logreg(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-8,
               ridge: float = 1e-6) -> np.ndarray:
    """Standard IRLS for binary logistic regression. X excludes the intercept
    column (added internally). `ridge` is a small L2 term on all coefficients
    (intercept included) purely for numerical stability under the
    near-perfect separation plan section 3 already documents for raw point
    count -- not a meaningful regularization choice otherwise, given only
    2-3 features and a few hundred samples."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(Xb.shape[1])
    for _ in range(max_iter):
        z = np.clip(Xb @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1 - p), 1e-6, None)
        XtWX = (Xb * w[:, None]).T @ Xb + ridge * np.eye(Xb.shape[1])
        grad = Xb.T @ (y - p) - ridge * beta
        delta = np.linalg.solve(XtWX, grad)
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            break
    return beta


def predict_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((len(X), 1)), X])
    z = np.clip(Xb @ beta, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Closed-form ROC-AUC via the Mann-Whitney-U / rank-sum identity (ties
    broken by average rank) -- mathematically exact, no external dependency."""
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score))
    ranks[order] = np.arange(1, len(y_score) + 1)
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg_rank = sums[inv] / counts[inv]
    sum_ranks_pos = avg_rank[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def accuracy(y_true: np.ndarray, y_score: np.ndarray, thresh: float = 0.5) -> float:
    return float(((y_score >= thresh).astype(int) == y_true).mean())


FEATURE_SETS = {
    "all3": ["n_raw", "nn_spacing", "flatness"],
    "cache_only": ["nn_spacing", "flatness"],
}


def main():
    out_dir = _REPO_ROOT / "results" / "pp_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    for pid, manifest_path in PIPELINES.items():
        if not manifest_path.exists():
            print(f"[skip] {pid}: manifest not found at {manifest_path}")
            continue
        print(f"Processing {pid} ...")
        feat_df = build_feature_table(pid, manifest_path)
        if feat_df.empty:
            print(f"  [skip] {pid}: no rows resolved")
            continue

        train = feat_df[feat_df["split"] == "crops3d_train"]
        val = feat_df[feat_df["split"] == "crops3d_val"]
        heldout = feat_df[feat_df["split"] == "pheno4d_heldout"]

        for fs_name, cols in FEATURE_SETS.items():
            Xtr_raw = train[cols].to_numpy(dtype=np.float64)
            ytr = train["label"].to_numpy(dtype=np.float64)
            mu, sigma = Xtr_raw.mean(axis=0), Xtr_raw.std(axis=0)
            sigma[sigma == 0] = 1.0
            Xtr = (Xtr_raw - mu) / sigma

            beta = fit_logreg(Xtr, ytr)

            for split_name, split_df in (("crops3d_train", train), ("crops3d_val", val),
                                          ("pheno4d_heldout", heldout)):
                if split_df.empty:
                    continue
                X_raw = split_df[cols].to_numpy(dtype=np.float64)
                y = split_df["label"].to_numpy(dtype=np.float64)
                X = (X_raw - mu) / sigma
                p = predict_proba(X, beta)
                result_rows.append(dict(
                    pipeline=pid, feature_set=fs_name, split=split_name,
                    n=len(split_df), accuracy=accuracy(y, p), auc=auc_score(y, p),
                ))

    result_df = pd.DataFrame(result_rows)
    csv_path = out_dir / "shortcut_probe.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Short text/markdown table: source (crops3d_val) and target (pheno4d_heldout) only,
    # per plan section 7 item 5 ("Report source accuracy and target accuracy").
    report = result_df[result_df["split"].isin(["crops3d_val", "pheno4d_heldout"])].copy()
    pivot = report.pivot_table(index=["pipeline", "feature_set"],
                                columns="split", values=["accuracy", "auc"])
    pivot = pivot.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0)

    lines = ["# Shortcut probe (docs/PREPROCESSING_COMPARISON_PLAN.md section 7, item 5)", "",
             "Logistic regression on non-shape features, trained on Crops3D train (species "
             "label), scored on Crops3D val (source) and Pheno4D held-out (target). "
             "`all3` = n_raw + nn_spacing + flatness; `cache_only` = nn_spacing + flatness "
             "(no manifest-derived raw point count).", ""]
    header = f"{'pipeline':8s} {'features':11s} {'src_acc':>8s} {'src_auc':>8s} {'tgt_acc':>8s} {'tgt_auc':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    for pid in PIPELINES:
        for fs_name in FEATURE_SETS:
            sub = report[(report["pipeline"] == pid) & (report["feature_set"] == fs_name)]
            if sub.empty:
                continue
            src = sub[sub["split"] == "crops3d_val"]
            tgt = sub[sub["split"] == "pheno4d_heldout"]
            src_acc = f"{src['accuracy'].iloc[0]:.4f}" if len(src) else "n/a"
            src_auc = f"{src['auc'].iloc[0]:.4f}" if len(src) else "n/a"
            tgt_acc = f"{tgt['accuracy'].iloc[0]:.4f}" if len(tgt) else "n/a"
            tgt_auc = f"{tgt['auc'].iloc[0]:.4f}" if len(tgt) else "n/a"
            lines.append(f"{pid:8s} {fs_name:11s} {src_acc:>8s} {src_auc:>8s} {tgt_acc:>8s} {tgt_auc:>8s}")

    md_path = out_dir / "shortcut_probe.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
