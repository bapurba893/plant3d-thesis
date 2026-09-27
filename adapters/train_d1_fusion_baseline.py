"""
train_d1_fusion_baseline.py

Block D, row D1 (F1: "deep geometric features only"). Predicts all 5
Pheno4D traits (L_trait) and 2 growth rates (L_growth, height and
stem_diameter only) from B3b's (KPConv+DA-S) frozen pooled feature.

**Backbone is frozen -- enforced structurally, not by a runtime flag.**
This file never imports KPConv_ClsSeg, PlantKPConvConfig, or any KPConv
module at all -- it only reads the 256-dim features already cached by
adapters/extract_b3b_features.py (data/features/Pheno4D/<species>/
<stem>.npz). There is no encoder in this script's computation graph to
accidentally unfreeze: `model` below (TraitFusionHead) is a plain ~35K-
parameter MLP, and its own docstring/parameter count printed at startup
is the audit trail that nothing KPConv-sized ever enters the optimizer.
This is the strongest form of "frozen" available -- not
requires_grad=False on a still-loaded encoder (which a future edit
could silently flip), but the encoder's absence from the file entirely.

**Loss**: L_trait (5 traits) = Huber + lambda_corr*(1-pearson_corr),
L_growth (height, stem_diameter only) = MSE. All 7 terms combined via
adapters.losses.KendallUncertaintyWeighting (regression-type). Each
term is masked to the samples in the current batch that have a valid
(non-NaN, non-low-confidence-flagged) target for THAT SPECIFIC term --
computed independently per term, so a NaN/flagged value in one term
(e.g. leaf_area on a sparse scan, or stem_diameter/stem_diameter_rate
on one of the 6 stem_leaf_boundary_low_confidence scans) never zeroes
out, corrupts, or otherwise affects the other 6 terms' loss for that
same sample -- see `masked_regression_loss` and `TARGET_VALIDITY`.

**Data**: reuses the Oracle-style plant-disjoint split already
established for A5/B5/C5 (data/pheno4d_oracle_train.csv: 8 plants/72
scans train, data/pheno4d_oracle_val.csv: 2 plants/18 scans val,
data/pheno4d_heldout_eval.csv: 4 plants/63 scans test), joined against
D0's per-scan traits (data/traits/pheno4d_traits_per_scan.csv), D1's
growth-rate targets (data/traits/pheno4d_growth_rate_targets.csv), and
D1's cached features (data/features/Pheno4D/) by (plant_id, scan_date).
Targets are z-score standardized using train-split statistics only
(computed over each term's own valid subset), predictions un-
standardized before computing/reporting RMSE/MAE/R^2.
"""

import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses import KendallUncertaintyWeighting  # noqa: E402

TRAIT_TERMS = ["height", "stem_diameter", "leaf_area", "leaf_count", "volume"]
GROWTH_TERMS = ["height_rate", "stem_diameter_rate"]
ALL_TERMS = TRAIT_TERMS + GROWTH_TERMS
# Which raw trait/growth values are affected by the confirmed segmentation
# failure on the 6 stem_leaf_boundary_low_confidence scans (see
# step_notes/D0_Trait_Extraction_Pipeline.md's ground-truth comparison):
# stem_diameter, leaf_area, leaf_count corrupted; height, volume unaffected.
# height_rate never touches segmentation-derived stem/leaf points at all
# (it's the fitted-curve derivative); stem_diameter_rate already excludes
# these scans at the source (compute_growth_rate_targets.py drops them
# before differencing), so its own NaN already encodes this -- listed
# here too for a single, consistent lookup used everywhere in this file.
FLAG_AFFECTED_TERMS = {"stem_diameter", "leaf_area", "leaf_count"}


class IOStream:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.f = open(os.path.join(path, "run.log"), "a")

    def cprint(self, text):
        ts = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
        line = f"{ts}: {text}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def build_scan_table(split_csv, traits_csv, growth_csv, features_dir):
    """Joins a plant-disjoint split CSV against D0's per-scan traits, D1's
    growth-rate targets, and D1's cached features, by (plant_id,
    scan_date). Returns a DataFrame with one row per resolvable scan:
    feat_path + every entry in ALL_TERMS + a `<term>_valid` bool column
    per term."""
    split = pd.read_csv(split_csv)
    traits = pd.read_csv(traits_csv)
    growth = pd.read_csv(growth_csv)

    merged = split.merge(traits, on=["plant_id", "scan_date"], how="inner", suffixes=("", "_t"))
    merged = merged.merge(growth, on=["plant_id", "scan_date"], how="inner", suffixes=("", "_g"))

    rows = []
    for _, row in merged.iterrows():
        stem = Path(row["filepath"]).stem
        species = row["species"]
        feat_path = Path(features_dir) / species / f"{stem}.npz"
        if not feat_path.exists():
            print(f"[warn] no cached feature for {species}/{stem}, skipping")
            continue
        entry = {"plant_id": row["plant_id"], "species": species, "scan_date": row["scan_date"],
                  "feat_path": str(feat_path)}
        low_conf = bool(row["stem_leaf_boundary_low_confidence"])
        for term in ALL_TERMS:
            val = row[term]
            valid = not pd.isna(val) and not (low_conf and term in FLAG_AFFECTED_TERMS)
            entry[term] = float(val) if not pd.isna(val) else 0.0  # placeholder, masked out if invalid
            entry[f"{term}_valid"] = valid
        rows.append(entry)
    return pd.DataFrame(rows)


class D1FusionDataset(Dataset):
    def __init__(self, table: pd.DataFrame, norm_stats: dict):
        self.table = table.reset_index(drop=True)
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        with np.load(row["feat_path"]) as d:
            feat = d["feat"].astype(np.float32)
        targets, valids = [], []
        for term in ALL_TERMS:
            mean, std = self.norm_stats[term]
            raw = row[term]
            z = (raw - mean) / std if row[f"{term}_valid"] else 0.0
            targets.append(z)
            valids.append(1.0 if row[f"{term}_valid"] else 0.0)
        return feat, np.array(targets, dtype=np.float32), np.array(valids, dtype=np.float32)


def compute_norm_stats(table: pd.DataFrame) -> dict:
    stats = {}
    for term in ALL_TERMS:
        valid_vals = table.loc[table[f"{term}_valid"], term]
        mean = float(valid_vals.mean()) if len(valid_vals) else 0.0
        std = float(valid_vals.std()) if len(valid_vals) > 1 else 1.0
        if std < 1e-8:
            std = 1.0
        stats[term] = (mean, std)
    return stats


class TraitFusionHead(nn.Module):
    """The ENTIRE model D1 trains -- no encoder, see module docstring.
    256 (B3b's frozen pooled feature) -> [hidden_dim ->]  7 outputs (5
    traits + 2 growth rates, standardized units).

    **Sized down after an explicit overfitting check, twice (2026-09-27)**:
    a 256->128->7 head (33,799 params) trained on synthetic data showed
    train R^2 = 0.995-0.997 on every term but held-out val R^2 only
    0.11-0.38 with the SAME clean, fully-informative synthetic signal --
    classic n<<p overfitting, not a masking/training-loop bug. A first
    fix (256->32->7, 8,455 params, weight_decay 1e-4->1e-2) helped some
    (mean val R^2 0.24->0.37) but a direct A/B test against the even
    smaller `hidden_dim=0` option (below) showed val R^2 keeps rising
    monotonically as capacity drops further (0.37->0.53), with train R^2
    barely moving (0.971->0.957) -- i.e. no signal was being lost by
    cutting capacity, only overfitting. So `hidden_dim=0` (a direct
    256->7 linear head, Dropout on the input only, ~1.8K params) is now
    the DEFAULT, not just a fallback option -- the real D1 dataset (~72
    training scans) is smaller than the 160-sample synthetic test that
    showed this trend, so the same direction is expected to matter even
    more there. Full comparison table in step_notes/D1_Fusion_Baseline.md.
    A larger `hidden_dim` remains available via `--hidden_dim N` if real
    training data (more samples/regularization than the synthetic test
    could probe) turns out to tolerate more capacity after all."""

    def __init__(self, in_dim=256, hidden_dim=0, dropout=0.3):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(ALL_TERMS)),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, len(ALL_TERMS)),
            )

    def forward(self, feat):
        return self.net(feat)  # (B, len(ALL_TERMS))


def masked_regression_loss(pred, target, valid, loss_type, lambda_corr=0.5):
    """pred, target, valid: (B,) tensors for ONE term. Computes the loss
    ONLY over valid==1 samples -- an invalid (NaN/flagged) sample in this
    term contributes nothing to this term's loss and, since every term is
    computed independently from the others via its own valid mask, has NO
    effect on any other term's loss for that same sample. Returns None if
    fewer than 2 valid samples remain in the batch (Pearson correlation is
    undefined below 2 points; MSE technically works with 1, but treated
    the same way for consistency and because a single-point Huber/MSE is
    a noisy, low-value gradient signal anyway) -- caller must skip this
    term's Kendall contribution entirely for the batch when this happens,
    not substitute a zero (a zero loss would falsely tell Kendall's
    learned uncertainty this term was perfectly predicted)."""
    mask = valid > 0.5
    n_valid = int(mask.sum().item())
    if n_valid < 2:
        return None
    p, t = pred[mask], target[mask]
    if loss_type == "trait":
        huber = F.smooth_l1_loss(p, t)  # PyTorch's smooth_l1_loss IS Huber (beta=1 default)
        p_c, t_c = p - p.mean(), t - t.mean()
        denom = torch.sqrt((p_c ** 2).sum()) * torch.sqrt((t_c ** 2).sum()) + 1e-8
        corr = (p_c * t_c).sum() / denom
        return huber + lambda_corr * (1.0 - corr)
    elif loss_type == "growth":
        return F.mse_loss(p, t)
    raise ValueError(loss_type)


def run_epoch(model, kendall, loader, device, optimizer=None, lambda_corr=0.5):
    """optimizer=None -> eval mode, no backward. Returns dict of
    {term: (sum_sq_err, sum_abs_err, sum_y, sum_y2, sum_pred_y, n)} in
    STANDARDIZED units (caller un-standardizes for reporting) plus the
    mean total Kendall-weighted loss actually optimized."""
    is_train = optimizer is not None
    model.train(is_train)
    stats = {term: {"pred": [], "target": [], "valid": []} for term in ALL_TERMS}
    total_loss_sum, n_batches = 0.0, 0

    for feat, targets, valids in loader:
        feat, targets, valids = feat.to(device), targets.to(device), valids.to(device)
        with torch.set_grad_enabled(is_train):
            pred = model(feat)  # (B, len(ALL_TERMS))
            losses = {}
            for i, term in enumerate(ALL_TERMS):
                loss_type = "trait" if term in TRAIT_TERMS else "growth"
                term_loss = masked_regression_loss(
                    pred[:, i], targets[:, i], valids[:, i], loss_type, lambda_corr)
                if term_loss is not None:
                    losses[term] = term_loss
            if losses:
                total = kendall(losses)
                if is_train:
                    optimizer.zero_grad()
                    total.backward()
                    optimizer.step()
                total_loss_sum += float(total.item())
                n_batches += 1

        for i, term in enumerate(ALL_TERMS):
            stats[term]["pred"].append(pred[:, i].detach().cpu().numpy())
            stats[term]["target"].append(targets[:, i].detach().cpu().numpy())
            stats[term]["valid"].append(valids[:, i].detach().cpu().numpy())

    mean_total_loss = total_loss_sum / max(n_batches, 1)
    return stats, mean_total_loss


def compute_metrics(stats, norm_stats):
    """Un-standardizes predictions/targets and computes RMSE/MAE/R^2 per
    term over that term's own valid subset only."""
    results = {}
    for term in ALL_TERMS:
        pred = np.concatenate(stats[term]["pred"])
        target = np.concatenate(stats[term]["target"])
        valid = np.concatenate(stats[term]["valid"]) > 0.5
        if valid.sum() < 2:
            results[term] = {"n": int(valid.sum()), "rmse": float("nan"),
                              "mae": float("nan"), "r2": float("nan")}
            continue
        mean, std = norm_stats[term]
        p_real = pred[valid] * std + mean
        t_real = target[valid] * std + mean
        err = p_real - t_real
        rmse = float(np.sqrt((err ** 2).mean()))
        mae = float(np.abs(err).mean())
        ss_res = float((err ** 2).sum())
        ss_tot = float(((t_real - t_real.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")
        results[term] = {"n": int(valid.sum()), "rmse": rmse, "mae": mae, "r2": r2}
    return results


def parse_args():
    p = argparse.ArgumentParser(description="D1: fusion baseline (F1, deep features only)")
    p.add_argument("--exp_name", type=str, default="D1_fusion_baseline")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2,
                    help="raised from an initial 1e-4 after a synthetic overfitting check "
                         "(train R2 ~0.996, val R2 0.11-0.38 with a clean signal) -- see "
                         "TraitFusionHead's docstring and step_notes/D1_Fusion_Baseline.md")
    p.add_argument("--hidden_dim", type=int, default=0,
                    help="0 (default) = direct linear head (256->7, dropout only, no hidden "
                         "layer) -- an A/B synthetic test showed val R2 rising monotonically as "
                         "capacity drops (128->32->0 hidden units), train R2 barely moving; see "
                         "TraitFusionHead's docstring and step_notes/D1_Fusion_Baseline.md")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lambda_corr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", type=int, default=-1, help="-1 for cpu")
    return p.parse_args()


def main():
    args = parse_args()
    exp_path = os.path.join(args.out_path, args.exp_name)
    io = IOStream(exp_path)
    io.cprint(str(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    io.cprint(f"Using {'GPU ' + str(args.gpu) if cuda else 'CPU'}")

    data_dir = _REPO_ROOT / "data"
    traits_csv = data_dir / "traits" / "pheno4d_traits_per_scan.csv"
    growth_csv = data_dir / "traits" / "pheno4d_growth_rate_targets.csv"
    features_dir = data_dir / "features" / "Pheno4D"

    train_table = build_scan_table(data_dir / "pheno4d_oracle_train.csv", traits_csv, growth_csv, features_dir)
    val_table = build_scan_table(data_dir / "pheno4d_oracle_val.csv", traits_csv, growth_csv, features_dir)
    test_table = build_scan_table(data_dir / "pheno4d_heldout_eval.csv", traits_csv, growth_csv, features_dir)
    io.cprint(f"train: {len(train_table)} scans, val: {len(val_table)}, test: {len(test_table)}")

    norm_stats = compute_norm_stats(train_table)
    io.cprint(f"norm_stats (mean, std), fit on train split only: {norm_stats}")

    train_set = D1FusionDataset(train_table, norm_stats)
    val_set = D1FusionDataset(val_table, norm_stats)
    test_set = D1FusionDataset(test_table, norm_stats)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = TraitFusionHead(in_dim=256, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    io.cprint(f"TraitFusionHead parameter count: {n_params} "
              f"(confirms this is a small MLP, not a KPConv-scale encoder -- "
              f"see module docstring's 'frozen backbone' audit-trail note)")

    kendall = KendallUncertaintyWeighting({term: "regression" for term in ALL_TERMS}).to(device)
    opt = optim.Adam(list(model.parameters()) + list(kendall.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(opt, args.epochs)

    best_val_loss, best_epoch = float("inf"), 0
    best_model_state = None
    for epoch in range(args.epochs):
        _, train_loss = run_epoch(model, kendall, train_loader, device, optimizer=opt, lambda_corr=args.lambda_corr)
        scheduler.step()
        val_stats, val_loss = run_epoch(model, kendall, val_loader, device, optimizer=None, lambda_corr=args.lambda_corr)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            io.cprint(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), os.path.join(exp_path, "model.pt"))

    io.cprint(f"Best model at epoch {best_epoch}, val total loss {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)

    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        stats, _ = run_epoch(model, kendall, loader, device, optimizer=None, lambda_corr=args.lambda_corr)
        metrics = compute_metrics(stats, norm_stats)
        io.cprint(f"\n=== FINAL {split_name} metrics (best epoch {best_epoch}) ===")
        for term, m in metrics.items():
            io.cprint(f"  {term:20s}: n={m['n']:3d}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()
