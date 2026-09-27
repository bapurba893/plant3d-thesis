"""
train_d3_fusion_temporal.py

Block D, row D3 (F3: "+ temporal info"). Progressive fusion, cumulative
per CLAUDE.md's Block D design ("D1 through D4 add one component at a
time"): D3's input = D2's input (B3b's 256-dim frozen feature + each
plant's 12 growth-curve params) + 1 new temporal feature, not a
replacement of D2's input.

**Design decision, flagged as an inference (not spec text)**: the design
docs say "+ temporal info" without defining it. Used `elapsed_days` (days
since that plant's own first scan, already computed in D0's
scripts/assemble_trait_timeseries.py and carried through
compute_growth_rate_targets.py's output) -- the simplest, most literal
reading of "temporal information" for a per-scan feature: how far into
this specific plant's own observed growth trajectory this scan sits.
Checked its value range before using it (2026-09-28): [0,20] for Tomato,
[0,12] for Maize, identical range across train/val/test by species, no
NaN, no outliers -- unlike the growth-curve B/Bg parameters (D2), no
clipping safeguard is needed for this feature.

**Per-term input relevance, following D2's corrected discipline --
applied, not mechanically copied**: D2 established that growth-curve
params should be withheld from leaf_area/leaf_count/volume (no
principled relevance -- those 3 traits have no growth-curve fit of their
own). elapsed_days is different: EVERY one of the 7 terms has a
principled claim on it, since every trait changes over the trajectory by
definition, and both rate terms (height_rate, stem_diameter_rate) are
themselves time-varying within a growth curve (e.g. logistic growth rate
peaks mid-trajectory, not constant) -- so restricting elapsed_days to a
subset of terms the way growth-curve params were restricted would be
applying the PATTERN (restrict inputs) without the REASONING (restrict
only where there's no principled relevance). All 7 terms get
elapsed_days appended to their existing (D2-established) input slice.

Reuses D2's build_scan_table_d2/compute_feature_norm_stats/D2FusionDataset
construction (growth-curve join, CLIP_STD clipping, per-term growth-param
slices) and D1's IOStream/ALL_TERMS/TRAIT_TERMS/build_scan_table/
compute_norm_stats/masked_regression_loss/run_epoch/compute_metrics --
none of those needed changes. Only new code: the elapsed_days join/
normalization and PerTermFusionHeadD3 (D2's PerTermFusionHead + a
universally-included temporal column per term).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses import KendallUncertaintyWeighting  # noqa: E402
from train_d1_fusion_baseline import (  # noqa: E402
    IOStream, ALL_TERMS, TRAIT_TERMS,
    build_scan_table, compute_norm_stats, masked_regression_loss,
    run_epoch, compute_metrics,
)
from train_d2_fusion_growth_curves import (  # noqa: E402
    build_scan_table_d2, compute_feature_norm_stats, GROWTH_PARAM_COLS,
    CLIP_STD, _GROWTH_RELEVANT_SLICE, _N_B3B, _N_GROWTH_PER_TRAIT,
)

_N_GROWTH_TOTAL = len(GROWTH_PARAM_COLS)  # 12
_TEMPORAL_COL_IDX = _N_B3B + _N_GROWTH_TOTAL  # column 268 in the 269-dim feat vector


def build_scan_table_d3(split_csv, traits_csv, growth_csv, growth_curves_csv, features_dir):
    """D2's scan table (feat_path + 7 targets/valid flags + 12 growth-curve
    param columns) with elapsed_days merged on from growth_csv (already
    computed there by compute_growth_rate_targets.py -- same file D2
    already reads for L_growth targets, re-read here just for this one
    extra column rather than threading it through D1/D2's functions)."""
    table = build_scan_table_d2(split_csv, traits_csv, growth_csv, growth_curves_csv, features_dir)
    elapsed = pd.read_csv(growth_csv)[["plant_id", "scan_date", "elapsed_days"]]
    merged = table.merge(elapsed, on=["plant_id", "scan_date"], how="left")
    assert merged["elapsed_days"].isna().sum() == 0, \
        "some scans had no elapsed_days -- check plant_id/scan_date consistency with growth_csv"
    return merged


def compute_temporal_norm_stats(table: pd.DataFrame) -> dict:
    """Same z-score convention as D2's compute_feature_norm_stats, for the
    single elapsed_days column -- fit on train split only."""
    vals = table["elapsed_days"]
    mean = float(vals.mean())
    std = float(vals.std()) if len(vals) > 1 else 1.0
    if std < 1e-8:
        std = 1.0
    return {"elapsed_days": (mean, std)}


class D3FusionDataset(Dataset):
    """D2's feature vector (256 B3b + 12 growth-curve params, clipped)
    with 1 more standardized column appended: elapsed_days. Resulting
    feat is (269,): [0:256]=B3b, [256:262]=height's growth params,
    [262:268]=stem_diameter's growth params, [268]=elapsed_days."""

    def __init__(self, table: pd.DataFrame, target_norm_stats: dict,
                 feature_norm_stats: dict, temporal_norm_stats: dict):
        self.table = table.reset_index(drop=True)
        self.target_norm_stats = target_norm_stats
        self.feature_norm_stats = feature_norm_stats
        self.temporal_norm_stats = temporal_norm_stats

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        with np.load(row["feat_path"]) as d:
            b3b_feat = d["feat"].astype(np.float32)
        growth_feat = np.array(
            [(row[col] - self.feature_norm_stats[col][0]) / self.feature_norm_stats[col][1]
             for col in GROWTH_PARAM_COLS], dtype=np.float32)
        growth_feat = np.clip(growth_feat, -CLIP_STD, CLIP_STD)
        t_mean, t_std = self.temporal_norm_stats["elapsed_days"]
        temporal_feat = np.array([(row["elapsed_days"] - t_mean) / t_std], dtype=np.float32)
        feat = np.concatenate([b3b_feat, growth_feat, temporal_feat])

        targets, valids = [], []
        for term in ALL_TERMS:
            mean, std = self.target_norm_stats[term]
            raw = row[term]
            z = (raw - mean) / std if row[f"{term}_valid"] else 0.0
            targets.append(z)
            valids.append(1.0 if row[f"{term}_valid"] else 0.0)
        return feat, np.array(targets, dtype=np.float32), np.array(valids, dtype=np.float32)


class PerTermFusionHeadD3(nn.Module):
    """D2's PerTermFusionHead extended with elapsed_days, included for
    EVERY term (see module docstring -- unlike growth-curve params,
    elapsed_days has a principled claim on all 7 terms, so no term is
    excluded from it). Per-term in_dim: leaf_area/leaf_count/volume =
    256+1=257; height/height_rate/stem_diameter/stem_diameter_rate =
    262+1=263."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.heads = nn.ModuleDict()
        for term in ALL_TERMS:
            in_dim = _N_B3B + (_N_GROWTH_PER_TRAIT if term in _GROWTH_RELEVANT_SLICE else 0) + 1
            self.heads[term] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 1))

    def forward(self, feat):
        b3b = feat[:, :_N_B3B]
        temporal = feat[:, _TEMPORAL_COL_IDX:_TEMPORAL_COL_IDX + 1]
        outs = []
        for term in ALL_TERMS:
            if term in _GROWTH_RELEVANT_SLICE:
                x = torch.cat([b3b, feat[:, _GROWTH_RELEVANT_SLICE[term]], temporal], dim=1)
            else:
                x = torch.cat([b3b, temporal], dim=1)
            outs.append(self.heads[term](x))
        return torch.cat(outs, dim=1)


def parse_args():
    p = argparse.ArgumentParser(description="D3: fusion (F3, + temporal info)")
    p.add_argument("--exp_name", type=str, default="D3_fusion_temporal")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
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
    growth_curves_csv = data_dir / "traits" / "pheno4d_growth_curves.csv"
    features_dir = data_dir / "features" / "Pheno4D"

    train_table = build_scan_table_d3(data_dir / "pheno4d_oracle_train.csv", traits_csv, growth_csv,
                                       growth_curves_csv, features_dir)
    val_table = build_scan_table_d3(data_dir / "pheno4d_oracle_val.csv", traits_csv, growth_csv,
                                     growth_curves_csv, features_dir)
    test_table = build_scan_table_d3(data_dir / "pheno4d_heldout_eval.csv", traits_csv, growth_csv,
                                      growth_curves_csv, features_dir)
    io.cprint(f"train: {len(train_table)} scans, val: {len(val_table)}, test: {len(test_table)}")

    target_norm_stats = compute_norm_stats(train_table)
    feature_norm_stats = compute_feature_norm_stats(train_table)
    temporal_norm_stats = compute_temporal_norm_stats(train_table)
    io.cprint(f"target norm_stats (mean, std), fit on train split only: {target_norm_stats}")
    io.cprint(f"growth-curve-param feature norm_stats (mean, std), fit on train split only: {feature_norm_stats}")
    io.cprint(f"temporal norm_stats (mean, std), fit on train split only: {temporal_norm_stats}")

    train_set = D3FusionDataset(train_table, target_norm_stats, feature_norm_stats, temporal_norm_stats)
    val_set = D3FusionDataset(val_table, target_norm_stats, feature_norm_stats, temporal_norm_stats)
    test_set = D3FusionDataset(test_table, target_norm_stats, feature_norm_stats, temporal_norm_stats)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = PerTermFusionHeadD3(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    per_term_in_dims = {term: (_N_B3B + _N_GROWTH_PER_TRAIT + 1 if term in _GROWTH_RELEVANT_SLICE else _N_B3B + 1)
                         for term in ALL_TERMS}
    io.cprint(f"PerTermFusionHeadD3 parameter count: {n_params} -- per-term in_dim: {per_term_in_dims} "
              f"(all 7 terms get elapsed_days; leaf_area/leaf_count/volume still never see growth-curve "
              f"columns, per D2's established relevance ruling)")

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
        metrics = compute_metrics(stats, target_norm_stats)
        io.cprint(f"\n=== FINAL {split_name} metrics (best epoch {best_epoch}) ===")
        for term, m in metrics.items():
            io.cprint(f"  {term:20s}: n={m['n']:3d}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()
