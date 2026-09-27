"""
train_d4_fusion_previous_stage.py

Block D, row D4 (F4: "+ previous growth stage" -- the "full model" per
CLAUDE.md, last of D1-D4's progressive fusion-input additions before
D5/D6 add the physics-informed loss). Cumulative: D4's input = D3's input
(B3b's 256-dim feature + 12 growth-curve params + elapsed_days) + 5 new
previous-growth-stage features, one per raw trait.

**Design decision, flagged as an inference (not spec text)**: "previous
growth stage" = that SAME plant's chronologically-immediately-preceding
scan's 5 raw trait values (height, stem_diameter, leaf_area, leaf_count,
volume) -- the literal, most natural reading of "stage" as a measured
STATE, not a rate (growth rates are already D1's L_growth targets, not
duplicated here as an input) and not a curve summary (that's D2's job).

**Handling of missing previous stage, checked against real data before
deciding (2026-09-28)**: a plant's first scan has no previous scan at
all (14/223 scans project-wide); separately, a previous scan can itself
have a NaN trait value (8/223, D0's DBSCAN min-point threshold) or be
`stem_leaf_boundary_low_confidence` flagged (5/223, the confirmed
Tomato segmentation failure -- see step_notes/D0_Trait_Extraction_
Pipeline.md). Rather than fabricate a previous-stage value for a scan
that never had a trustworthy one (the same category of mistake D2's
first run made with an unconstrained curve-fit parameter -- see
step_notes/D2_Fusion_Growth_Curves.md), or invent a new per-term
input-masking mechanism that would itself need separate verification
(the reasoning that ruled out option (b) during D2's architecture fix),
rows are EXCLUDED from D4's tables entirely when their previous scan
doesn't exist or isn't fully trustworthy across all 5 traits -- matching
this project's established preference (D0's low-confidence scans were
EXCLUDED from stem_diameter curve-fitting, not imputed over). Real,
checked impact: train 72->59 (18% dropped), val 18->15 (17%),
test 63->54 (14%) -- a real, reported cost of this row, not hidden.
Checked the remaining 5 raw traits' value ranges (train+val vs. test)
before deciding whether clipping was needed like D2's growth-curve
params: full overlap, 0 outliers for all 5 traits -- no clipping
safeguard needed here (real measured quantities, not an unconstrained
curve-fit parameter).

**Per-term input relevance, applying (not mechanically copying) D2/D3's
discipline at finer granularity**: D3 gave `elapsed_days` to ALL 7 terms
since every trait/rate is genuinely time-dependent. D4's previous-stage
features are different: the single most direct, well-justified
autoregressive relationship is a trait's OWN lag-1 value (current_height
is most directly informed by previous_height, not previous_leaf_count) --
a standard panel-data assumption, and the minimal, literal reading absent
a specified cross-trait design in the docs. So each term gets access to
ONLY its own trait's previous value: height/height_rate get prev_height;
stem_diameter/stem_diameter_rate get prev_stem_diameter; leaf_area gets
prev_leaf_area; leaf_count gets prev_leaf_count; volume gets prev_volume.
Every term's in_dim grows by exactly +1 relative to D3 (not a blanket
+5), consistent with D2's original height/stem_diameter-only restriction
of growth-curve params -- cross-trait relevance (e.g. does leaf_area
have a principled claim on previous height, via allometry) is a real but
weaker, unspecified relationship not added here without being asked.

Reuses D3's build_scan_table_d3/compute_temporal_norm_stats and D2's
compute_feature_norm_stats/GROWTH_PARAM_COLS/CLIP_STD/
_GROWTH_RELEVANT_SLICE and D1's IOStream/ALL_TERMS/TRAIT_TERMS/
build_scan_table/compute_norm_stats/masked_regression_loss/run_epoch/
compute_metrics, all unmodified.
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
    compute_feature_norm_stats, GROWTH_PARAM_COLS, CLIP_STD,
    _GROWTH_RELEVANT_SLICE, _N_B3B, _N_GROWTH_PER_TRAIT,
)
from train_d3_fusion_temporal import (  # noqa: E402
    build_scan_table_d3, compute_temporal_norm_stats, _TEMPORAL_COL_IDX,
)

_N_GROWTH_TOTAL = len(GROWTH_PARAM_COLS)  # 12
PREV_STAGE_COLS = [f"prev_{t}" for t in TRAIT_TERMS]  # prev_height, ..., prev_volume
# Column layout of the 274-dim feat vector this file produces:
# [0:256]=B3b, [256:262]=height growth params, [262:268]=stem growth params,
# [268]=elapsed_days, [269:274]=prev_height..prev_volume (TRAIT_TERMS order).
_PREV_STAGE_START = _TEMPORAL_COL_IDX + 1  # 269
_TERM_TO_PREV_IDX = {term: _PREV_STAGE_START + TRAIT_TERMS.index(term) for term in TRAIT_TERMS}
_TERM_TO_PREV_IDX["height_rate"] = _TERM_TO_PREV_IDX["height"]
_TERM_TO_PREV_IDX["stem_diameter_rate"] = _TERM_TO_PREV_IDX["stem_diameter"]


def build_scan_table_d4(split_csv, traits_csv, growth_csv, growth_curves_csv,
                         timeseries_csv, features_dir):
    """D3's scan table (feat_path + 7 targets/valid + 12 growth-curve
    params + elapsed_days) with 5 previous-growth-stage columns attached,
    EXCLUDING rows whose previous scan doesn't exist or isn't fully
    trustworthy (see module docstring)."""
    table = build_scan_table_d3(split_csv, traits_csv, growth_csv, growth_curves_csv, features_dir)
    ts = pd.read_csv(timeseries_csv)

    rows = []
    n_excluded_first, n_excluded_bad = 0, 0
    for _, row in table.iterrows():
        plant_series = ts[ts["plant_id"] == row["plant_id"]].sort_values("elapsed_days").reset_index(drop=True)
        idx = plant_series.index[plant_series["scan_date"] == row["scan_date"]]
        assert len(idx) == 1, f"expected exactly 1 match for {row['plant_id']}/{row['scan_date']}"
        idx = idx[0]
        if idx == 0:
            n_excluded_first += 1
            continue
        prev = plant_series.iloc[idx - 1]
        if prev[TRAIT_TERMS].isna().any() or bool(prev["stem_leaf_boundary_low_confidence"]):
            n_excluded_bad += 1
            continue
        entry = row.to_dict()
        for t in TRAIT_TERMS:
            entry[f"prev_{t}"] = float(prev[t])
        rows.append(entry)

    print(f"[build_scan_table_d4] {split_csv}: {len(table)} scans -> {len(rows)} usable "
          f"(excluded {n_excluded_first} first-scans, {n_excluded_bad} bad-previous-scan)")
    return pd.DataFrame(rows)


def compute_prev_stage_norm_stats(table: pd.DataFrame) -> dict:
    """z-score for the 5 prev_* columns, train-only stats. No validity
    masking needed -- build_scan_table_d4 already excluded every row
    lacking a trustworthy previous stage, so every remaining row has a
    clean value for all 5 columns."""
    stats = {}
    for col in PREV_STAGE_COLS:
        vals = table[col]
        mean = float(vals.mean())
        std = float(vals.std()) if len(vals) > 1 else 1.0
        if std < 1e-8:
            std = 1.0
        stats[col] = (mean, std)
    return stats


class D4FusionDataset(Dataset):
    """D3's 269-dim feat with 5 more standardized columns appended
    (274-dim total): prev_height, prev_stem_diameter, prev_leaf_area,
    prev_leaf_count, prev_volume, in that fixed order."""

    def __init__(self, table: pd.DataFrame, target_norm_stats: dict, feature_norm_stats: dict,
                 temporal_norm_stats: dict, prev_stage_norm_stats: dict):
        self.table = table.reset_index(drop=True)
        self.target_norm_stats = target_norm_stats
        self.feature_norm_stats = feature_norm_stats
        self.temporal_norm_stats = temporal_norm_stats
        self.prev_stage_norm_stats = prev_stage_norm_stats

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
        prev_feat = np.array(
            [(row[col] - self.prev_stage_norm_stats[col][0]) / self.prev_stage_norm_stats[col][1]
             for col in PREV_STAGE_COLS], dtype=np.float32)
        feat = np.concatenate([b3b_feat, growth_feat, temporal_feat, prev_feat])

        targets, valids = [], []
        for term in ALL_TERMS:
            mean, std = self.target_norm_stats[term]
            raw = row[term]
            z = (raw - mean) / std if row[f"{term}_valid"] else 0.0
            targets.append(z)
            valids.append(1.0 if row[f"{term}_valid"] else 0.0)
        return feat, np.array(targets, dtype=np.float32), np.array(valids, dtype=np.float32)


class PerTermFusionHeadD4(nn.Module):
    """D3's PerTermFusionHeadD3 extended with each term's OWN previous-
    trait value (see module docstring's per-term relevance note -- own
    trait lag-1 only, not all 5 previous-stage columns). Per-term
    in_dim: leaf_area/leaf_count/volume = 257(D3)+1=258;
    height/height_rate/stem_diameter/stem_diameter_rate = 263(D3)+1=264."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.heads = nn.ModuleDict()
        for term in ALL_TERMS:
            in_dim = _N_B3B + (_N_GROWTH_PER_TRAIT if term in _GROWTH_RELEVANT_SLICE else 0) + 1 + 1
            self.heads[term] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 1))

    def forward(self, feat):
        b3b = feat[:, :_N_B3B]
        temporal = feat[:, _TEMPORAL_COL_IDX:_TEMPORAL_COL_IDX + 1]
        outs = []
        for term in ALL_TERMS:
            prev_idx = _TERM_TO_PREV_IDX[term]
            prev = feat[:, prev_idx:prev_idx + 1]
            if term in _GROWTH_RELEVANT_SLICE:
                x = torch.cat([b3b, feat[:, _GROWTH_RELEVANT_SLICE[term]], temporal, prev], dim=1)
            else:
                x = torch.cat([b3b, temporal, prev], dim=1)
            outs.append(self.heads[term](x))
        return torch.cat(outs, dim=1)


def parse_args():
    p = argparse.ArgumentParser(description="D4: fusion (F4, + previous growth stage, full model)")
    p.add_argument("--exp_name", type=str, default="D4_fusion_previous_stage")
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
    timeseries_csv = data_dir / "traits" / "pheno4d_traits_timeseries_long.csv"
    features_dir = data_dir / "features" / "Pheno4D"

    train_table = build_scan_table_d4(data_dir / "pheno4d_oracle_train.csv", traits_csv, growth_csv,
                                       growth_curves_csv, timeseries_csv, features_dir)
    val_table = build_scan_table_d4(data_dir / "pheno4d_oracle_val.csv", traits_csv, growth_csv,
                                     growth_curves_csv, timeseries_csv, features_dir)
    test_table = build_scan_table_d4(data_dir / "pheno4d_heldout_eval.csv", traits_csv, growth_csv,
                                      growth_curves_csv, timeseries_csv, features_dir)
    io.cprint(f"train: {len(train_table)} scans, val: {len(val_table)}, test: {len(test_table)} "
              f"(reduced from D1-D3's 72/18/63 -- see module docstring's exclusion note)")

    target_norm_stats = compute_norm_stats(train_table)
    feature_norm_stats = compute_feature_norm_stats(train_table)
    temporal_norm_stats = compute_temporal_norm_stats(train_table)
    prev_stage_norm_stats = compute_prev_stage_norm_stats(train_table)
    io.cprint(f"target norm_stats: {target_norm_stats}")
    io.cprint(f"growth-curve-param feature norm_stats: {feature_norm_stats}")
    io.cprint(f"temporal norm_stats: {temporal_norm_stats}")
    io.cprint(f"prev-stage norm_stats: {prev_stage_norm_stats}")

    train_set = D4FusionDataset(train_table, target_norm_stats, feature_norm_stats, temporal_norm_stats, prev_stage_norm_stats)
    val_set = D4FusionDataset(val_table, target_norm_stats, feature_norm_stats, temporal_norm_stats, prev_stage_norm_stats)
    test_set = D4FusionDataset(test_table, target_norm_stats, feature_norm_stats, temporal_norm_stats, prev_stage_norm_stats)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = PerTermFusionHeadD4(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    per_term_in_dims = {term: (_N_B3B + _N_GROWTH_PER_TRAIT + 2 if term in _GROWTH_RELEVANT_SLICE else _N_B3B + 2)
                         for term in ALL_TERMS}
    io.cprint(f"PerTermFusionHeadD4 parameter count: {n_params} -- per-term in_dim: {per_term_in_dims} "
              f"(every term gets elapsed_days + its OWN previous-trait value only)")

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
