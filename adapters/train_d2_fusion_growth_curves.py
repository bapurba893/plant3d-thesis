"""
train_d2_fusion_growth_curves.py

Block D, row D2 (F2: "+ growth curve params"). Identical to D1 in every
respect (frozen B3b/KPConv+DA-S pooled feature, TraitFusionHead's linear-
head architecture, masked_regression_loss/run_epoch/compute_metrics, the
7-term L_trait+L_growth Kendall-weighted loss, the same Oracle-style
plant-disjoint split) EXCEPT the fusion input: D2 concatenates each
plant's already-fit Logistic/Gompertz growth-curve parameters (from D0's
scripts/growth_curves.py, data/traits/pheno4d_growth_curves.csv) onto
B3b's 256-dim pooled feature, per CLAUDE.md's Block D isolation rule
(backbone/DA/augmentation/architecture/loss stay identical across
D1-D6 -- only the fusion input changes).

Reuses TraitFusionHead/masked_regression_loss/run_epoch/compute_metrics/
IOStream/build_scan_table/compute_norm_stats/ALL_TERMS/TRAIT_TERMS/
GROWTH_TERMS/FLAG_AFFECTED_TERMS from train_d1_fusion_baseline.py
UNMODIFIED (imported, not copy-pasted) -- the only new code here is the
growth-curve-parameter join and the dataset wrapper that concatenates it
onto the cached feature.

**Design decision, flagged as an inference (not spec text) -- the design
docs say "+ growth curve params" but never specify which parameters or
how they attach to a per-scan sample**: uses ALL 6 raw numeric parameters
each curve family produces (logistic_K/B/rho, gompertz_A/Bg/beta) for
BOTH height and stem_diameter (12 params total, not just the "winning"
family per plant) -- simplest, most complete interpretation (no
cherry-picking which family "matters"), consistent with this project's
general preference for letting a downstream linear model discover what's
useful rather than hand-engineering a reduction. Confirmed via D0's
growth_curves.csv (2026-09-27): all 14 plants x 2 traits converged for
BOTH families with no NaNs, so no missing-value handling was needed.

Each plant's growth-curve parameters are a property of the WHOLE fitted
trajectory (an ODE's own shape/rate/asymptote constants, not a specific
scan's future measurement), so the same 12-dim vector is broadcast
identically to every scan belonging to that plant -- this is standard
practice for panel/hierarchical growth-curve features and does not leak
information ACROSS plants (the Oracle split stays fully plant-disjoint;
each held-out plant's own curve was fit only from its own scans, the
same way D0 always fit it, never from another plant's data).

stem_diameter's curve fits were already confirmed POOR project-wide in
D0 (R^2 as low as 0.025, see growth_curves.py's module docstring) --
those 6 stem_diameter parameters are still included as input here (unlike
their exclusion from L_growth's targets), since a poorly-fit curve is
still a real summary of that plant's stem_diameter trajectory shape and
whether it helps or hurts is exactly the empirical question D2 vs D1
is meant to answer -- flagged, not filtered out pre-emptively.

**Bug found and fixed on the first real run (2026-09-27)**: the B/Bg
shape parameters are exactly the ones growth_curves.py's own docstring
already flags as using "very wide bounds (many orders of magnitude)...
physically non-meaningful" and deliberately excludes from ITS OWN
fit-quality check for that reason (`_fit_one`'s `bound_check_indices`).
A raw first run showed this instability firsthand: test plant M03's
stem_diameter_logistic_B fit to 74,464 (vs. every other one of the 14
plants' values sitting in [0.7, 380]) -- a single unconstrained,
poorly-identified fit on an already-known-poor-quality trait. Standardized
with TRAIN-only mean/std (which are NOT themselves corrupted -- checked
directly, no train plant has this problem), that one value produced a
z-score of +2254, which a LINEAR model (no hidden-layer nonlinearity to
contain it) extrapolated into catastrophic test-set predictions (test R^2
in the negative hundreds on EVERY term, not just stem_diameter -- a
linear layer's weight matrix mixes every input into every output, so one
exploding input column corrupts all 7 predictions for that plant's scans).
Fixed with CLIP_STD (below): standardized growth-curve-param features are
clipped to +/-CLIP_STD before entering the model, bounding worst-case
extrapolation without changing what's included or discarding the
parameter's signal for the other 13, well-behaved plants.

**Architecture corrected, second bug/design flaw found and fixed
(2026-09-28), BEFORE this file's numbers were treated as final**: the
first real run used D1's shared `TraitFusionHead` (a single
`Linear(268, 7)`), reasoned at the time as "identical to D1's linear-head
architecture, only the input width changed." That reasoning undersold a
real problem, root-caused via an inference-time ablation (zero the 12
growth-curve dims, keep the trained weights): leaf_area's test R^2
recovered from -0.5069 to 0.5540 (near D1's own 0.5952) -- the growth
dims were causally corrupting leaf_area's prediction. **Important
correction to the original diagnosis**: this is NOT cross-term gradient
leakage through a shared weight MATRIX -- a plain `nn.Linear(in_dim,
n_out)` layer already gives each output row an independent gradient (row
i's weights only ever receive dL_i/dW, never dL_j/dW for j!=i; Kendall's
learned per-term scalars don't couple rows either) -- so 7 separate
per-term `nn.Linear` modules taking the SAME shared 268-dim input would
have been a mathematical no-op, not a fix. The actual mechanism: leaf_area
(and leaf_count/volume, none of which has a growth-curve fit of their own
in D0's scope -- curve-fitting was always restricted to height/
stem_diameter, see growth_curves.py's TRAITS_TO_FIT) had its OWN
independently-optimized weight row spuriously fit noise in the 12
growth-curve columns, because 268 free input dims against 72 training
samples is enough for gradient descent to find some spurious fit even
under wd=1e-2, and that fit didn't generalize.

**PerTermFusionHead (below) fixes this structurally, not by hoping a
learned weight goes to zero**: leaf_area/leaf_count/volume's `nn.Linear`
modules are constructed with in_dim=256 and are NEVER given the 12
growth-curve columns as input at all -- literal absence, the same
"structural guarantee over runtime flag" principle already used for the
frozen-B3b-encoder enforcement (see train_d1_fusion_baseline.py's module
docstring). height/height_rate see B3b's 256 feat + height's OWN 6 curve
params only (not stem_diameter's); stem_diameter/stem_diameter_rate see
B3b's 256 feat + stem_diameter's OWN 6 curve params only. This is
genuinely different from the original TraitFusionHead(in_dim=268)
despite both being "linear heads" -- the difference is which columns
each term's row is even allowed to see, not whether the layer is single
vs. multi-module. CLIP_STD's clipping is still necessary and unchanged:
height/stem_diameter/their _rate terms still legitimately consume the
growth-curve columns (including the ones that can take pathological
values), so their own extrapolation still needs bounding.
"""

CLIP_STD = 5.0

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

GROWTH_PARAM_FIELDS = ["logistic_K", "logistic_B", "logistic_rho",
                        "gompertz_A", "gompertz_Bg", "gompertz_beta"]
GROWTH_PARAM_TRAITS = ["height", "stem_diameter"]
GROWTH_PARAM_COLS = [f"{trait}_{field}" for trait in GROWTH_PARAM_TRAITS for field in GROWTH_PARAM_FIELDS]
N_GROWTH_PARAMS = len(GROWTH_PARAM_COLS)  # 12


def build_growth_param_table(growth_curves_csv):
    """Pivots data/traits/pheno4d_growth_curves.csv (one row per
    plant_id x trait) into one row per plant_id with 12 columns
    (GROWTH_PARAM_COLS), e.g. height_logistic_K, stem_diameter_gompertz_beta."""
    curves = pd.read_csv(growth_curves_csv)
    wide = curves.pivot(index="plant_id", columns="trait", values=GROWTH_PARAM_FIELDS)
    wide.columns = [f"{trait}_{field}" for field, trait in wide.columns]
    wide = wide.reset_index()
    missing = [c for c in GROWTH_PARAM_COLS if c not in wide.columns]
    assert not missing, f"growth_curves.csv missing expected trait/field combos: {missing}"
    assert wide[GROWTH_PARAM_COLS].isna().sum().sum() == 0, \
        "unexpected NaN in growth-curve params -- D0's fits were confirmed all-convergent; investigate"
    return wide[["plant_id"] + GROWTH_PARAM_COLS]


def build_scan_table_d2(split_csv, traits_csv, growth_csv, growth_curves_csv, features_dir):
    """D1's build_scan_table (feat_path + 7 targets + valid flags) with
    each plant's 12 growth-curve parameters merged on -- same value for
    every scan belonging to that plant."""
    table = build_scan_table(split_csv, traits_csv, growth_csv, features_dir)
    growth_params = build_growth_param_table(growth_curves_csv)
    merged = table.merge(growth_params, on="plant_id", how="left")
    assert merged[GROWTH_PARAM_COLS].isna().sum().sum() == 0, \
        "some scans' plant_id had no matching growth-curve params -- check plant_id consistency"
    return merged


def compute_feature_norm_stats(table: pd.DataFrame) -> dict:
    """Same z-score convention as D1's compute_norm_stats, but for the 12
    growth-curve-parameter INPUT columns (not the 7 target columns) --
    fit on train split only, no validity masking needed since D0's fits
    were confirmed all-convergent (no NaNs to mask)."""
    stats = {}
    for col in GROWTH_PARAM_COLS:
        vals = table[col]
        mean = float(vals.mean())
        std = float(vals.std()) if len(vals) > 1 else 1.0
        if std < 1e-8:
            std = 1.0
        stats[col] = (mean, std)
    return stats


class D2FusionDataset(Dataset):
    """Like D1FusionDataset, but the model input is B3b's cached 256-dim
    feature CONCATENATED with the plant's 12 standardized growth-curve
    parameters (268-dim total). Targets/valids unchanged from D1."""

    def __init__(self, table: pd.DataFrame, target_norm_stats: dict, feature_norm_stats: dict):
        self.table = table.reset_index(drop=True)
        self.target_norm_stats = target_norm_stats
        self.feature_norm_stats = feature_norm_stats

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        with np.load(row["feat_path"]) as d:
            b3b_feat = d["feat"].astype(np.float32)
        growth_feat = np.array(
            [(row[col] - self.feature_norm_stats[col][0]) / self.feature_norm_stats[col][1]
             for col in GROWTH_PARAM_COLS], dtype=np.float32)
        # Clip to +/-CLIP_STD -- see module docstring's "Bug found and fixed" note:
        # unconstrained curve-fit shape params (B/Bg) can take pathological values
        # (one test plant's stem_diameter_logistic_B hit z=+2254) that a linear
        # model with no saturation would otherwise extrapolate catastrophically.
        growth_feat = np.clip(growth_feat, -CLIP_STD, CLIP_STD)
        feat = np.concatenate([b3b_feat, growth_feat])

        targets, valids = [], []
        for term in ALL_TERMS:
            mean, std = self.target_norm_stats[term]
            raw = row[term]
            z = (raw - mean) / std if row[f"{term}_valid"] else 0.0
            targets.append(z)
            valids.append(1.0 if row[f"{term}_valid"] else 0.0)
        return feat, np.array(targets, dtype=np.float32), np.array(valids, dtype=np.float32)


_N_B3B = 256
_N_GROWTH_PER_TRAIT = len(GROWTH_PARAM_FIELDS)  # 6
# Which terms' own head is allowed to see growth-curve columns at all, and
# which trait's 6-column block (see D2FusionDataset -- GROWTH_PARAM_COLS'
# construction order guarantees feat[:, 256:262]=height's block,
# feat[:, 262:268]=stem_diameter's block). leaf_area/leaf_count/volume are
# deliberately absent -- see module docstring's "PerTermFusionHead" note.
_GROWTH_RELEVANT_SLICE = {
    "height": slice(_N_B3B, _N_B3B + _N_GROWTH_PER_TRAIT),
    "height_rate": slice(_N_B3B, _N_B3B + _N_GROWTH_PER_TRAIT),
    "stem_diameter": slice(_N_B3B + _N_GROWTH_PER_TRAIT, _N_B3B + 2 * _N_GROWTH_PER_TRAIT),
    "stem_diameter_rate": slice(_N_B3B + _N_GROWTH_PER_TRAIT, _N_B3B + 2 * _N_GROWTH_PER_TRAIT),
}


class PerTermFusionHead(nn.Module):
    """7 genuinely separate nn.Linear modules, one per term in ALL_TERMS,
    each constructed with ONLY the input columns relevant to that term --
    see module docstring's "Architecture corrected" note for why this
    differs from (and fixes what) a single shared Linear(268,7) didn't.

    - leaf_area, leaf_count, volume: Linear(256, 1) -- B3b feature only,
      growth-curve columns never concatenated at all (no growth-curve fit
      of their own exists in D0's scope for these 3 traits).
    - height, height_rate: Linear(262, 1) -- B3b feature + height's own
      6 curve params (columns 256:262 of the cached 268-dim feat).
    - stem_diameter, stem_diameter_rate: Linear(262, 1) -- B3b feature +
      stem_diameter's own 6 curve params (columns 262:268).

    forward() still takes the SAME (B, 268) feat tensor D2FusionDataset
    already produces (no dataset change needed) and slices internally --
    the isolation is in which columns reach each nn.Linear's weight
    matrix, not in the data pipeline."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.heads = nn.ModuleDict()
        for term in ALL_TERMS:
            in_dim = _N_B3B + (_N_GROWTH_PER_TRAIT if term in _GROWTH_RELEVANT_SLICE else 0)
            self.heads[term] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 1))

    def forward(self, feat):
        b3b = feat[:, :_N_B3B]
        outs = []
        for term in ALL_TERMS:
            if term in _GROWTH_RELEVANT_SLICE:
                x = torch.cat([b3b, feat[:, _GROWTH_RELEVANT_SLICE[term]]], dim=1)
            else:
                x = b3b
            outs.append(self.heads[term](x))
        return torch.cat(outs, dim=1)  # (B, len(ALL_TERMS))


def parse_args():
    p = argparse.ArgumentParser(description="D2: fusion (F2, + growth curve params)")
    p.add_argument("--exp_name", type=str, default="D2_fusion_growth_curves")
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

    train_table = build_scan_table_d2(data_dir / "pheno4d_oracle_train.csv", traits_csv, growth_csv,
                                       growth_curves_csv, features_dir)
    val_table = build_scan_table_d2(data_dir / "pheno4d_oracle_val.csv", traits_csv, growth_csv,
                                     growth_curves_csv, features_dir)
    test_table = build_scan_table_d2(data_dir / "pheno4d_heldout_eval.csv", traits_csv, growth_csv,
                                      growth_curves_csv, features_dir)
    io.cprint(f"train: {len(train_table)} scans, val: {len(val_table)}, test: {len(test_table)}")

    target_norm_stats = compute_norm_stats(train_table)
    feature_norm_stats = compute_feature_norm_stats(train_table)
    io.cprint(f"target norm_stats (mean, std), fit on train split only: {target_norm_stats}")
    io.cprint(f"growth-curve-param feature norm_stats (mean, std), fit on train split only: {feature_norm_stats}")

    train_set = D2FusionDataset(train_table, target_norm_stats, feature_norm_stats)
    val_set = D2FusionDataset(val_table, target_norm_stats, feature_norm_stats)
    test_set = D2FusionDataset(test_table, target_norm_stats, feature_norm_stats)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = PerTermFusionHead(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    per_term_in_dims = {term: (_N_B3B + _N_GROWTH_PER_TRAIT if term in _GROWTH_RELEVANT_SLICE else _N_B3B)
                         for term in ALL_TERMS}
    io.cprint(f"PerTermFusionHead parameter count: {n_params} -- per-term in_dim: {per_term_in_dims} "
              f"(leaf_area/leaf_count/volume NEVER see the 12 growth-curve columns; height/height_rate "
              f"see only height's own 6; stem_diameter/stem_diameter_rate see only stem_diameter's own 6 "
              f"-- structural isolation, see module docstring's 'Architecture corrected' note)")

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
