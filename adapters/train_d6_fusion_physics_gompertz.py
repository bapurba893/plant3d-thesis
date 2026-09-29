"""
train_d6_fusion_physics_gompertz.py

Block D, row D6 (D4 + Gompertz ODE physics loss + L_mono). Reuses D4's exact fusion input,
architecture (PerTermFusionHeadD4), and dataset (build_scan_table_d4/D4FusionDataset)
UNMODIFIED, and physics_loss.py's already ode_family-parameterized machinery UNCHANGED -- D6
is the Gompertz counterpart of D5 (adapters/train_d5_fusion_physics_logistic.py), same
architecture/isolation discipline: only the physics loss's ODE family changes, not the fusion
input.

Design decisions verified for D6 SPECIFICALLY, not assumed to carry over from D5, per explicit
user instruction -- full writeup in step_notes/D5_D6_Physics_Loss.md:

1. **Finite-difference L_phys transfers correctly to the Gompertz residual** -- verified via
   scripts/d6_synthetic_recovery_test.py (a KNOWN Gompertz trajectory, alpha/beta recovered to
   ~2%/~1% relative error). The autodiff-vs-total-trajectory bug D5 found and fixed was generic
   (about elapsed_days vs. prev_height, not specific to the Logistic residual formula), so this
   was expected to transfer -- confirmed, not assumed.
2. **alpha does NOT need D5's boosted-learning-rate treatment.** D5 found K's gradient
   (d(eps)/dK = rho*y^2/K^2) vanishes while rho is still small/wrong, causing K to get stuck
   unless given 5x rho's learning rate. Gompertz's residual is structurally different:
   d(eps)/d(alpha) = -1 (constant, full-strength regardless of beta's current value) --
   analytically no reason to expect the same suppression. Confirmed empirically: the same
   synthetic test at 1x/5x/10x lr_beta all recovered alpha to within 1.8-2.1% (noise-level
   differences, no trend). D6 therefore uses a SINGLE shared learning rate for alpha and beta
   (one optimizer param group, `lr_ode`), NOT D5's split lr_ode/lr_K -- checked, not copied.
3. **alpha's population-average init comes from D0's own already-derived
   `gompertz_alpha_hat = beta * ln(A)`** (scripts/growth_curves.py::fit_gompertz), not a fresh
   guess -- D0 fits the raw-space 3-parameter Gompertz curve (A, B_g, beta), and
   `gompertz_alpha_hat`/`gompertz_beta` are the exact (alpha, beta) pair CLAUDE.md's log-space
   ODE `dy/dt = alpha - beta*y` needs, already present as columns in
   data/traits/pheno4d_growth_curves.csv. beta is softplus-bounded (effective_rate_param, same
   floor-collapse-prevention rationale as D5's rho) via the same inverse_softplus_init
   convention; alpha is unconstrained, matching D5's K.
4. Everything else (once-per-epoch shared forward pass for L_phys+L_mono, fixed/never-Kendall
   weighting, PHYSICS_TERMS={height, stem_diameter}, warm-start from D4's checkpoint, same
   train/val/test split D4 used) is IDENTICAL to D5 -- see physics_loss.py's module docstring
   for the shared machinery both rows reuse unchanged.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses import KendallUncertaintyWeighting  # noqa: E402
from train_d1_fusion_baseline import (  # noqa: E402
    IOStream, ALL_TERMS, compute_norm_stats, compute_metrics, run_epoch,
)
from train_d2_fusion_growth_curves import compute_feature_norm_stats  # noqa: E402
from train_d3_fusion_temporal import compute_temporal_norm_stats  # noqa: E402
from train_d4_fusion_previous_stage import (  # noqa: E402
    PerTermFusionHeadD4, build_scan_table_d4, compute_prev_stage_norm_stats, D4FusionDataset,
)
from physics_loss import (  # noqa: E402
    PHYSICS_TERMS, run_physics_mono_step,
    effective_rate_param, inverse_softplus_init,
)

ODE_FAMILY = "gompertz"


def load_d0_population_init(growth_curves_csv):
    """(alpha, beta) initial values -- D0's own already-derived gompertz_alpha_hat/gompertz_beta
    columns (population mean), NOT a fixed target; both remain nn.Parameter and get updated
    during D6 training. See this file's module docstring point 3."""
    gc = pd.read_csv(growth_curves_csv)
    init = {}
    for term in PHYSICS_TERMS:
        sub = gc[gc.trait == term]
        init[term] = (float(sub.gompertz_alpha_hat.mean()), float(sub.gompertz_beta.mean()))
    return init


def parse_args():
    p = argparse.ArgumentParser(description="D6: fusion + Gompertz ODE physics loss + L_mono")
    p.add_argument("--exp_name", type=str, default="D6_fusion_physics_gompertz")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lambda_corr", type=float, default=0.5)
    p.add_argument("--lambda_phys", type=float, default=1e-4,
                    help="fixed weight on L_phys -- same value D5 used (checked empirically on "
                         "D4's own trained checkpoint, see train_d5_fusion_physics_logistic.py), "
                         "not re-tuned for Gompertz specifically")
    p.add_argument("--lambda_mono", type=float, default=0.1,
                    help="fixed weight on L_mono -- same value D5 used, unaffected by ODE family "
                         "choice (L_mono doesn't depend on ode_family at all)")
    p.add_argument("--lr_ode", type=float, default=2e-2,
                    help="SINGLE shared LR for BOTH trainable alpha and beta -- unlike D5's split "
                         "lr_ode/lr_K, a dedicated synthetic recovery test "
                         "(scripts/d6_synthetic_recovery_test.py) found alpha does NOT get stuck "
                         "the way K did: d(eps)/d(alpha)=-1 is constant/full-strength regardless "
                         "of beta's current value, unlike d(eps)/dK which scales with rho. 1x/5x/"
                         "10x lr_beta all recovered alpha to within 1.8-2.1% relative error in "
                         "that test (noise-level differences, no trend) -- checked, not assumed "
                         "from D5. Given its own weight_decay=0.0 parameter group (see main()), "
                         "same rationale as D5 (physical constants shouldn't decay toward 0).")
    p.add_argument("--init_from_d4", type=str,
                    default=str(_REPO_ROOT / "results" / "D4_fusion_previous_stage" / "model.pt"),
                    help="warm-start PerTermFusionHeadD4 weights from D4's own trained checkpoint "
                         "(same architecture) rather than training from scratch -- same as D5, "
                         "NOT D5's own checkpoint, so D6 is directly comparable to D4/D5 rather "
                         "than chained off D5")
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
              f"(same D4-filtered tables, unmodified -- comparable to D4/D5 on the SAME 54-scan "
              f"test subset)")
    full_timeseries_table = pd.read_csv(timeseries_csv)

    target_norm_stats = compute_norm_stats(train_table)
    feature_norm_stats = compute_feature_norm_stats(train_table)
    temporal_norm_stats = compute_temporal_norm_stats(train_table)
    prev_stage_norm_stats = compute_prev_stage_norm_stats(train_table)

    train_set = D4FusionDataset(train_table, target_norm_stats, feature_norm_stats, temporal_norm_stats, prev_stage_norm_stats)
    val_set = D4FusionDataset(val_table, target_norm_stats, feature_norm_stats, temporal_norm_stats, prev_stage_norm_stats)
    test_set = D4FusionDataset(test_table, target_norm_stats, feature_norm_stats, temporal_norm_stats, prev_stage_norm_stats)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = PerTermFusionHeadD4(dropout=args.dropout).to(device)
    if args.init_from_d4 and Path(args.init_from_d4).exists():
        model.load_state_dict(torch.load(args.init_from_d4, map_location=device))
        io.cprint(f"warm-started model weights from {args.init_from_d4}")
    else:
        io.cprint("training model weights from scratch (no D4 checkpoint found)")

    init_params = load_d0_population_init(growth_curves_csv)
    io.cprint(f"D0 population-average init for (alpha, beta): {init_params}")
    io.cprint("beta is softplus-bounded (effective_rate_param) to prevent the degenerate beta->0 "
              "collapse (same rationale as D5's rho) -- see physics_loss.py's module docstring "
              "and step_notes/D5_D6_Physics_Loss.md")
    ode_params = {}
    ode_trainable_params = []
    for term in PHYSICS_TERMS:
        alpha0, beta0 = init_params[term]
        raw_beta = torch.nn.Parameter(torch.tensor(inverse_softplus_init(beta0), dtype=torch.float32, device=device))
        alpha = torch.nn.Parameter(torch.tensor(alpha0, dtype=torch.float32, device=device))
        ode_params[term] = (raw_beta, alpha)
        ode_trainable_params += [raw_beta, alpha]

    kendall = KendallUncertaintyWeighting({term: "regression" for term in ALL_TERMS}).to(device)
    opt = optim.Adam([
        {"params": list(model.parameters()) + list(kendall.parameters()), "weight_decay": args.wd},
        {"params": ode_trainable_params, "weight_decay": 0.0, "lr": args.lr_ode},
    ], lr=args.lr)
    scheduler = CosineAnnealingLR(opt, args.epochs)

    best_val_loss, best_epoch = float("inf"), 0
    best_model_state, best_ode_params = None, None
    for epoch in range(args.epochs):
        _, train_loss = run_epoch(model, kendall, train_loader, device, optimizer=opt, lambda_corr=args.lambda_corr)
        train_phys, train_mono = run_physics_mono_step(
            model, train_table, D4FusionDataset, target_norm_stats, feature_norm_stats,
            temporal_norm_stats, prev_stage_norm_stats, full_timeseries_table, ODE_FAMILY, ode_params,
            device, args.lambda_phys, args.lambda_mono, optimizer=opt)
        scheduler.step()

        val_stats, val_loss = run_epoch(model, kendall, val_loader, device, optimizer=None, lambda_corr=args.lambda_corr)
        val_phys, val_mono = run_physics_mono_step(
            model, val_table, D4FusionDataset, target_norm_stats, feature_norm_stats,
            temporal_norm_stats, prev_stage_norm_stats, full_timeseries_table, ODE_FAMILY, ode_params,
            device, args.lambda_phys, args.lambda_mono, optimizer=None)
        val_total = val_loss + args.lambda_phys * val_phys + args.lambda_mono * val_mono

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            raw_beta_h, alpha_h = ode_params["height"]
            raw_beta_s, alpha_s = ode_params["stem_diameter"]
            beta_h, beta_s = effective_rate_param(raw_beta_h).item(), effective_rate_param(raw_beta_s).item()
            io.cprint(f"Epoch {epoch}: train_loss={train_loss:.4f} (phys={train_phys:.4f}, mono={train_mono:.4f}), "
                      f"val_loss={val_loss:.4f} (phys={val_phys:.4f}, mono={val_mono:.4f}) val_total={val_total:.4f}, "
                      f"alpha_h={alpha_h.item():.4f} beta_h={beta_h:.4f} alpha_s={alpha_s.item():.4f} beta_s={beta_s:.4f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ode_params = {term: (alpha.item(), effective_rate_param(raw_beta).item())
                                for term, (raw_beta, alpha) in ode_params.items()}
            torch.save(model.state_dict(), os.path.join(exp_path, "model.pt"))
            torch.save(best_ode_params, os.path.join(exp_path, "ode_params.pt"))

    io.cprint(f"Best model at epoch {best_epoch}, val total loss {best_val_loss:.4f}")
    io.cprint(f"Best (alpha, beta) per trait: {best_ode_params}")
    model.load_state_dict(best_model_state)

    for split_name, loader, table in [("train", train_loader, train_table), ("val", val_loader, val_table),
                                       ("test", test_loader, test_table)]:
        stats, _ = run_epoch(model, kendall, loader, device, optimizer=None, lambda_corr=args.lambda_corr)
        phys_final, mono_final = run_physics_mono_step(
            model, table, D4FusionDataset, target_norm_stats, feature_norm_stats,
            temporal_norm_stats, prev_stage_norm_stats, full_timeseries_table, ODE_FAMILY, ode_params,
            device, args.lambda_phys, args.lambda_mono, optimizer=None)
        metrics = compute_metrics(stats, target_norm_stats)
        io.cprint(f"\n=== FINAL {split_name} metrics (best epoch {best_epoch}) === L_phys={phys_final:.4f}, L_mono={mono_final:.4f}")
        for term, m in metrics.items():
            io.cprint(f"  {term:20s}: n={m['n']:3d}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()
