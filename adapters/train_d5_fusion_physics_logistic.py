"""
train_d5_fusion_physics_logistic.py

Block D, row D5 (D4 + Logistic ODE physics loss + L_mono). Reuses D4's
exact fusion input, architecture (PerTermFusionHeadD4), and dataset
(build_scan_table_d4/D4FusionDataset) UNMODIFIED -- per the user's own
framing ("D1 through D4 add one component at a time, D5/D6 add the
physics-informed growth constraint"), D5 changes ONLY the loss function,
not the fusion input.

Design decisions verified against the primary source docs before
implementation -- see adapters/physics_loss.py's module docstring for
the full sourced findings and step_notes/D5_D6_Physics_Loss.md for the
complete writeup, INCLUDING a documented departure from the docs: dy/dt
was first implemented via autodiff (the docs' literal instruction),
verified correct in isolation to machine precision, then found via a
synthetic recovery test to constrain the WRONG quantity (an isolated
partial derivative disagreeing with the true total trajectory derivative
by ~11-40x RMS). L_phys now uses a finite-difference TOTAL trajectory
derivative instead (adapters/physics_loss.py::compute_l_phys_finite_diff),
computed once per epoch alongside L_mono from a single shared forward
pass (run_physics_mono_step) -- trainable global per-trait rho/K
(softplus-bounded) and L_phys/L_mono's fixed-weight (never
Kendall-weighted) regime are unchanged from the original design.

**lambda_phys/lambda_mono are fixed hyperparameters with no spec value**
-- chosen from an empirical magnitude check on D4's own trained
checkpoint (see step_notes/D5_D6_Physics_Loss.md) rather than guessed,
so the physics/monotonicity terms are comparably scaled to the
Kendall-combined L_trait+L_growth total at the start of D5 training, not
silently dominating or negligible.
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

ODE_FAMILY = "logistic"


def load_d0_population_init(growth_curves_csv):
    """rho,K initial values -- a reasonable starting point (population
    mean of D0's already-fit per-plant curves), NOT a fixed target; both
    remain nn.Parameter and get updated during D5 training. See
    physics_loss.py's module docstring point 3."""
    gc = pd.read_csv(growth_curves_csv)
    init = {}
    for term in PHYSICS_TERMS:
        sub = gc[gc.trait == term]
        init[term] = (float(sub.logistic_rho.mean()), float(sub.logistic_K.mean()))
    return init


def parse_args():
    p = argparse.ArgumentParser(description="D5: fusion + Logistic ODE physics loss + L_mono")
    p.add_argument("--exp_name", type=str, default="D5_fusion_physics_logistic")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lambda_corr", type=float, default=0.5)
    p.add_argument("--lambda_phys", type=float, default=1e-4,
                    help="fixed weight on L_phys -- checked empirically on D4's own trained "
                         "checkpoint (raw L_phys~182, contributes ~0.018 at this weight, "
                         "comparable to D4's own Kendall-combined loss scale ~0.1-1), not tuned")
    p.add_argument("--lambda_mono", type=float, default=0.1,
                    help="fixed weight on L_mono -- checked empirically on D4's own trained "
                         "checkpoint (raw L_mono~1.19; the naive default of 1.0 would have "
                         "contributed ~1.19, likely DOMINATING D4's own ~0.1-1 loss scale; "
                         "0.1 contributes ~0.12, comparably scaled instead), not tuned")
    p.add_argument("--lr_ode", type=float, default=2e-2,
                    help="separate, higher LR for the trainable ODE parameters -- also given "
                         "their own weight_decay=0.0 parameter group (see main()), since applying "
                         "the network's own weight_decay to physical constants was a real bug found "
                         "during synthetic verification (it pulled K toward 0 identically for both "
                         "traits, see step_notes/D5_D6_Physics_Loss.md)")
    p.add_argument("--init_from_d4", type=str,
                    default=str(_REPO_ROOT / "results" / "D4_fusion_previous_stage" / "model.pt"),
                    help="warm-start PerTermFusionHeadD4 weights from D4's own trained checkpoint "
                         "(same architecture) rather than training from scratch")
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
              f"(same D4-filtered tables, unmodified -- comparable to D4 on the SAME 54-scan test "
              f"subset, not D1-D3's 63)")
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
    io.cprint(f"D0 population-average init for (rho, K): {init_params}")
    io.cprint("rho is softplus-bounded (effective_rate_param) to prevent the degenerate rho->0 "
              "collapse found via synthetic verification -- see physics_loss.py's module docstring "
              "and step_notes/D5_D6_Physics_Loss.md")
    ode_params = {}
    trainable_ode_params = []
    for term in PHYSICS_TERMS:
        rho0, K0 = init_params[term]
        raw_rho = torch.nn.Parameter(torch.tensor(inverse_softplus_init(rho0), dtype=torch.float32, device=device))
        K = torch.nn.Parameter(torch.tensor(K0, dtype=torch.float32, device=device))
        ode_params[term] = (raw_rho, K)
        trainable_ode_params += [raw_rho, K]

    kendall = KendallUncertaintyWeighting({term: "regression" for term in ALL_TERMS}).to(device)
    opt = optim.Adam([
        {"params": list(model.parameters()) + list(kendall.parameters()), "weight_decay": args.wd},
        {"params": trainable_ode_params, "weight_decay": 0.0, "lr": args.lr_ode},
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
            raw_rho_h, K_h = ode_params["height"]
            raw_rho_s, K_s = ode_params["stem_diameter"]
            rho_h, rho_s = effective_rate_param(raw_rho_h).item(), effective_rate_param(raw_rho_s).item()
            io.cprint(f"Epoch {epoch}: train_loss={train_loss:.4f} (phys={train_phys:.4f}, mono={train_mono:.4f}), "
                      f"val_loss={val_loss:.4f} (phys={val_phys:.4f}, mono={val_mono:.4f}) val_total={val_total:.4f}, "
                      f"rho_h={rho_h:.4f} K_h={K_h.item():.2f} rho_s={rho_s:.4f} K_s={K_s.item():.2f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ode_params = {term: (effective_rate_param(raw_rate).item(), other.item())
                                for term, (raw_rate, other) in ode_params.items()}
            torch.save(model.state_dict(), os.path.join(exp_path, "model.pt"))
            torch.save(best_ode_params, os.path.join(exp_path, "ode_params.pt"))

    io.cprint(f"Best model at epoch {best_epoch}, val total loss {best_val_loss:.4f}")
    io.cprint(f"Best (rho, K) per trait: {best_ode_params}")
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
