"""
d6_synthetic_recovery_test.py -- standalone diagnostic, NOT part of the training pipeline
(same category as scripts/d5_synthetic_recovery_test.py / scripts/inspect_crops3d_sf.py).

Gompertz counterpart of scripts/d5_synthetic_recovery_test.py. Two questions, per explicit
user instruction (do not assume D5's findings transfer, verify them) -- see
step_notes/D5_D6_Physics_Loss.md for the full writeup and result this script produced:

1. Does the finite-difference L_phys correctly recover alpha/beta from a KNOWN Gompertz
   trajectory? (The autodiff-vs-total-trajectory bug that motivated the finite-difference
   redesign was generic -- about elapsed_days vs. prev_height, not specific to the Logistic
   residual formula -- so this should transfer, but is verified here rather than assumed.)
2. Does alpha need its own boosted learning rate the way K did for Logistic? Analytically this
   looks LESS likely: in eps = dydt - (alpha - beta*y_mid), d(eps)/d(alpha) = -1 (constant,
   full-strength regardless of beta's current value) -- unlike Logistic's d(eps)/dK =
   rho*y^2/K^2, which vanishes when rho is still small/wrong. beta's own gradient,
   d(eps)/d(beta) = y_mid (the log-space trait value, e.g. ln(150)~5 for height -- not tied to
   alpha's correctness either). So neither parameter should show Logistic's K-style multiplicative
   suppression -- tested directly rather than assumed.

RESULT (2026-09-29): both questions confirmed. Finite-difference recovers both parameters well
(alpha ~2% relative error, beta ~1%) even from a bad init (alpha0=0.2 vs true 0.6, beta0=0.03 vs
true 0.1). Unlike K, alpha does NOT need a boosted learning rate: 1x/5x/10x lr_beta all land
within 1.8-2.1% alpha error (noise-level differences, no trend) -- D6 uses a SINGLE shared
learning rate for alpha and beta, not D5's split lr_ode/lr_K.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adapters"))
from physics_loss import effective_rate_param, inverse_softplus_init, gompertz_residual  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

ALPHA_TRUE = 0.60
BETA_TRUE = 0.10
V0 = 5.0


def analytic_gompertz_log(t, alpha, beta, v0):
    y0 = np.log(v0)
    y_ss = alpha / beta
    return y_ss + (y0 - y_ss) * np.exp(-beta * t)


N_PLANTS = 12
SCAN_DAYS = [0, 4, 8, 13, 18, 24, 30]
NOISE_STD_LOG = 0.08  # additive noise in LOG space (V's own noise would scale with V's magnitude)

rows = []
for pid in range(N_PLANTS):
    beta_p = BETA_TRUE * np.random.uniform(0.85, 1.15)
    v0_p = V0 * np.random.uniform(0.8, 1.2)
    for d in SCAN_DAYS:
        y_true_log = analytic_gompertz_log(d, ALPHA_TRUE, beta_p, v0_p)
        y_noisy_log = y_true_log + np.random.normal(0, NOISE_STD_LOG)
        v_noisy = np.exp(y_noisy_log)
        rows.append((pid, float(d), float(v_noisy)))

days = torch.tensor([r[1] for r in rows], dtype=torch.float32)
targets = torch.tensor([r[2] for r in rows], dtype=torch.float32)  # RAW-space targets, matching
# how the real fusion model predicts raw trait values (log() is applied inside L_phys, not here)
plant_ids = [r[0] for r in rows]
day_mean, day_std = days.mean().item(), days.std().item()
days_z = (days - day_mean) / day_std

by_plant = {}
for i, pid in enumerate(plant_ids):
    by_plant.setdefault(pid, []).append(i)
for pid in by_plant:
    by_plant[pid] = sorted(by_plant[pid], key=lambda i: days[i].item())


class TinyPredictor(nn.Module):
    """Predicts RAW-space y (not log) -- matches the real model, which predicts raw trait
    values; physics_loss.py's compute_l_phys_finite_diff itself applies log() for Gompertz."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, t_z):
        return self.net(t_z.unsqueeze(-1)).squeeze(-1)


def run_trial(lr_alpha, lr_beta=2e-2, lambda_phys=0.1, epochs=3000, verbose_every=0):
    model = TinyPredictor()
    raw_beta = nn.Parameter(torch.tensor(inverse_softplus_init(0.03)))  # start far from true 0.10
    alpha = nn.Parameter(torch.tensor(0.20))  # start far from true 0.60

    opt = optim.Adam([
        {"params": model.parameters(), "lr": 1e-2},
        {"params": [raw_beta], "lr": lr_beta, "weight_decay": 0.0},
        {"params": [alpha], "lr": lr_alpha, "weight_decay": 0.0},
    ])

    for epoch in range(epochs):
        opt.zero_grad()
        preds_raw = model(days_z)  # raw-space prediction, must stay positive-ish for log()
        l_fit = ((preds_raw - targets) ** 2).mean()

        beta = effective_rate_param(raw_beta)
        preds_log = torch.log(preds_raw.clamp(min=1e-3))
        residuals = []
        for pid, idxs_sorted in by_plant.items():
            for k in range(len(idxs_sorted) - 1):
                i0, i1 = idxs_sorted[k], idxs_sorted[k + 1]
                dt = days[i1] - days[i0]
                dydt = (preds_log[i1] - preds_log[i0]) / dt
                y_mid = 0.5 * (preds_log[i0] + preds_log[i1])
                eps = gompertz_residual(y_mid, dydt, alpha, beta)
                residuals.append(eps.unsqueeze(0))
        l_phys = (torch.cat(residuals) ** 2).mean()

        loss = l_fit + lambda_phys * l_phys
        loss.backward()
        opt.step()

        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:5d}: l_fit={l_fit.item():8.3f} l_phys={l_phys.item():10.4f} "
                  f"alpha={alpha.item():.4f} beta={beta.item():.4f}")

    beta_final = effective_rate_param(raw_beta).item()
    alpha_final = alpha.item()
    alpha_err = abs(alpha_final - ALPHA_TRUE) / ALPHA_TRUE * 100
    beta_err = abs(beta_final - BETA_TRUE) / BETA_TRUE * 100
    return alpha_final, beta_final, alpha_err, beta_err


if __name__ == "__main__":
    print(f"Ground truth: alpha={ALPHA_TRUE}, beta={BETA_TRUE}. Init: alpha0=0.20, beta0=0.03\n")
    for lr_alpha in [0.02, 0.1, 0.2]:
        multiple = lr_alpha / 0.02
        print(f"=== lr_alpha = {lr_alpha} ({multiple:.0f}x lr_beta=0.02) ===")
        alpha_f, beta_f, alpha_err, beta_err = run_trial(lr_alpha, verbose_every=1000)
        print(f"  FINAL: alpha={alpha_f:.4f} (true {ALPHA_TRUE}, {alpha_err:.1f}% rel err), "
              f"beta={beta_f:.4f} (true {BETA_TRUE}, {beta_err:.1f}% rel err)\n")
