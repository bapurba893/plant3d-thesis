"""
d5_synthetic_recovery_test.py -- standalone diagnostic, NOT part of the
training pipeline (same category as scripts/inspect_crops3d_sf.py). Reruns
D5's rho/K recovery test with K given its own, higher learning rate
(separate from rho's lr_ode), per explicit user instruction, to check
whether that fixes K's stuck recovery (see step_notes/D5_D6_Physics_Loss.md
for the full writeup and the 2026-09-29 result this script produced).

Generates known-ground-truth Logistic trajectories (rho_true=0.30,
K_true=100, per-plant variation in K/y0 to mimic real inter-plant
variability), fits a small MLP to noisy per-scan observations while
jointly optimizing rho (softplus-bounded, physics_loss.py's
effective_rate_param) and K via the SAME finite-difference residual
formula physics_loss.py::compute_l_phys_finite_diff/logistic_residual use
(consecutive real scan pairs, midpoint y, dt-normalized derivative).

rho and K both start deliberately far from their true values (rho0=0.05
vs true 0.30, K0=40 vs true 100) to reproduce the same "stuck" regime
seen in D5's earlier synthetic test. Compares K's final recovery across
lr_K in {0.02 (=lr_ode baseline), 0.1 (5x), 0.2 (10x)}.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adapters"))
from physics_loss import effective_rate_param, inverse_softplus_init, logistic_residual  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

RHO_TRUE = 0.30
K_TRUE = 100.0
Y0 = 5.0


def analytic_logistic(t, rho, K, y0):
    return K / (1.0 + ((K - y0) / y0) * np.exp(-rho * t))


N_PLANTS = 12
SCAN_DAYS = [0, 4, 8, 13, 18, 24, 30]
NOISE_STD = 1.5

rows = []
for pid in range(N_PLANTS):
    K_p = K_TRUE * np.random.uniform(0.85, 1.15)
    y0_p = Y0 * np.random.uniform(0.8, 1.2)
    for d in SCAN_DAYS:
        y_true = analytic_logistic(d, RHO_TRUE, K_p, y0_p)
        y_noisy = y_true + np.random.normal(0, NOISE_STD)
        rows.append((pid, float(d), float(y_noisy)))

days = torch.tensor([r[1] for r in rows], dtype=torch.float32)
targets = torch.tensor([r[2] for r in rows], dtype=torch.float32)
plant_ids = [r[0] for r in rows]
day_mean, day_std = days.mean().item(), days.std().item()
days_z = (days - day_mean) / day_std

by_plant = {}
for i, pid in enumerate(plant_ids):
    by_plant.setdefault(pid, []).append(i)
for pid in by_plant:
    by_plant[pid] = sorted(by_plant[pid], key=lambda i: days[i].item())


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, t_z):
        return self.net(t_z.unsqueeze(-1)).squeeze(-1)


def run_trial(lr_K, lr_rho=2e-2, lambda_phys=0.1, epochs=3000, verbose_every=0):
    model = TinyPredictor()
    raw_rho = nn.Parameter(torch.tensor(inverse_softplus_init(0.05)))
    K = nn.Parameter(torch.tensor(40.0))

    opt = optim.Adam([
        {"params": model.parameters(), "lr": 1e-2},
        {"params": [raw_rho], "lr": lr_rho, "weight_decay": 0.0},
        {"params": [K], "lr": lr_K, "weight_decay": 0.0},
    ])

    for epoch in range(epochs):
        opt.zero_grad()
        preds = model(days_z)
        l_fit = ((preds - targets) ** 2).mean()

        rate = effective_rate_param(raw_rho)
        residuals = []
        for pid, idxs_sorted in by_plant.items():
            for k in range(len(idxs_sorted) - 1):
                i0, i1 = idxs_sorted[k], idxs_sorted[k + 1]
                dt = days[i1] - days[i0]
                dydt = (preds[i1] - preds[i0]) / dt
                y_mid = 0.5 * (preds[i0] + preds[i1])
                eps = logistic_residual(y_mid, dydt, rate, K)
                residuals.append(eps.unsqueeze(0))
        l_phys = (torch.cat(residuals) ** 2).mean()

        loss = l_fit + lambda_phys * l_phys
        loss.backward()
        opt.step()

        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:5d}: l_fit={l_fit.item():8.3f} l_phys={l_phys.item():10.4f} "
                  f"rho={rate.item():.4f} K={K.item():7.2f}")

    rho_final = effective_rate_param(raw_rho).item()
    K_final = K.item()
    rho_err = abs(rho_final - RHO_TRUE) / RHO_TRUE * 100
    K_err = abs(K_final - K_TRUE) / K_TRUE * 100
    return rho_final, K_final, rho_err, K_err


if __name__ == "__main__":
    print(f"Ground truth: rho={RHO_TRUE}, K={K_TRUE}. Init: rho0=0.05, K0=40.0\n")
    for lr_K in [0.02, 0.1, 0.2]:
        multiple = lr_K / 0.02
        print(f"=== lr_K = {lr_K} ({multiple:.0f}x lr_rho={0.02}) ===")
        rho_f, K_f, rho_err, K_err = run_trial(lr_K, verbose_every=1000)
        print(f"  FINAL: rho={rho_f:.4f} (true {RHO_TRUE}, {rho_err:.1f}% rel err), "
              f"K={K_f:.2f} (true {K_TRUE}, {K_err:.1f}% rel err)\n")
