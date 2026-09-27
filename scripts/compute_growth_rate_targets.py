"""
compute_growth_rate_targets.py

D1 (Block D fusion baseline) prerequisite: computes per-scan growth-RATE
targets for L_growth ("Growth Trend Prediction", L_growth = MSE per the
design docs) -- a ground truth the docs never define, resolved directly
with the user (see step_notes/D1_Fusion_Baseline.md for the full
reasoning):

- **height**: the ALREADY-FITTED growth curve's own closed-form
  derivative at each scan's elapsed_days, using whichever of Logistic/
  Gompertz has the higher R^2 for that specific plant (height's curve
  fits were confirmed good in D0, R^2 0.95-0.99 across all 14 plants --
  trustworthy enough to differentiate). Per-plant choice, not a
  hardcoded family: Logistic wins 10/14 plants, Gompertz 4/14.
- **stem_diameter**: real finite-difference rates via np.gradient
  (NOT the fitted curve -- its fits were confirmed poor, R^2 as low as
  0.025, so its derivative would be worse than finite-difference noise).
  The 6 `stem_leaf_boundary_low_confidence` scans are dropped from each
  plant's series BEFORE computing gradients (not just their own target
  excluded -- a bad point between two good ones would corrupt both
  neighbors' finite differences too), so those scans get no
  stem_diameter_rate target at all (NaN), consistent with already
  lacking a trustworthy stem_diameter value.

Derivatives use the ODE forms directly (not a numerical/autodiff
derivative of the closed-form expression), since the closed forms were
constructed to satisfy these ODEs exactly:
- Logistic: dy/dt = rho*y*(1 - y/K)  (the ODE this project's Logistic
  row is defined by; the closed form's derivative equals this exactly,
  by construction -- see scripts/growth_curves.py's logistic_curve
  docstring).
- Gompertz: dy/dt = y*beta*Bg*exp(-beta*t)  -- derived directly from
  y(t) = A*exp(-Bg*exp(-beta*t)) (chain rule), and cross-checked
  algebraically to equal y*(alpha_hat - beta*ln(y)) using this project's
  own alpha_hat = beta*ln(A) definition (scripts/growth_curves.py::
  fit_gompertz) -- i.e. consistent with the documented Gompertz ODE
  ε(t) = dy/dt - (alpha - beta*y_log) with y_log=ln(y), just computed in
  the numerically simpler A/Bg/beta form (avoids a log near y->0).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent


def logistic_rate(t, K, B, rho):
    """dy/dt = rho*y*(1-y/K), evaluated via the closed-form y(t) first."""
    y = K / (1.0 + B * np.exp(-rho * t))
    return rho * y * (1.0 - y / K)


def gompertz_rate(t, A, Bg, beta):
    """dy/dt = y*beta*Bg*exp(-beta*t), from y(t)=A*exp(-Bg*exp(-beta*t))."""
    y = A * np.exp(-Bg * np.exp(-beta * t))
    return y * beta * Bg * np.exp(-beta * t)


def compute_height_rates(plant_group: pd.DataFrame, curve_row: pd.Series) -> np.ndarray:
    t = plant_group["elapsed_days"].to_numpy(dtype=np.float64)
    logi_r2 = curve_row.get("logistic_R2", -np.inf)
    gomp_r2 = curve_row.get("gompertz_R2", -np.inf)
    if pd.isna(logi_r2):
        logi_r2 = -np.inf
    if pd.isna(gomp_r2):
        gomp_r2 = -np.inf
    if logi_r2 >= gomp_r2:
        return logistic_rate(t, curve_row["logistic_K"], curve_row["logistic_B"], curve_row["logistic_rho"])
    return gompertz_rate(t, curve_row["gompertz_A"], curve_row["gompertz_Bg"], curve_row["gompertz_beta"])


def compute_stem_diameter_rates(plant_group: pd.DataFrame) -> pd.Series:
    """Returns a Series indexed like plant_group, NaN for flagged scans
    and for plants with <2 usable points (np.gradient needs >=2)."""
    out = pd.Series(np.nan, index=plant_group.index, dtype=np.float64)
    usable = plant_group[~plant_group["stem_leaf_boundary_low_confidence"]]
    if len(usable) < 2:
        return out
    t = usable["elapsed_days"].to_numpy(dtype=np.float64)
    y = usable["stem_diameter"].to_numpy(dtype=np.float64)
    rates = np.gradient(y, t)
    out.loc[usable.index] = rates
    return out


def main():
    ap = argparse.ArgumentParser(description="D1: compute per-scan growth-rate (L_growth) targets")
    ap.add_argument("--timeseries_csv", type=str,
                     default=str(_REPO_ROOT / "data" / "traits" / "pheno4d_traits_timeseries_long.csv"))
    ap.add_argument("--growth_curves_csv", type=str,
                     default=str(_REPO_ROOT / "data" / "traits" / "pheno4d_growth_curves.csv"))
    ap.add_argument("--out", type=str,
                     default=str(_REPO_ROOT / "data" / "traits" / "pheno4d_growth_rate_targets.csv"))
    args = ap.parse_args()

    ts = pd.read_csv(args.timeseries_csv)
    curves = pd.read_csv(args.growth_curves_csv)
    height_curves = curves[curves["trait"] == "height"].set_index("plant_id")

    rows = []
    for plant_id, group in ts.groupby("plant_id"):
        group = group.sort_values("elapsed_days")
        if plant_id not in height_curves.index:
            print(f"[warn] no height growth-curve fit for {plant_id}, skipping height_rate")
            height_rates = pd.Series(np.nan, index=group.index)
        else:
            curve_row = height_curves.loc[plant_id]
            height_rates = pd.Series(compute_height_rates(group, curve_row), index=group.index)

        stem_rates = compute_stem_diameter_rates(group)

        for idx in group.index:
            rows.append({
                "plant_id": plant_id,
                "species": group.loc[idx, "species"],
                "scan_date": group.loc[idx, "scan_date"],
                "elapsed_days": group.loc[idx, "elapsed_days"],
                "height_rate": height_rates.loc[idx],
                "stem_diameter_rate": stem_rates.loc[idx],
            })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(f"height_rate NaN: {out_df['height_rate'].isna().sum()}, "
          f"stem_diameter_rate NaN: {out_df['stem_diameter_rate'].isna().sum()} "
          f"(expected 6, matching the flagged low-confidence scans)")
    print(out_df.groupby("species")[["height_rate", "stem_diameter_rate"]].describe())


if __name__ == "__main__":
    main()
