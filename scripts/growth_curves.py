"""
growth_curves.py

D0 (trait-extraction pipeline, Block D prerequisite): fits Logistic and
Gompertz growth curves per plant to `data/traits/pheno4d_traits_
timeseries_long.csv` (from `scripts/assemble_trait_timeseries.py`).

**Scope, confirmed with the user before this pipeline was built**:
curve-fitting applies ONLY to height and stem_diameter -- the two traits
CLAUDE.md's `L_mono` monotonicity rule already singles out. leaf_area,
leaf_count, and volume are extracted and time-series-assembled but never
curve-fit here: Logistic/Gompertz both assume monotonic saturating
growth, which those three traits can legitimately violate via senescence
(a plant can lose leaf area/count, or even canopy volume, without
anything being wrong). This exclusion is enforced structurally below (a
fixed `TRAITS_TO_FIT` list, not a runtime check) -- see the module-level
comment at its definition for why a future edit should not "helpfully"
extend it.

**stem_diameter-specific exclusion**: 6 scans (T01_320, T02_322, T03_324,
T04_324, T07_322, T07_325) have a confirmed, ground-truth-verified
segmentation failure that inflates stem_diameter by 13-24x (see
step_notes/D0_Trait_Extraction_Pipeline.md's "known segmentation failure
mode" section) -- these are excluded from stem_diameter fitting via the
`stem_leaf_boundary_low_confidence` flag column
(`trait_extraction.py::flag_stem_leaf_boundary_confidence`). height is
NOT excluded for these scans -- confirmed via the same ground-truth
comparison that height is unaffected by this failure mode (−1.8% to
+0.0% difference on the 3 checked scans), so dropping perfectly good
height data points would only reduce curve-fit quality for no reason.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Curve-fitting scope: height and stem_diameter ONLY. leaf_area/leaf_count/
# volume are deliberately never added to this list (see module docstring) --
# a future edit that "helpfully" extends fitting to those traits would be
# fitting a monotonic saturating-growth model to traits that can legitimately
# decrease, producing a physically meaningless fit.
TRAITS_TO_FIT = ["height", "stem_diameter"]

MIN_POINTS_TO_FIT = 4
R2_POOR_THRESHOLD = 0.8
BOUND_PROXIMITY_FRACTION = 0.01  # flag if a fitted param sits within 1% of its bound


def logistic_curve(t, K, B, rho):
    """y(t) = K / (1 + B*exp(-rho*t)), the closed-form solution of
    dy/dt = rho*y*(1 - y/K) with B = (K-y0)/y0 folding in y(0)=y0.
    Fit (K,B,rho) directly rather than fixing y0 from the (possibly
    noisy) first data point."""
    return K / (1.0 + B * np.exp(-rho * t))


def gompertz_curve(t, A, B_g, beta):
    """y(t) = A*exp(-B_g*exp(-beta*t)), the closed-form solution
    (exponentiated back to raw space) of the LOG-space linear ODE
    d(ln y)/dt = alpha - beta*ln(y): with C = alpha/beta (asymptotic
    log-value), ln(y)(t) = C + (ln(y0)-C)*exp(-beta*t), so
    A = exp(alpha/beta) = exp(C) and B_g = C - ln(y0). Reproduces the
    textbook 3-parameter Gompertz form exactly -- a self-consistency
    check on the derivation, not an independently guessed formula."""
    return A * np.exp(-B_g * np.exp(-beta * t))


def _fit_one(curve_fn, t, y, p0, bounds, param_names, bound_check_indices=(0,)):
    """bound_check_indices: which popt indices to run the near-bound
    check on -- defaults to just index 0 (K for logistic, A for
    Gompertz), the asymptote parameter, which has a tight, physically
    meaningful bound (0.9x-5x of y_max) where hitting it genuinely
    signals a poorly-constrained fit. B/rho/beta deliberately use very
    wide bounds (many orders of magnitude, since their natural scale
    varies a lot with y0's relative size) -- checking "fraction of span"
    proximity on those is meaningless (almost any realistic value sits
    near 0% of a [1e-6, 1e6]-scale span), so they're excluded by
    default rather than producing a spurious "poor" flag on every fit."""
    try:
        popt, _ = curve_fit(curve_fn, t, y, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        return None
    y_pred = curve_fn(t, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    near_bound = False
    lo_list, hi_list = bounds
    for i in bound_check_indices:
        val, lo, hi = popt[i], lo_list[i], hi_list[i]
        span = hi - lo
        if span > 0 and (val - lo < BOUND_PROXIMITY_FRACTION * span or
                          hi - val < BOUND_PROXIMITY_FRACTION * span):
            near_bound = True

    quality = "poor" if (np.isnan(r2) or r2 < R2_POOR_THRESHOLD or near_bound) else "ok"
    result = dict(zip(param_names, popt))
    result["R2"] = float(r2)
    result["fit_quality"] = quality
    return result


def fit_logistic(t, y):
    y_max = float(y.max())
    y0 = float(y[0]) if y[0] > 0 else 1e-3
    p0 = [1.2 * y_max, max((y_max - y0) / max(y0, 1e-6), 1e-3), 0.1]
    bounds = ([0.9 * y_max, 1e-6, 1e-4], [5 * y_max, 1e6, 5])
    result = _fit_one(logistic_curve, t, y, p0, bounds, ["logistic_K", "logistic_B", "logistic_rho"])
    if result is not None:
        K, B = result["logistic_K"], result["logistic_B"]
        result["logistic_y0_hat"] = K / (1.0 + B)
    return result


def fit_gompertz(t, y):
    y_max = float(y.max())
    y0 = float(y[0]) if y[0] > 0 else 1e-3
    A0 = 1.2 * y_max
    Bg0 = max(np.log(max(A0 / max(y0, 1e-6), 1.001)), 1e-3)
    p0 = [A0, Bg0, 0.1]
    bounds = ([0.9 * y_max, 1e-6, 1e-4], [5 * y_max, 50, 5])
    result = _fit_one(gompertz_curve, t, y, p0, bounds, ["gompertz_A", "gompertz_Bg", "gompertz_beta"])
    if result is not None:
        A, Bg, beta = result["gompertz_A"], result["gompertz_Bg"], result["gompertz_beta"]
        result["gompertz_alpha_hat"] = beta * np.log(A)
        result["gompertz_y0hat_log"] = np.log(A) - Bg
    return result


def fit_all(timeseries_csv, out_csv):
    df = pd.read_csv(timeseries_csv)

    rows = []
    for (plant_id, species), group in df.groupby(["plant_id", "species"]):
        group = group.sort_values("elapsed_days")
        for trait in TRAITS_TO_FIT:
            sub = group
            excluded_n = 0
            if trait == "stem_diameter":
                mask = ~sub["stem_leaf_boundary_low_confidence"]
                excluded_n = int((~mask).sum())
                sub = sub[mask]
            t = sub["elapsed_days"].to_numpy(dtype=np.float64)
            y = sub[trait].to_numpy(dtype=np.float64)
            valid = ~np.isnan(y)
            t, y = t[valid], y[valid]

            row = {"plant_id": plant_id, "species": species, "trait": trait,
                   "n_points_fit": len(t), "n_excluded_low_confidence": excluded_n}
            if len(t) < MIN_POINTS_TO_FIT:
                row["fit_quality"] = "insufficient_points"
                rows.append(row)
                continue

            logi = fit_logistic(t, y)
            gomp = fit_gompertz(t, y)
            if logi is None or gomp is None:
                row["fit_quality"] = "non_convergent"
                rows.append(row)
                continue

            # fit_logistic/fit_gompertz already return keys prefixed with
            # "logistic_"/"gompertz_" (from param_names in _fit_one) except
            # the two generic ones _fit_one adds ("R2", "fit_quality"),
            # which get their prefix added here.
            for k, v in logi.items():
                row[f"logistic_{k}" if k in ("R2", "fit_quality") else k] = v
            for k, v in gomp.items():
                row[f"gompertz_{k}" if k in ("R2", "fit_quality") else k] = v
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(out_df.groupby(["trait", "species"])[["logistic_fit_quality", "gompertz_fit_quality"]]
          .apply(lambda g: g.apply(pd.Series.value_counts).fillna(0)))
    return out_df


def main():
    ap = argparse.ArgumentParser(description="D0: fit Logistic/Gompertz growth curves (height, stem_diameter only)")
    ap.add_argument("--timeseries_csv", type=str,
                     default=str(_REPO_ROOT / "data" / "traits" / "pheno4d_traits_timeseries_long.csv"))
    ap.add_argument("--out", type=str,
                     default=str(_REPO_ROOT / "data" / "traits" / "pheno4d_growth_curves.csv"))
    args = ap.parse_args()
    fit_all(args.timeseries_csv, args.out)


if __name__ == "__main__":
    main()
