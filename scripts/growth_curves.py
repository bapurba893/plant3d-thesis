"""
growth_curves.py

D0 (trait-extraction pipeline, Block D prerequisite): fits Logistic and
Gompertz growth curves per plant to `data/traits/pheno4d_traits_
timeseries_long.csv` (from `scripts/assemble_trait_timeseries.py`).

Two modes: the default (`fit_all`) fits one curve per plant on the whole
trajectory; `--causal` (`fit_all_causal`) fits one curve per scan on that
plant's strictly earlier scans only. Block D's growth-curve inputs
(D2-D6) use the causal file since 2026-10-27; the whole-trajectory file
gave each test scan curve parameters fitted on its own and later trait
values (see step_notes/FW_5_Results.md).

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


def fit_all_causal(timeseries_csv, out_csv):
    """Causal (expanding-window) version of fit_all, used as the Block D
    growth-curve INPUT (D2-D6) since 2026-10-27. For every scan of every
    plant and each trait in TRAITS_TO_FIT, fits Logistic and Gompertz
    using ONLY that plant's scans strictly earlier than the scan
    (elapsed_days < t), so a scan's curve parameters never contain its
    own or any later trait value. fit_all (whole-trajectory fits,
    data/traits/pheno4d_growth_curves.csv) is kept unchanged: it is the
    D0 record, is read by compute_growth_rate_targets.py for the
    height_rate targets, and (training plants only) supplies D2's
    fallback values and D5/D6's starting values.

    Same exclusions and thresholds as fit_all: stem_diameter windows
    drop low-confidence scans, NaN values are dropped, and a window with
    fewer than MIN_POINTS_TO_FIT points is written as
    "insufficient_points" with no parameters (D2 then uses the fallback,
    see train_d2_fusion_growth_curves.py::build_growth_feature_table).
    Output: one row per (plant_id, scan_date, trait)."""
    df = pd.read_csv(timeseries_csv)

    rows = []
    for (plant_id, species), group in df.groupby(["plant_id", "species"]):
        group = group.sort_values("elapsed_days")
        for _, scan in group.iterrows():
            t_now = scan["elapsed_days"]
            earlier = group[group["elapsed_days"] < t_now]
            for trait in TRAITS_TO_FIT:
                sub = earlier
                if trait == "stem_diameter":
                    sub = sub[~sub["stem_leaf_boundary_low_confidence"]]
                t = sub["elapsed_days"].to_numpy(dtype=np.float64)
                y = sub[trait].to_numpy(dtype=np.float64)
                valid = ~np.isnan(y)
                t, y = t[valid], y[valid]

                row = {"plant_id": plant_id, "species": species, "scan_date": scan["scan_date"],
                       "elapsed_days": t_now, "trait": trait, "n_points_fit": len(t)}
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

                row["fit_quality"] = "fitted"
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
    print(out_df.groupby(["trait", "species"])["fit_quality"].value_counts().unstack(fill_value=0))
    return out_df


def main():
    ap = argparse.ArgumentParser(description="D0: fit Logistic/Gompertz growth curves (height, stem_diameter only)")
    ap.add_argument("--timeseries_csv", type=str,
                     default=str(_REPO_ROOT / "data" / "traits" / "pheno4d_traits_timeseries_long.csv"))
    ap.add_argument("--out", type=str, default=None,
                     help="default: data/traits/pheno4d_growth_curves.csv, or "
                          "data/traits/pheno4d_growth_curves_causal.csv with --causal")
    ap.add_argument("--causal", action="store_true",
                     help="fit each scan's curves on that plant's strictly earlier scans only "
                          "(fit_all_causal), instead of one whole-trajectory fit per plant")
    args = ap.parse_args()
    if args.causal:
        out = args.out or str(_REPO_ROOT / "data" / "traits" / "pheno4d_growth_curves_causal.csv")
        fit_all_causal(args.timeseries_csv, out)
    else:
        out = args.out or str(_REPO_ROOT / "data" / "traits" / "pheno4d_growth_curves.csv")
        fit_all(args.timeseries_csv, out)


if __name__ == "__main__":
    main()
