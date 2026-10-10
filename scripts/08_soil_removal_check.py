"""
08_soil_removal_check.py

Block PP, Stage 1 / the Pheno4D "soil removal" definition in
docs/PREPROCESSING_COMPARISON_PLAN.md section 6. As of the 2026-10 Stage 1
part-C revision, this applies a LABEL-FREE horizontal-slab soil detector to
EVERY Pheno4D scan (annotated and unannotated alike) -- per the plan's
corrected section 6, soil/pot labels must never be used on target (Pheno4D)
scans during actual preprocessing, since that would leak target labels into
a pipeline step. Labels are used ONLY here, in this validation script, to
SCORE the geometric cut (soil-removed% / plant-points-lost%) -- never to
find it.

Why this replaced the earlier tilted-plane RANSAC-on-lowest-15%-of-points
approach: that approach failed specifically on late-growing-season scans,
where drooping leaves/stem occupy most of the lowest 15% of POINTS (by
count), so RANSAC fit a plane through plant material instead of soil (see
superseded analysis in step_notes/PP_0_Prerequisite_Checks.md, Part B).

The new method, label-free throughout
--------------------------------------
Up axis for Pheno4D is CONFIRMED +z for both species (step_notes/
PP_0_Prerequisite_Checks.md, check 2), and Pheno4D's soil is CONFIRMED a
near-horizontal sheet (same check) -- so rather than fitting an arbitrarily
tilted plane, this looks for the HEIGHT z0 of the densest thin horizontal
slab that ALSO has a wide lateral footprint:

  1. Restrict the search to the bottom `--slab-search-frac` of the scan's
     PHYSICAL z-RANGE (not bottom-%-of-points -- that distinction is the
     fix: soil is physically the lowest surface in the pot, below the plant
     it supports, which is a geometric fact about the scene, not a label;
     restricting by physical range, rather than by point-count quantile,
     is what stops a dense drooping-leaf cluster from dominating the
     candidate pool the way it did in the old lowest-15%-of-points version).
  2. Slide a thin (`--slab-window-mm`) horizontal window through that
     restricted range in `--slab-bin-width-mm` steps. For each window,
     score = (point count in the window) x (lateral footprint = number of
     occupied dx,y grid cells of size `--slab-cell-size-mm`, converted to
     mm^2). Soil wins this score because it is uniquely BOTH dense (a solid
     surface) AND wide (spans close to the whole pot diameter); a drooping
     leaf cluster can be locally dense but is narrower, and a sparse
     low-lying structure is wide but not dense.
  3. z0 = the top edge of the best-scoring window. Cut: remove points with
     z <= z0 + margin.

Validated (see step_notes/PP_0_Prerequisite_Checks.md Part C) against the
known worst-failing scans from the superseded RANSAC approach, and against
a 30-scan random sample: matches the true (label-derived) soil location in
every case except one scan (`T02_0325_a.txt`) whose OWN label-0 is itself
anomalous (spans 77.6% of that scan's full z-range vs. a 6.8% median across
all 126 annotated scans) -- a genuine single-file label-quality issue, not a
detector failure; see the step note for the full investigation.

What this script does
----------------------
1. Label-free slab detection (above) on every scan.
2. On ANNOTATED scans (both species, used ONLY to score, never to detect):
   for a grid of margins, soil-removed% and plant-points-lost%. Reports
   per-scan and overall, and states plainly whether the plan's starting
   target (>95% soil removed, <1% plant lost) is reached.
3. Geometric pot check (unchanged from the prior version): does label id 0
   hide a separate pot-wall ring, and would a cut leave a rim behind.
4. The SAME label-free detector + cut is then applied to a sample of
   UNANNOTATED scans and reports how many points it removes -- this is now
   a real preview of the actual preprocessing step (not just a sanity
   check), since the method never touched labels for either annotated or
   unannotated scans.

This script only reads raw Pheno4D data (treated as read-only) and writes
its own CSV outputs; it does not touch data/preprocessed/ or any other
pipeline script.

Usage:
    python scripts/08_soil_removal_check.py \
        --data-root /home/apurba/plant3d-data/pheno4d \
        --out-dir results/pp_checks
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from data_io import load_pheno4d_xyz  # noqa: E402

UP_AXIS = 2  # +z, confirmed for Pheno4D in step_notes/PP_0_Prerequisite_Checks.md check 2


def find_soil_slab(pts: np.ndarray, bin_width: float = 1.0, window_mm: float = 3.0,
                    cell_size: float = 2.0, search_frac: float = 0.3,
                    max_points: int = 150000, seed: int = 0):
    """
    Label-free horizontal-slab soil detector. See module docstring for the
    full method. Returns (z_lo, z_hi, density, footprint_area_mm2) for the
    best-scoring window, where z_hi is the height to cut at (before adding
    a margin) -- or None if the scan has too few points to search.

    `pts` is subsampled to at most `max_points` BEFORE scoring (mirrors the
    same pattern used for the superseded RANSAC fit): the density/footprint
    pattern that identifies soil survives a uniform subsample, and this is
    what makes scoring ~500 candidate windows tractable on scans with
    millions of points. The final cut is applied to the FULL point set by
    the caller -- this only affects which z0 is found, not what gets
    removed.
    """
    n = len(pts)
    if n < 50:
        return None
    rng = np.random.RandomState(seed)
    sub = pts[rng.choice(n, size=max_points, replace=False)] if n > max_points else pts

    z_full = sub[:, UP_AXIS]
    zmin_full, zmax_full = float(z_full.min()), float(z_full.max())
    z_search_hi = zmin_full + search_frac * (zmax_full - zmin_full)
    other_axes = [a for a in range(3) if a != UP_AXIS]
    in_search = z_full <= z_search_hi
    sub = sub[in_search]
    if len(sub) < 20:
        # Degenerate scan (e.g. almost no points in the bottom search_frac);
        # fall back to searching the whole range rather than failing outright.
        sub = pts[rng.choice(n, size=max_points, replace=False)] if n > max_points else pts
        z_search_hi = zmax_full

    z = sub[:, UP_AXIS]
    zmin, zmax = zmin_full, z_search_hi
    window_bins = max(1, int(round(window_mm / bin_width)))
    nbins = max(1, int(np.ceil((zmax - zmin) / bin_width)) + 1)
    iz = np.clip(((z - zmin) / bin_width).astype(np.int64), 0, nbins - 1)

    x, y = sub[:, other_axes[0]], sub[:, other_axes[1]]
    ix = np.floor((x - x.min()) / cell_size).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_size).astype(np.int64)
    iy_span = int(iy.max()) + 1 if len(iy) else 1
    cell_id = ix * iy_span + iy

    counts = np.bincount(iz, minlength=nbins)
    cell_mod = int(cell_id.max()) + 1 if len(cell_id) else 1
    # Dedupe (z-bin, cell) pairs globally in one pass -- this is what makes
    # per-window footprint (distinct occupied cells in the window) cheap to
    # compute for every candidate window without a fresh unique() call over
    # raw points each time.
    combo = iz * cell_mod + cell_id
    combo_u = np.unique(combo)
    iz_u = combo_u // cell_mod
    cell_u = combo_u % cell_mod
    order = np.argsort(iz_u, kind="stable")
    iz_u_sorted, cell_u_sorted = iz_u[order], cell_u[order]
    bin_starts = np.searchsorted(iz_u_sorted, np.arange(nbins))
    bin_ends = np.searchsorted(iz_u_sorted, np.arange(nbins) + 1)

    best_score = -1.0
    best = None
    for k in range(0, max(1, nbins - window_bins + 1)):
        lo, hi = k, min(k + window_bins, nbins)
        density = int(counts[lo:hi].sum())
        s, e = bin_starts[lo], bin_ends[hi - 1]
        footprint_cells = len(np.unique(cell_u_sorted[s:e])) if e > s else 0
        footprint_area = footprint_cells * cell_size * cell_size
        score = density * footprint_area
        if score > best_score:
            best_score = score
            best = (zmin + lo * bin_width, zmin + hi * bin_width, density, footprint_area)
    return best


def evaluate_margins(pts, labels, z0, margins):
    """For each margin (mm), returns the fraction of label-0 points removed
    (soil-removed%) and the fraction of label>=1 points removed
    (plant-points-lost%). 'Removed' = z <= z0 + margin."""
    z = pts[:, UP_AXIS]
    is_soil = labels == 0
    is_plant = labels >= 1
    n_soil, n_plant = int(is_soil.sum()), int(is_plant.sum())
    rows = []
    for m in margins:
        removed = z <= (z0 + m)
        soil_removed = float((removed & is_soil).sum()) / max(n_soil, 1)
        plant_lost = float((removed & is_plant).sum()) / max(n_plant, 1)
        rows.append(dict(margin=m, soil_removed_frac=soil_removed, plant_lost_frac=plant_lost,
                          n_soil=n_soil, n_plant=n_plant))
    return rows


def pot_rim_check(pts, labels, up_axis=UP_AXIS):
    """Unchanged from the prior version: checks whether label-0 (soil)
    contains a ring/wall structure at larger radius than its own bulk (a pot
    rim folded into the soil label), and whether label-0 reaches the scan's
    full horizontal radius."""
    is_soil = labels == 0
    if is_soil.sum() < 10:
        return None
    other_axes = [a for a in range(3) if a != up_axis]
    hv = pts[:, other_axes]
    ctr = np.median(hv[is_soil], axis=0)
    r_all = np.linalg.norm(hv - ctr, axis=1)
    r_soil = np.linalg.norm(hv[is_soil] - ctr, axis=1)
    r_soil_99 = np.percentile(r_soil, 99)
    r_all_99 = np.percentile(r_all, 99.9)
    inner_half_frac = float((r_soil < 0.5 * r_soil_99).mean())
    z = pts[:, up_axis]
    order = np.argsort(r_soil)
    n = len(r_soil)
    k = max(1, n // 20)
    z_inner = float(np.median(z[is_soil][order[:k]]))
    z_outer = float(np.median(z[is_soil][order[-k:]]))
    return dict(r_soil_99=float(r_soil_99), r_all_99=float(r_all_99),
                coverage_frac=float(r_soil_99 / max(r_all_99, 1e-9)),
                inner_half_frac=inner_half_frac,
                rim_height_delta=z_outer - z_inner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/apurba/plant3d-data/pheno4d")
    ap.add_argument("--out-dir", default="results/pp_checks")
    ap.add_argument("--slab-bin-width-mm", type=float, default=1.0)
    ap.add_argument("--slab-window-mm", type=float, default=3.0,
                     help="Thickness of the candidate soil slab, mm (plan's soil z-span was "
                          "measured at 2-6mm in step_notes/PP_0 check 2).")
    ap.add_argument("--slab-cell-size-mm", type=float, default=2.0,
                     help="Grid cell size for the lateral-footprint measure, mm.")
    ap.add_argument("--slab-search-frac", type=float, default=0.3,
                     help="Restrict the slab search to the bottom this-fraction of the scan's "
                          "PHYSICAL z-range (not bottom-%% of points -- see module docstring for "
                          "why that distinction is the fix for drooping-leaf scans). Validated "
                          "at 0.2/0.3/0.4 with identical results on a 30-scan sample; 0.3 is the "
                          "default.")
    ap.add_argument("--slab-max-points", type=int, default=150000,
                     help="Subsample cap for the slab search itself (mirrors the superseded "
                          "RANSAC version's candidate cap). The final cut uses the FULL point set.")
    ap.add_argument("--margins", default="0,0.5,1,1.5,2,2.5,3,4,5,7,10,15",
                     help="Comma-separated candidate margins (mm) to evaluate on annotated scans")
    ap.add_argument("--n-unannotated-sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    margins = [float(x) for x in args.margins.split(",")]
    slab_kwargs = dict(bin_width=args.slab_bin_width_mm, window_mm=args.slab_window_mm,
                        cell_size=args.slab_cell_size_mm, search_frac=args.slab_search_frac,
                        max_points=args.slab_max_points, seed=args.seed)

    # ---------------- Part 1: ALL annotated scans, margin tuning ----------------
    annotated_files = sorted(glob.glob(str(Path(args.data_root) / "*" / "*_a.txt")))
    print(f"Found {len(annotated_files)} annotated Pheno4D scans under {args.data_root}")

    per_scan_rows = []
    pot_rows = []
    slab_rows = []
    for fp in annotated_files:
        pts, labels_raw = load_pheno4d_xyz(fp)
        pts = pts.astype(np.float64)
        labels = labels_raw[:, 0] if labels_raw.ndim == 2 else labels_raw.reshape(-1)
        name = Path(fp).name

        slab = find_soil_slab(pts, **slab_kwargs)
        if slab is None:
            print(f"  [warn] slab detection failed on {name} (too few points)")
            continue
        z_lo, z_hi, density, footprint_area = slab
        z0 = z_hi  # cut at the top of the found slab, plus margin

        rows = evaluate_margins(pts, labels, z0, margins)
        for r in rows:
            r["file"] = name
            r["species"] = "Tomato" if name.startswith("T") else "Maize"
            r["z0"] = z0
            r["slab_density"] = density
            r["slab_footprint_mm2"] = footprint_area
            per_scan_rows.append(r)

        # Validation-only (labels used ONLY to check, never to detect): does
        # the found slab actually land inside the true soil's own z-range?
        true_soil_z = pts[labels == 0, UP_AXIS]
        true_med = float(np.median(true_soil_z)) if len(true_soil_z) else float("nan")
        slab_rows.append(dict(file=name, z_lo=z_lo, z_hi=z_hi, true_soil_median_z=true_med,
                               detector_matches_label=bool(z_lo <= true_med <= z_hi)))

        pr = pot_rim_check(pts, labels)
        if pr is not None:
            pr["file"] = name
            pr["species"] = "Tomato" if name.startswith("T") else "Maize"
            pot_rows.append(pr)

    per_scan_df = pd.DataFrame(per_scan_rows)
    per_scan_csv = out_dir / "pheno4d_soil_removal_annotated.csv"
    per_scan_df.to_csv(per_scan_csv, index=False)
    print(f"\nWrote per-scan, per-margin results: {per_scan_csv}")

    slab_df = pd.DataFrame(slab_rows)
    slab_csv = out_dir / "pheno4d_slab_detector_validation.csv"
    slab_df.to_csv(slab_csv, index=False)
    n_match = int(slab_df["detector_matches_label"].sum())
    print(f"\n=== Detector validation (labels used ONLY to check, never to detect) ===")
    print(f"Found slab lands inside the true (label-derived) soil z-range: "
          f"{n_match}/{len(slab_df)} scans")
    mismatches = slab_df[~slab_df["detector_matches_label"]]
    if len(mismatches):
        print("Mismatches (file, found [z_lo,z_hi], true soil median z):")
        for _, r in mismatches.iterrows():
            print(f"  {r['file']}: found=[{r['z_lo']:.2f},{r['z_hi']:.2f}] "
                  f"true_median={r['true_soil_median_z']:.2f}")
    print(f"Wrote {slab_csv}")

    print("\n=== Overall (all annotated scans pooled, point-count-weighted) per margin ===")
    print(f"{'margin(mm)':>10} {'soil_removed%':>14} {'plant_lost%':>12} "
          f"{'min_soil_removed%':>18} {'max_plant_lost%':>16}")
    summary_rows = []
    for m in margins:
        sub = per_scan_df[per_scan_df["margin"] == m]
        tot_soil = sub["n_soil"].sum()
        tot_plant = sub["n_plant"].sum()
        soil_removed_pts = (sub["soil_removed_frac"] * sub["n_soil"]).sum()
        plant_lost_pts = (sub["plant_lost_frac"] * sub["n_plant"]).sum()
        overall_soil = soil_removed_pts / max(tot_soil, 1) * 100
        overall_plant = plant_lost_pts / max(tot_plant, 1) * 100
        min_soil = sub["soil_removed_frac"].min() * 100
        max_plant = sub["plant_lost_frac"].max() * 100
        print(f"{m:>10.1f} {overall_soil:>13.2f}% {overall_plant:>11.3f}% "
              f"{min_soil:>17.2f}% {max_plant:>15.3f}%")
        summary_rows.append(dict(margin=m, overall_soil_removed_pct=overall_soil,
                                  overall_plant_lost_pct=overall_plant,
                                  min_per_scan_soil_removed_pct=min_soil,
                                  max_per_scan_plant_lost_pct=max_plant))
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "pheno4d_soil_removal_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote margin summary: {summary_csv}")

    print("\n=== Worst scans by plant-points-lost at margin=2.0mm ===")
    m2 = per_scan_df[per_scan_df["margin"] == 2.0].sort_values("plant_lost_frac", ascending=False)
    print(m2[["file", "species", "soil_removed_frac", "plant_lost_frac", "n_soil", "n_plant"]]
          .head(10).to_string(index=False))

    print("\n=== Target check (plan's starting target: >95% soil removed AND <1% plant lost) ===")
    hits = summary_df[(summary_df["overall_soil_removed_pct"] > 95) &
                       (summary_df["overall_plant_lost_pct"] < 1)]
    if len(hits) > 0:
        best = hits.sort_values("margin").iloc[0]
        print(f"Target reached at margin={best['margin']}mm: "
              f"soil_removed={best['overall_soil_removed_pct']:.2f}%, "
              f"plant_lost={best['overall_plant_lost_pct']:.3f}%")
    else:
        print("Target NOT reached at any tested margin (reporting honestly, per instructions). "
              "See the per-margin table above and the best-achievable trade-off it implies.")
        best_row = summary_df.loc[summary_df["overall_soil_removed_pct"].sub(95).abs().idxmin()]
        print(f"Closest approach to the soil-removed target: margin={best_row['margin']}mm, "
              f"soil_removed={best_row['overall_soil_removed_pct']:.2f}%, "
              f"plant_lost={best_row['overall_plant_lost_pct']:.3f}%")

    # ---------------- Part 2: pot-rim geometric check ----------------
    pot_df = pd.DataFrame(pot_rows)
    pot_csv = out_dir / "pheno4d_pot_rim_check.csv"
    pot_df.to_csv(pot_csv, index=False)
    print(f"\n=== Pot-rim geometric check (does label-0 hide a separate pot-wall ring?) ===")
    print(f"Wrote {pot_csv}")
    for sp in ("Tomato", "Maize"):
        sub = pot_df[pot_df["species"] == sp]
        if len(sub) == 0:
            continue
        print(f"  {sp}: n={len(sub)}  coverage_frac (soil radius / full-cloud radius): "
              f"median={sub['coverage_frac'].median():.3f}  "
              f"inner_half_frac (hollowness; low=ring): median={sub['inner_half_frac'].median():.3f}  "
              f"rim_height_delta (mm, +=outer edge higher=pot-rim-like): "
              f"median={sub['rim_height_delta'].median():+.3f}")

    # ---------------- Part 3: unannotated scans, same label-free method ----------------
    unannotated_files = sorted(
        f for f in glob.glob(str(Path(args.data_root) / "*" / "*.txt"))
        if not f.endswith("_a.txt"))
    rng = np.random.RandomState(args.seed)
    sample = list(rng.choice(unannotated_files, size=min(args.n_unannotated_sample,
                                                          len(unannotated_files)), replace=False))
    chosen_margin = float(summary_df.loc[
        (summary_df["overall_plant_lost_pct"] - 1).abs().idxmin(), "margin"]) if len(summary_df) else margins[0]
    print(f"\n=== Unannotated-scan pass (same label-free detector; no ground truth, "
          f"points-removed% only), n={len(sample)} of {len(unannotated_files)} ===")
    print(f"Using margin={chosen_margin}mm (closest overall pooled plant-lost% to the 1% target)")
    sanity_rows = []
    for fp in sample:
        pts, _ = load_pheno4d_xyz(fp)
        pts = pts.astype(np.float64)
        name = Path(fp).name
        slab = find_soil_slab(pts, **slab_kwargs)
        if slab is None:
            print(f"  [warn] slab detection failed on {name}")
            continue
        z_lo, z_hi, density, footprint_area = slab
        z0 = z_hi
        z = pts[:, UP_AXIS]
        removed_frac = float((z <= (z0 + chosen_margin)).mean())
        print(f"  {name}: n={len(pts)}  found_slab=[{z_lo:.2f},{z_hi:.2f}] density={density} "
              f"footprint={footprint_area:.0f}mm^2  points_removed={removed_frac*100:.2f}%")
        sanity_rows.append(dict(file=name, n=len(pts), z_lo=z_lo, z_hi=z_hi, density=density,
                                 footprint_mm2=footprint_area, points_removed_frac=removed_frac))
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_csv = out_dir / "pheno4d_soil_removal_unannotated_sanity.csv"
    sanity_df.to_csv(sanity_csv, index=False)
    print(f"\nWrote unannotated results: {sanity_csv}")
    if len(sanity_df):
        print(f"points_removed% across the sample: min={sanity_df['points_removed_frac'].min()*100:.2f}% "
              f"median={sanity_df['points_removed_frac'].median()*100:.2f}% "
              f"max={sanity_df['points_removed_frac'].max()*100:.2f}%")


if __name__ == "__main__":
    main()
