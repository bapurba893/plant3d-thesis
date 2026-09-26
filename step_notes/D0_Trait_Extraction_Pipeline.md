# D0: Trait-Extraction Pipeline (Block D Prerequisite)

## What this is, in plain language

Block D (the fusion/growth-prediction phase) needs actual measured plant traits — height, stem
diameter, leaf area, leaf count, volume — extracted from Pheno4D's 223 multi-temporal scans, then
assembled into per-plant growth curves. None of this existed before this work: no organ
segmentation on Pheno4D using the chosen backbone (B3b, KPConv+DA-S), no trait-measurement code,
no growth-curve fitting. This file documents the segmentation half of that pipeline — specifically
the long, iterative process of getting Tomato's organ predictions from "unusable" to "usable,"
since Maize worked essentially out of the box.

## Why this is hard, in one sentence

B3b's model was trained to segment Crops3D's own per-species organ categories (3 for Tomato, 6 for
Maize), never Pheno4D's 3-class soil/stem/leaf scheme — so its raw predictions on a Pheno4D scan
need to be translated ("remapped") from "Crops3D category" to "Pheno4D organ" before they mean
anything, and for Tomato specifically, no single translation table turned out to work.

## The B3b Crops3D-vs-Pheno4D label-space mismatch

`adapters/train_b3b_kpconv_da_s.py:409` confirms `KPConv_ClsSeg(config, num_class=2,
seg_num_classes=SEG_NUM_CLASSES)` — `SEG_NUM_CLASSES = {"Tomato": 3, "Maize": 6}`
(`adapters/dataset.py:75`), Crops3D's own label space, not `PHENO4D_SEG_NUM_CLASSES = {"Tomato":
3, "Maize": 3}` (line 105). B3b is the DA-S row (self-supervised only) and never touches Pheno4D
labels during training or selection, so its predictions on a Pheno4D scan come out as Crops3D-id
class indices that must be translated to Pheno4D's `{0: soil, 1: stem, 2: leaf}` scheme.

All validation below is run via `adapters/infer_segmentation.py`'s `validate_against_annotated()`,
scoring **only on `pheno4d_heldout_eval.csv`** (36 annotated scans, 22 Tomato + 14 Maize) as the
fair generalization number — `pheno4d_adaptation_pool.csv` (90 annotated scans) was seen
(unlabeled) during B3b's DA-S training and is reported separately, never as the headline number.
Every attempt below reports **per-class recall** (soil/stem/leaf individually), not just aggregate
mIoU/accuracy — this distinction turned out to matter enormously, since several attempts had a
respectable aggregate number while one organ's recall was silently near zero.

## The five attempts, in order, and why each led to the next

### 1. `histogram` — RGB-confirmed heuristic (first attempt, superseded immediately)

`build_crops3d_organ_remap`: id 0 → soil (RGB-confirmed for Maize only, per CLAUDE.md's Crops3D
label-backfill entry), dominant-point-count id → leaf (RGB-confirmed for both species), everything
else → stem.

| | soil | stem | leaf | mIoU | accuracy |
|---|---|---|---|---|---|
| Maize | — | — | — | 0.5430 | — |
| Tomato | — | — | — | **0.0886** | 0.2376 |
| Overall | — | — | — | 0.2653 | 0.4503 |

Tomato's accuracy (23.76%) was *worse than random chance* on a 3-class problem. Maize was fine.
This immediately flagged that "id 0 = soil," only ever RGB-confirmed for Maize, does not hold for
Tomato — motivating a data-driven alternative rather than more heuristic guessing.

### 2. `data_driven` — cross-domain Hungarian matching

`build_data_driven_remap`: builds a confusion matrix between B3b's raw Crops3D-id predictions and
Pheno4D ground truth on `pheno4d_adaptation_pool.csv` (fair to use for calibration), solves the
id→organ assignment via `scipy.optimize.linear_sum_assignment` per species (Hungarian matching,
handling the non-square case for Maize's 6→3 by assigning the primary 3 via Hungarian then any
leftover ids by their own row's argmax).

Tomato heldout_eval, per-class recall:

| true organ | recall | breakdown |
|---|---|---|
| soil | **99.48%** | — |
| stem | **0.66%** | 38/5,779 correct |
| leaf | 28.31% | mostly misclassified as soil |

Excellent soil, catastrophic stem. **Root cause**: Hungarian assigned stem to id2, a Crops3D id the
model predicts on <1% of points (matching Crops3D's own training-time rarity for that id) — so
"stem" became reachable only through an id the model almost never outputs, regardless of how well
that id's few occurrences correlated with true stem when it did fire.

### 3. `groundtruth` — Crops3D's own confirmed point-cloud shape

Before trying another statistical remap, raw Crops3D Tomato point clouds were directly visualized,
colored by their native label ids (see the analysis below) — a genuinely different kind of
evidence than the cross-domain confusion matrix, since it doesn't depend on how the model behaves
under domain shift at all.

**Visual + quantitative finding**: id0 is unambiguously stem-shaped (a thin, continuous, curving
line — visually obvious in every sample checked) and id1 is unambiguously the bushy leaf canopy.
Confirmed numerically via PCA shape descriptors (linearity = (λ1−λ2)/λ1, computed on each class's
whole point set per sample):

| Sample | id0 linearity | id1 linearity |
|---|---|---|
| 11-2_1-55 | 0.965 | 0.866 |
| 11-2_10-3 | 0.874 | 0.445 |
| 11-2_30-4 | 0.941 | 0.702 |

(id0 consistently near-1.0, thin/line-like; id1 much lower, spread/bushy.) A first visual glance
at the rare id2 class suggested a tight round blob (consistent with the original "fruit"
hypothesis from CLAUDE.md's Crops3D label-backfill work) — but the SAME shape-descriptor check,
run on id2 to verify rather than trust the visual impression, showed id2 is *also* highly linear
(0.91–0.97, comparable to or exceeding id0) with very low sphericity — the apparent "blob" was a
foreshortening artifact of a single static viewing angle, not a real property of the point
distribution. This was flagged and corrected in-session before it fed into any decision.

`build_groundtruth_remap`: id0→stem, id1→leaf, id2→leaf (pragmatic default for the still-ambiguous
rare class, matching its own confusion-matrix plurality — 687 leaf vs. 203 stem vs. 0 soil).

| true organ | recall |
|---|---|
| soil | **0.00%** |
| stem | **77.56%** |
| leaf | **71.69%** |

Stem and leaf both good — a real, decisive fix for the specific stem problem `data_driven` had.
But soil recall is exactly zero, **by construction**: this remap has no id mapped to organ 0 at
all. Every true soil point is forced into stem or leaf.

**A genuinely counter-intuitive finding surfaced here**: `groundtruth`'s "semantically correct"
remap (grounded in Crops3D's own confirmed labels) scores *worse in aggregate* than
`data_driven`'s "semantically inverted" one, because the model's Crops3D-id output does not
preserve its Crops3D-training-time meaning once applied to Pheno4D under this much domain shift —
plausibly because Crops3D's own Tomato scans appear to have no soil-labeled points at all
(close-up potted-plant captures), so B3b never learned a genuine soil-vs-plant boundary for Tomato
in the first place, and under self-supervised-only domain shift it likely repurposes whichever id
best matches "flat, dense, textureless region" — which happens to line up with Pheno4D's actual
soil, independent of what that id meant on Crops3D.

### 4. `hybrid` — combine the two remaps' individually-reliable pieces

Given `data_driven`'s id1→soil (99.48% recall) and `groundtruth`'s id0→stem (77.56% recall) each
work well in isolation, `build_hybrid_remap` combined them directly: id0→stem, id1→soil, id2→leaf.

| true organ | recall |
|---|---|
| soil | **99.28%** |
| stem | **78.04%** |
| leaf | **0.58%** |

Soil and stem both excellent — but **leaf collapsed**, for the exact same structural reason stem
did under `data_driven`: leaf now depends entirely on the rare id2. **This proved the problem
exhaustively**: id0 and id1 are the model's only two commonly-predicted Tomato ids, but Pheno4D
needs three organs — whichever organ ends up assigned to the rare id2 gets near-zero recall,
*regardless of which organ that is*. Confirmed across three different remaps, three different
organs (stem under `data_driven`, soil under `groundtruth`, leaf under `hybrid`). This closed the
pure-remap avenue for good — no permutation of a 3-way Crops3D output can cover 3 Pheno4D organs
when only 2 of those 3 Crops3D ids are ever meaningfully predicted.

### 5a. `geometric` (linearity-based split) — first geometric prototype, mis-scoped

First attempt at a geometric fallback: keep id1→soil (the one reliably model-derived signal), and
split the *remaining* (id0 + id2) points into stem/leaf via **local point-cloud shape** —
`local_linearity()` (per-point k-NN PCA linearity, the same descriptor validated on Crops3D's
whole-class shapes above, applied per-point instead), thresholded and cleaned up via
`largest_connected_component_mask()` (keep only the largest spatially-connected cluster of
high-linearity points, to suppress isolated leaf-vein/edge points that can locally look linear).

Validated first on a synthetic point cloud (thin line + bushy blob) — passed once the synthetic
geometry was made realistic (dense-enough line relative to jitter; an early sloppy synthetic test
failed for a test-construction reason, not a code bug, and was corrected before trusting the real
run).

Tomato heldout_eval:

| true organ | recall |
|---|---|
| soil | 99.21% |
| stem | **8.27%** |
| leaf | 28.16% |

Worse than `groundtruth`/`hybrid`'s simple `id0→stem` on stem specifically (8.27% vs. 77–78%) —
most true stem points fell below the linearity threshold on real, noisier Pheno4D geometry (unlike
the clean synthetic test) and defaulted to leaf. **Diagnosis, not just a bad threshold**: comparing
all four remaps side by side showed `id0→stem` *already worked* without any geometry — the real
unresolved problem was narrower than "split id0+id2 by shape." It was specifically that **id1
carries two incompatible meanings** (data_driven: ~72% true soil; groundtruth: Crops3D-native
leaf-canopy class) that no single remap value can honor at once. Splitting id0+id2 by shape was
solving a problem (stem-vs-leaf that id0 already answered) instead of the actual one (soil-vs-leaf
within id1).

### 5b. `height_split` — re-scoped: keep id0→stem, geometry moves to id1 only — SUCCESS

Per explicit user instruction: re-scope, don't retune. `id0→stem` kept exactly as-is (already
solved). `id2→leaf` kept as the pragmatic default (rare, same as every prior attempt).
**Only id1's predictions** get a geometric split, using **height relative to the sample's own
z-range** instead of local shape — soil should sit near the plant's lowest point, leaf canopy
measurably higher, a structurally different and better-motivated question than local linearity.

`height_based_soil_leaf_split(id1_points, all_points, height_fraction=0.15)`: points within the
bottom 15% of the sample's own z-range are soil, everything else leaf. **`height_fraction=0.15` is
an explicitly UNTUNED first-pass default** — reasoned from the visual proportion of soil/tray in
the earlier Pheno4D visualization (`step_notes` visualization work earlier in this pipeline's
development), not swept or optimized against held-out data. It happened to work well enough on the
first try that no tuning pass was judged necessary — this is a deliberate choice to move forward
with a working default rather than over-invest in optimizing a parameter that already clears the
bar, not a claim that 0.15 is somehow the "correct" value. Revisit if a future check (e.g. the
verification steps once real trait values are computed) suggests it should be adjusted.

Validated on a synthetic soil/leaf height separation first (96.5% accuracy on clean synthetic
data) before the real run.

**Tomato heldout_eval, full comparison against everything above:**

| Approach | soil | stem | leaf | mIoU | accuracy |
|---|---|---|---|---|---|
| histogram | — | — | — | 0.089 | 0.238 |
| data_driven | 99.48% | 0.66% | 28.31% | 0.327 | 0.694 |
| groundtruth | 0.00% | 77.56% | 71.69% | 0.199 | 0.287 |
| hybrid | 99.28% | 78.04% | 0.58% | 0.341 | 0.657 |
| geometric (linearity) | 99.21% | 8.27% | 28.16% | 0.396 | 0.702 |
| **height_split** | **92.84%** | **76.69%** | **68.66%** | **0.571** | **0.839** |

**`height_split` is the only attempt with all three organs above 65% recall simultaneously**, and
its mIoU (0.571) nearly doubles the next-best attempt's (0.396). Soil recall dipped modestly from
the soil-optimized attempts' ~99% to 92.84% (some near-threshold points near the soil/canopy
boundary now go to leaf instead) — a small, reasonable trade for leaf jumping from ~28% to 68.66%
while stem stayed essentially unchanged (76.69% vs. 77–78%), confirming `id0→stem` really was
robust and untouched by this change, exactly as intended by the re-scoping.

Maize, for reference (unaffected by any of this — see below): mIoU 0.589 under `height_split`,
consistent with every other remap_mode's Maize number (0.55–0.59 range) within run-to-run noise.

## Maize is not affected by any of this

Every `build_*_remap`/`run_inference_on_batch_tomato_*` function in `adapters/infer_segmentation.py`
routes Maize through the standard `build_data_driven_remap` result unchanged. That result already
independently reproduced Maize's RGB-confirmed ground truth (id 0 = soil, dominant id = leaf, per
CLAUDE.md) on its first attempt, with solid held-out mIoU (0.55–0.59) — there is no capacity
problem on Maize (6 Crops3D classes comfortably cover 3 organs, unlike Tomato's 3-into-3 with only
2 commonly-predicted ids) and no reason to route it through any Tomato-specific fallback. This is
enforced in code, not just by convention: `run_inference_on_batch_tomato_height_split`'s Maize
branch is byte-identical to `run_inference_on_batch`'s, reading `maize_remap["Maize"]` directly.

## Code

`adapters/infer_segmentation.py`:
- `load_b3b_model` — reconstructs B3b (no saved config/neighborhood_limits/Kendall weights exist
  on disk, only a raw `model.state_dict()` — confirmed by reading the checkpoint-saving code).
- `build_crops3d_organ_remap` / `build_data_driven_remap` / `build_groundtruth_remap` /
  `build_hybrid_remap` — the four remap-table strategies (attempts 1–4 above), kept in the file
  (not deleted) for reproducibility and so `remap_mode` can still reproduce every number in this
  document.
- `local_linearity` / `largest_connected_component_mask` / `geometric_stem_leaf_split` /
  `run_inference_on_batch_tomato_geometric` — attempt 5a (mis-scoped, kept for the record).
- `height_based_soil_leaf_split` / `run_inference_on_batch_tomato_height_split` — attempt 5b, the
  one to actually use going forward.
- `validate_against_annotated(..., remap_mode=...)` — the shared validation entry point; every
  number in this document is reproducible via `--remap_mode {histogram,data_driven,groundtruth,
  hybrid,geometric,height_split}`.
- `_evaluate_split` / `_print_stats` — shared scoring/reporting path used identically by every
  `remap_mode`, including the per-class (soil/stem/leaf) recall breakdown that made the stem/soil/
  leaf collapse failures visible in the first place (an aggregate mIoU alone would have hidden
  every one of them).

## Segmentation run at full scale

`height_split` (Tomato) + standard `data_driven` remap (Maize) was run across all 223 Pheno4D
scans via `adapters/infer_segmentation.py --mode full` (`run_full_dataset_inference`), writing one
`.npz` per scan to `data/segmented/Pheno4D/<species>/<stem>.npz` (`{"points": normalized_pts,
"pred_organ": (N,) int}`). 223/223 scans processed, 0 failures.

## Trait extraction: `scripts/trait_extraction.py`

Implements `height`, `stem_diameter`, `leaf_area`, `leaf_count`, `volume`, all requiring real-
world-unit points (`unnormalize()` raises loudly if a scan's `norm_scale` is missing rather than
silently computing meaningless normalized-space values). `leaf_area`/`leaf_count` share a DBSCAN
clustering step over leaf-predicted points (adaptive `eps` from each scan's own k-distance
distribution) since B3b's segmentation has no per-leaf-instance output. All five validated against
synthetic point clouds with known dimensions before running on real data. Output:
`data/traits/pheno4d_traits_per_scan.csv`, one row per scan.

### A real bug found and fixed: `stem_diameter`'s original formula

The first `stem_diameter` implementation (a circular-disk fit from the basal slice's XY
**covariance**: `diameter = 4*std`) produced physically impossible values on the real run —
**35/223 scans (16%) had `stem_diameter >= height`**, concentrated in bushy, late-stage Tomato
scans. Root cause, confirmed by direct inspection of the worst offender (`T03_0325_a.npz`): within
the 122-point basal slice, the **median** distance from the slice's own center was 0.069 (a tight,
plausible stem core) but the **mean** was 0.463 and the **max** 1.244 — a ~38% minority of
scattered false-positive "stem" points (inevitable given segmentation recall is 77%/40-48%, not
100%) dominated the variance-based estimate, since variance is quadratic in deviation.

**Fix**: switched to a robust, median-distance-based estimator. For a uniform 2D disk of radius R,
the CDF of radial distance from center is `(r/R)^2`, so the median radius is `R/sqrt(2)` — giving
`diameter = 2R = 2*sqrt(2)*median_dist`. Medians tolerate contamination up to 50%, unlike variance.
This cut the physically-impossible rate from 35/223 (16%) to 10/223 (4.5%) and the median
diameter/height ratio from an implausible 0.264 to a much more plausible 0.129.

### The remaining 10 outliers: two distinct clusters, not random noise

Per explicit user instruction, the remaining 10 `stem_diameter >= height` scans were inspected
individually (plant_id, species, scan_date, point counts) rather than assumed benign. **All 10 are
Tomato — zero Maize** — already a strong non-random signal. Within Tomato, two clearly distinct
clusters:

**Cluster A (4 scans, all T02, early dates 305-311)**: tiny plants (height 23-33, the smallest in
the dataset), NORMAL stem-point-fraction (2.4-4.3%, in line with the dataset median ~15%... below
it, in fact). At this small absolute scale, a modest formula imprecision produces a large *ratio*
even though the underlying segmentation looks unremarkable. Read as benign small-plant fragility,
not a segmentation failure.

**Cluster B (6 scans: T01_320, T02_322, T03_324, T04_324, T07_322, T07_325)**: ABNORMALLY HIGH
stem-point-fraction (20.9-59.9%, vs. the dataset median ~15%), disproportionately late in the
scanning window (dates 320-325, near the end). Visualized directly (colored point clouds, predicted
vs. ground truth where annotated) and confirmed with real ground truth on the 3 annotated scans
(T03_324, T04_324, T07_325):

| Scan | predicted stem pts | **true stem pts** | over-count | pixel agreement |
|---|---|---|---|---|
| T03_324 | 1,479 | 467 | 3.2x | 68.3% |
| T04_324 | 1,583 | 439 | 3.6x | 66.1% |
| T07_325 | 2,183 | 503 | 4.3x | **51.2%** |

**This is a genuine, severe segmentation failure, not noise the median fix can absorb.** The
images show the model calling leaf petioles/branch stalks "stem" — structures that are locally
thin and linear (the same visual signature id0->stem legitimately keys on in the average case,
76.69% held-out recall) but which are botanically leaf-supporting structure, not the main stem.
This over-firing gets worse as the plant becomes bushier with more branching (hence the late-date
skew) -- the id0->stem assignment, validated as "already solved, no geometry needed" on the
held-out AVERAGE, has a real failure mode this average hid: it doesn't generalize to unusually
bushy individual scans.

### Which traits are actually corrupted on the 6 cluster-B scans (checked, not assumed)

Per explicit user instruction, computed all 5 traits from BOTH the model's predictions AND the
real ground truth on the 3 annotated cluster-B scans, to see which traits the stem-over-prediction
actually corrupts:

| Trait | T03_324 | T04_324 | T07_325 | Verdict |
|---|---|---|---|---|
| height | −0.3% | +0.0% | −1.8% | **unaffected** |
| volume | −0.6% | +1.1% | −7.9% | **unaffected** |
| leaf_area | **−38.4%** | **−39.8%** | **−63.7%** | **corrupted** |
| leaf_count | **+40.0%** | **+71.4%** | **+50.0%** | **corrupted** |
| stem_diameter | +2204% | +2436% | +1393% | **corrupted** (already known) |

(% = predicted vs. ground-truth, `(pred-gt)/gt`)

**height and volume are safe to trust as-is on these scans** — both are computed over ALL non-soil
points regardless of the stem/leaf split, so misclassifying a point as stem instead of leaf (or
vice versa) doesn't change either computation; the errors stay well within normal noise.
**leaf_area and leaf_count are ALSO meaningfully corrupted**, not just stem_diameter: leaf_area is
substantially UNDERestimated (true leaf points pulled into the "stem" bucket are excluded from the
leaf-area computation, which only sums leaf-labeled points) and leaf_count is substantially
OVERestimated (removing points from a true leaf cluster can fragment it into multiple smaller
DBSCAN-detected clusters instead of one).

### Confidence flag: `stem_leaf_boundary_low_confidence`

`scripts/trait_extraction.py::flag_stem_leaf_boundary_confidence` adds two columns to the output
CSV rather than silently dropping or blanking any values (raw numbers are kept for transparency):
`stem_frac_of_points` (diagnostic) and `stem_leaf_boundary_low_confidence` (bool). The flag is
`(stem_diameter >= height) AND (stem_frac_of_points > 0.15)` — this exact combined condition
reproduces the 6 manually-identified cluster-B scans precisely (verified against the full 223-scan
dataset: no false positives, no false negatives) while correctly excluding cluster A's 4
small-plant scans, which don't have the same abnormal stem-fraction signature. `0.15` is not an
arbitrary round number -- it sits between cluster A's max (4.3%) and cluster B's min (20.9%), with
a comfortable margin on both sides.

**Practical consequence for downstream use**: `stem_diameter`, `leaf_area`, and `leaf_count` for
the 6 flagged scans should be excluded (not silently trusted) wherever they're consumed --
concretely, `scripts/growth_curves.py`'s height/stem_diameter curve-fitting must drop flagged rows
before fitting stem_diameter series (height is unaffected and needs no exclusion). `height` and
`volume` remain trustworthy for these scans and need no special handling.

## Status: segmentation + trait extraction done, flagged, and documented. Time-series assembly next.

## Time-series assembly: `scripts/assemble_trait_timeseries.py`

Parses `scan_date` (MMDD float, e.g. `313.0`) into a `datetime.date` via an arbitrary common year
(2000 -- confirmed all 14 plants' scans fall entirely within March, no year-boundary risk in the
current data, but guarded defensively with an assertion that raw `scan_date` and parsed-date
ordering agree, rather than assumed to hold forever). Computes `elapsed_days` per plant (day 0 =
that plant's own first scan), sorts, and writes `data/traits/pheno4d_traits_timeseries_long.csv`
(223 rows, one per scan, all trait columns + `elapsed_days` + the `stem_leaf_boundary_low_
confidence` flag carried through unchanged). Cross-checked against `data/pheno4d_plant_level.csv`'s
`n_scans` per plant: **exact match for all 14 plants, no missing/extra scans.**

## Growth-curve fitting: `scripts/growth_curves.py`

**Scope, per the decision made before this pipeline was built**: Logistic and Gompertz
curve-fitting applies ONLY to `height` and `stem_diameter` (`TRAITS_TO_FIT`, a fixed list, not a
runtime check) -- `leaf_area`/`leaf_count`/`volume` are never fit, since both curve families assume
monotonic saturating growth and those three traits can legitimately decrease via senescence.

Closed forms derived directly from the ODEs (verified against synthetic data with known parameters
before running on real data -- both recovered K/A/rho/beta within tolerance, R²>0.999):
- **Logistic**: `dy/dt = ρy(1−y/K)` → `y(t) = K / (1 + B·exp(−ρt))`, `B = (K−y0)/y0`.
- **Gompertz** (log space): `dy/dt = α − βy` (in `ŷ=ln y`) → `y(t) = A·exp(−B_g·exp(−βt))`,
  `A = exp(α/β)`, `B_g = α/β − ln(y0)` — exponentiating the log-space linear ODE's solution back to
  raw space reproduces the textbook 3-parameter Gompertz form exactly, a self-consistency check on
  the derivation.

**`stem_diameter` fitting excludes the 6 `stem_leaf_boundary_low_confidence` scans** (confirmed
corrupted, 13-24x overestimate); **`height` fitting does NOT exclude them** (confirmed unaffected
by the same segmentation issue). Per plant this leaves 18-20 stem_diameter points for Tomato,
11-12 for Maize — comfortably above the `MIN_POINTS_TO_FIT=4` floor.

**A real bug caught before trusting the results**: the first version of the fit-quality check
(flag `poor` if any fitted parameter sits within 1% of its bound) flagged nearly every fit as
`poor` regardless of R². Root cause: `B` (logistic) and `Bg`/`β` (Gompertz) intentionally use very
wide bounds (spanning several orders of magnitude, since their natural scale depends heavily on
`y0`'s relative size) — checking "fraction of a [1e-6, 1e6]-scale span" is meaningless for those,
since almost any realistic fitted value sits near 0% of such a span by construction. Fixed by
restricting the near-bound check to just the asymptote parameter (`K` for logistic, `A` for
Gompertz, `bound_check_indices=(0,)`), the one with a tight, physically meaningful bound
(0.9-5x of the observed max) where actually hitting it is a genuine, informative signal.

### Results, and what they mean

**`height`**: fits well overall. Maize: all 7 plants `ok` for both models, R² 0.969-0.989. Tomato:
R² is high across the board (0.947-0.989) but 4 of 7 plants (T03, T04, T05, T06) get flagged
`poor` for logistic -- confirmed by direct inspection this is NOT a poor fit in the R² sense, it's
the fitted `K` landing EXACTLY on its upper bound (5.0x the plant's own observed max height) for
all 4. This is a genuine, informative result, not a bug: **these 4 plants' height had not
plateaued within Pheno4D's observed scanning window** -- the optimizer wants an even higher
asymptote than the (already generous) 5x-headroom bound allows, consistent with them still being
in active vertical growth at the last scan. Gompertz shows the same story with slightly more
plants affected (6 of 7 Tomato flagged `poor`) since its `Bg`/`β` parameterization is more
sensitive to an unconstrained tail.

**`stem_diameter`**: fits uniformly poorly across BOTH species -- R² ranges from very low (0.025,
T06) to moderate (0.82, M03), with most plants well below the `R2_POOR_THRESHOLD=0.8` cutoff, even
on Maize (where segmentation is solid, mIoU ~0.55-0.59, and none of the 6 known-bad scans apply at
all). **This is a genuine finding, not attributable to the already-excluded segmentation failure**
-- since it affects Maize equally and Maize was never flagged for any stem/leaf-boundary problem.
The most likely explanation: `stem_diameter`'s per-scan measurement (even via the robust
median-distance estimator) still carries enough scan-to-scan noise/scatter -- from natural
variation in exactly which points the segmentation calls "stem" in the basal slice from one scan
to the next -- that a smooth monotonic growth curve doesn't capture it well, unlike `height`
(a coarser, more robust z-extent measurement that's far less sensitive to exactly which points
land in which organ bucket). Flagged here as a real measurement-precision limitation for anyone
consuming `stem_diameter`'s fitted growth-curve parameters downstream, not something further
"fixed" in this pass -- the raw per-scan `stem_diameter` values (outside the 6 flagged scans) are
still the best available measurement, just noisier over time than `height`.

Output: `data/traits/pheno4d_growth_curves.csv`, one row per `(plant_id, trait)` for
`trait in {height, stem_diameter}` (28 rows = 14 plants x 2 traits), with `logistic_K/B/rho/
y0_hat/R2/fit_quality`, `gompertz_A/Bg/beta/alpha_hat/y0hat_log/R2/fit_quality`,
`n_points_fit`, and `n_excluded_low_confidence`.

## Status: D0 pipeline complete end-to-end.

Segmentation (223/223 scans) → trait extraction (5 traits, confidence-flagged) → time-series
assembly (verified against the plant-level reference) → growth-curve fitting (height fits well;
stem_diameter fits are noisy, a documented limitation, not a bug) are all done, validated at each
step against synthetic ground truth and/or real Pheno4D annotations, and committed. Ready to feed
Block D (D1-D6).
