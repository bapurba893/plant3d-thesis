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

## Status: segmentation approach settled. Full run pending.

`height_split` (Tomato) + standard `data_driven` remap (Maize) is the segmentation approach to use
for the full 223-scan trait-extraction run. Preprocessing's `norm_scale`/`norm_center` fix (see
`scripts/preprocessing.py`, committed separately) is already in place, so real-world-unit trait
extraction can proceed once segmentation runs at full scale.
