# D2: Fusion + Growth-Curve Params (Block D, F2)

**Status: complete.** Identical to D1 in every respect (frozen B3b/KPConv+DA-S pooled feature,
`TraitFusionHead`'s linear-head architecture, `masked_regression_loss`/`run_epoch`/
`compute_metrics`, the 7-term `L_trait`+`L_growth` Kendall-weighted loss, the same Oracle-style
plant-disjoint split) except the fusion input, per CLAUDE.md's Block D isolation rule: D2
concatenates each plant's already-fit D0 growth-curve parameters onto B3b's 256-dim pooled
feature (268-dim total input).

## Design decision (flagged as inference, not spec text)

The design docs say "F2: + growth curve params" but never specify which parameters or how they
attach to a per-scan sample. Used ALL 6 raw numeric parameters each curve family produces
(`logistic_K/B/rho`, `gompertz_A/Bg/beta`) for both height and stem_diameter (12 params total),
not just the "winning" family per plant — simplest, most complete interpretation, consistent
with this project's general preference for letting a downstream model discover what's useful
rather than hand-engineering a reduction. Each plant's growth-curve parameters are a property of
the whole fitted trajectory (an ODE's own shape/rate/asymptote constants), so the same 12-dim
vector is broadcast to every scan belonging to that plant — standard practice for panel/
hierarchical growth-curve features, and does not leak across plants (the Oracle split stays
fully plant-disjoint; each plant's own curve was fit only from its own scans, same as D0).
Confirmed via `pheno4d_growth_curves.csv`: all 14 plants × 2 traits converged for both families,
no NaNs — no missing-value handling needed.

stem_diameter's curve fits were already confirmed poor project-wide in D0 (R² as low as 0.025).
Those 6 stem_diameter parameters are still included as input here (unlike their exclusion from
L_growth's targets in D1) — a poorly-fit curve is still a real summary of trajectory shape, and
whether it helps or hurts is exactly the empirical question D2 vs D1 answers. It hurt (see below).

## Bug found and fixed on the first real run (2026-09-27)

First run produced test R² in the negative hundreds on every term (e.g. height RMSE=630 vs. a
~20-450 physical scale) — not a modeling result, a real bug. Root cause: test plant M03's
`stem_diameter_logistic_B` fit to **74,464**, a z-score of **+2254** against train-only
standardization statistics, while every other one of the 14 plants' values sat in [0.7, 380].
This is exactly the parameter class `growth_curves.py`'s own docstring already flags as using
"very wide bounds (many orders of magnitude)... physically non-meaningful" and deliberately
excludes from its own fit-quality check (`_fit_one`'s `bound_check_indices`) — a poorly-
identified shape parameter on an already-known-poor-quality trait fit. Checked whether the
train-side standardization statistics were themselves corrupted first (they weren't — no train
plant has this problem, confirmed directly). With `TraitFusionHead`'s linear head (no hidden-
layer nonlinearity to contain a single exploding input), that one column's z=+2254 value
corrupted every one of the 7 output predictions for M03's scans (a linear layer's weight matrix
mixes every input into every output).

**Fix**: standardized growth-curve-param features are clipped to `±CLIP_STD=5.0` before entering
the model (`D2FusionDataset.__getitem__`) — bounds worst-case extrapolation without discarding
signal for the other 13, well-behaved plants or changing which parameters are included. Verified
directly (loaded M03's row, confirmed both previously-exploding columns clip to exactly ±5.0,
all others unaffected) before rerunning.

## Real-data result (2026-09-27), after the fix

Same 72/18/63 Oracle split, `wd=1e-2`, linear head (`hidden_dim=0`), 200 epochs, best epoch 175
(val total loss 0.0947).

**Per-term R² (train n=72 / val n=18 / test n=63), D1 vs. D2:**

| Term | D1 train | D2 train | D1 val | D2 val | D1 test | D2 test |
|---|---|---|---|---|---|---|
| height | 0.9695 | 0.9704 | 0.9711 | 0.9634 | 0.8802 | 0.8891 |
| height_rate | 0.8350 | 0.9036 | 0.8739 | 0.8716 | 0.7165 | 0.7670 |
| stem_diameter | 0.3737 | 0.3990 | 0.3272 | 0.2495 | 0.1438 | 0.1474 |
| stem_diameter_rate | 0.3538 | 0.3989 | −0.1548 | −0.1830 | 0.1464 | 0.2118 |
| leaf_area | 0.6235 | 0.7157 | 0.5956 | 0.3346 | 0.5952 | **−0.5069** |
| leaf_count | 0.3777 | 0.4103 | 0.1844 | 0.2652 | 0.3439 | 0.1175 |
| volume | 0.7055 | 0.7302 | 0.8444 | 0.8534 | 0.0618 | 0.0527 |

Full RMSE/MAE/R² in `results/D2_fusion_growth_curves/run.log`.

## D2 vs. D1 — mixed, not a clear win

- **Modest, real gains**: height_rate (+0.05 test R²), stem_diameter_rate (+0.07 test R²) —
  plausibly because the growth-curve rate parameters (rho/beta) give the model information
  directly related to these two targets specifically.
- **Roughly flat**: height, stem_diameter, volume — no meaningful change either direction.
- **A real regression, root-caused via ablation (2026-09-27), not left as a guess**: leaf_area's
  test R² collapsed from 0.5952 to **−0.5069**. First checked whether the trained weight matrix
  puts disproportionate magnitude on the 12 growth-curve dims vs. the 256 B3b dims (mean |weight|
  ratio) — inconclusive: leaf_area's ratio (1.254) was actually LOWER than height_rate's (1.520),
  which improved, so weight magnitude alone doesn't separate the terms that broke from the ones
  that didn't. Ran a more direct causal test instead: zeroed the 12 growth-curve input dims at
  inference time on the already-trained model (same weights, no retraining) and recomputed test
  R² for all 7 terms.

  | Term | As-trained | Growth dims zeroed | D1's own R² |
  |---|---|---|---|
  | leaf_area | −0.5069 | **0.5540** | 0.5952 |
  | leaf_count | 0.1175 | 0.3126 | 0.3439 |
  | volume | 0.0527 | 0.1541 | 0.0618 |
  | height | 0.8891 | 0.8958 | 0.8802 |
  | stem_diameter | 0.1474 | 0.1191 | 0.1438 |
  | height_rate | 0.7670 | 0.5104 | 0.7165 |
  | stem_diameter_rate | 0.2118 | 0.1625 | 0.1464 |

  **Confirmed, not just plausible**: zeroing only the growth-curve dims (keeping every learned
  weight fixed) restores leaf_area to 0.554 — almost exactly D1's own 0.595. leaf_count and
  volume show the same pattern to a lesser degree. The flip side is equally clean: height_rate
  and stem_diameter_rate get WORSE when zeroed (0.767→0.510, 0.212→0.163) — those are the only
  two terms genuinely using the growth-curve signal productively (unsurprising, since they're
  the two terms most directly related to the trajectory-shape parameters being added). Every
  other term is picking up noise through the SHARED linear layer's single weight matrix, which
  lets 12 dimensions relevant to only 2 of 7 terms corrupt the other 5's predictions.
- **Motivating question (does D2 fix D1's diagnosed Maize-volume generalization failure)**:
  checked directly by species. Maize test volume R² moved **−0.270 (D1) → −0.207 (D2)** — a
  marginal nudge, still deeply negative/broken, not resolved. Tomato volume actually got worse
  (0.674→0.520). Per-plant SSE breakdown confirms M03/M04 (Maize) still dominate test error.

**Conclusion: growth-curve parameters, as included here (all 12 raw params, both trait
families), do not clearly help D2 over D1** — two real, causally-confirmed gains on the rate
terms they're most directly related to, no help for the identified Maize-volume weak point, and
a causally-confirmed regression on leaf_area (and to a lesser extent leaf_count/volume) from
noisy dimensions leaking through the single shared linear layer into terms they have no genuine
relationship to.

**Structural implication for D3/D4 (explicitly flagged by the user as the reason this check
mattered, not just a D2 footnote)**: this is a SHARED-LINEAR-LAYER problem, not a "not enough
data" problem that more regularization alone will fix. A single `Linear(in_dim, 7)` layer lets
any input dimension influence every output term's prediction, regardless of whether that
dimension has any real relationship to that term. D3 (+temporal info) and D4 (+previous growth
stage) will each add more dimensions on the same 72-sample training set, so the same failure
mode — a term-irrelevant addition hijacking an unrelated, previously-healthy term's prediction —
should be EXPECTED to recur unless addressed structurally, not just hoped away with weight
decay. Worth considering for D3/D4: per-term input gating/masking (only expose each term's head
to the input dimensions plausibly relevant to it), or separate small per-term linear heads
instead of one shared `Linear(in_dim, 7)`, rather than continuing to grow one fully-shared input
vector. Not yet implemented — flagged here for the D3 design discussion, not decided unilaterally.

## Files

- `adapters/train_d2_fusion_growth_curves.py` — new training script; imports `TraitFusionHead`,
  `masked_regression_loss`, `run_epoch`, `compute_metrics`, `IOStream`, `build_scan_table`,
  `compute_norm_stats` from `train_d1_fusion_baseline.py` unmodified.
- `results/D2_fusion_growth_curves/run.log` — full training log + final per-split metrics.

## Next

D3 (Block D, F3): add temporal info to the fusion input, same backbone/DA/augmentation/
architecture/loss as D1-D2 — only the fusion input changes.
