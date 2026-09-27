# D3: Fusion + Temporal Info (Block D, F3)

**Status: complete.** Progressive fusion, cumulative per CLAUDE.md's Block D design: D3's input
= D2's input (B3b's 256-dim frozen feature + each plant's 12 growth-curve params) + 1 new
temporal feature — not a replacement of D2's input. Backbone/DA/augmentation/architecture-family/
loss stay identical across D1-D3 per the isolation rule; only the fusion input changes.

## Design decision (flagged as inference, not spec text)

The design docs say "F3: + temporal info" without defining it. Used `elapsed_days` — days since
that plant's own first scan, already computed in D0's `assemble_trait_timeseries.py` and carried
through `compute_growth_rate_targets.py`'s output — the simplest, most literal reading of
"temporal information" for a per-scan feature: how far into this specific plant's own observed
growth trajectory this scan sits. Checked its value range before using it (2026-09-28): [0,20]
for Tomato, [0,12] for Maize, identical range across train/val/test by species, no NaN, no
outliers — unlike D2's growth-curve B/Bg parameters, no clipping safeguard was needed.

## Per-term input relevance — applied, not mechanically copied from D2

D2 established that growth-curve params should be withheld from leaf_area/leaf_count/volume (no
principled relevance — those 3 traits have no growth-curve fit of their own in D0's scope).
`elapsed_days` is different: **every one of the 7 terms has a principled claim on it**, since
every trait changes over the trajectory by definition, and both rate terms are themselves
time-varying within a growth curve (e.g. logistic growth rate peaks mid-trajectory, not
constant). Restricting `elapsed_days` to a subset of terms the way growth-curve params were
restricted would apply the PATTERN (restrict inputs) without the REASONING (restrict only where
there's no principled relevance) — so all 7 terms get `elapsed_days` appended to their existing
(D2-established) input slice. Per-term in_dim: leaf_area/leaf_count/volume = 257 (256 B3b + 1
elapsed_days); height/height_rate/stem_diameter/stem_diameter_rate = 263 (256 B3b + 6 own growth
params + 1 elapsed_days).

## Smoke tests before touching real data

1. **Structural isolation**: confirmed each term's `nn.Linear.in_features` matches the design
   exactly. In eval mode, altering the 12 growth-curve columns by 1000x changed leaf_area's
   output by exactly `0.0`; altering `elapsed_days` changed leaf_area's output by `69.9997` and
   height's by `51.4513` — confirms the intended split (growth-curve params isolated from
   leaf_area/leaf_count/volume, elapsed_days universally connected) holds exactly.
2. **Real-data join check**: `build_scan_table_d3` produces 72/18/63 rows matching D1/D2 exactly,
   `elapsed_days` present with no NaN.
3. **Synthetic training-loop + ablation**: gave leaf_area a synthetic true signal depending on
   B3b dims AND `elapsed_days` (mirroring the real per-term relevance design) with zero true
   relationship to the growth-curve columns. Trained `PerTermFusionHeadD3`: train R²=0.949
   (learns the injected signal). Ablation on held-out synthetic data: zeroing growth-curve
   columns left leaf_area's val R² exactly unchanged (0.0265 → 0.0265); zeroing `elapsed_days`
   degraded it substantially (0.0265 → −0.3789) — confirms the model genuinely uses
   `elapsed_days` where it's given access, and remains exactly isolated from growth-curve columns
   where it isn't.

## Real-data result (2026-09-28)

Same 72/18/63 Oracle split, `wd=1e-2`, `dropout=0.3`, 200 epochs, best epoch 169 (val total loss
0.1118). `PerTermFusionHeadD3` parameter count: 1,830.

**Per-term test R² — D1, D2 (corrected architecture), D3:**

| Term | D1 | D2 (corrected) | D3 |
|---|---|---|---|
| height | 0.8802 | 0.8891 | **0.8902** |
| height_rate | 0.7165 | **0.7567** | 0.7438 |
| stem_diameter | 0.1438 | 0.1377 | **0.1501** |
| stem_diameter_rate | 0.1464 | 0.1954 | **0.2121** |
| leaf_area | 0.5952 | 0.5703 | **0.5956** |
| leaf_count | 0.3439 | **0.3540** | 0.3482 |
| volume | 0.0618 | 0.0178 | **0.1571** |

Full RMSE/MAE/R² in `results/D3_fusion_temporal/run.log`.

**D3 is best-of-three on 5 of 7 terms** (height, stem_diameter, stem_diameter_rate, leaf_area,
volume), with height_rate and leaf_count landing marginally below D2 but still above D1. No term
regressed below D1's own baseline — a cleaner across-the-board result than D2's (which had one
real regression before its architecture fix, and even after the fix wasn't uniformly better than
D1 on every term).

## The Maize-volume question, checked directly (per explicit user instruction not to expect a fix)

Volume's test R² jumped notably (0.0178→0.1571), so this was checked by species before writing
it up, rather than assumed to be Maize-driven or dismissed as unrelated:

| | D1 | D2 (corrected) | D3 |
|---|---|---|---|
| Maize volume test R² | −0.270 | −0.337 | **−0.132** |
| Tomato volume test R² | 0.674 | 0.674 | **0.690** |
| Maize share of test SSE | 88.6% | ~88% | **87.9%** |

**Real, partial improvement — not a fix, exactly as anticipated going in.** Maize test R² moves
in the right direction for the first time since D1 (−0.270→−0.337→−0.132), but stays deeply
negative and still accounts for essentially the same ~88% share of total test error it always
has. Tomato also improves modestly. This reads as `elapsed_days` giving the model a genuine,
useful prior on "roughly what growth stage this plant should be at" that helps generically across
both species, rather than anything that resolves the underlying few-shot problem (only 4 Maize
plants total in train+val, 2 held out) — consistent with the user's framing of this as a known,
separately-caused limitation that D3 was never expected to resolve.

## Files

- `adapters/train_d3_fusion_temporal.py` — training script. Defines `build_scan_table_d3`
  (extends D2's table with `elapsed_days`), `compute_temporal_norm_stats`, `D3FusionDataset`
  (269-dim feat: 256 B3b + 12 growth-curve params + 1 elapsed_days), `PerTermFusionHeadD3`
  (extends D2's `PerTermFusionHead` with elapsed_days universally included). Reuses D1's
  `IOStream`/`ALL_TERMS`/`TRAIT_TERMS`/`build_scan_table`/`compute_norm_stats`/
  `masked_regression_loss`/`run_epoch`/`compute_metrics` and D2's `build_scan_table_d2`/
  `compute_feature_norm_stats`/`GROWTH_PARAM_COLS`/`CLIP_STD`/`_GROWTH_RELEVANT_SLICE` unmodified.
- `results/D3_fusion_temporal/run.log` — full training log + final per-split metrics.

## Next

D4 (Block D, F4, "full model", complete — see `step_notes/D4_Fusion_Previous_Stage.md`): added
previous growth stage (each term's own lag-1 trait value) to the fusion input. Required excluding
~15-18% of scans lacking a trustworthy previous stage, so D4's numbers were compared against D3
on the SAME reduced test subset (not the naive full-63 D1-D3 numbers) before drawing conclusions
— on that honest comparison, D4 shows large real gains on height/leaf_area/leaf_count/volume and
real regressions on stem_diameter/height_rate/stem_diameter_rate. Maize-volume continues its
gradual, still-unresolved improvement trend (test R² −0.132→−0.043), a third consecutive row
without a real fix, consistent with it being a data-scarcity limitation. D1-D4 (the fusion-input
progression) are now complete; D5/D6 add the physics-informed growth constraint next.
