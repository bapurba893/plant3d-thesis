# D4: Fusion + Previous Growth Stage (Block D, F4 — "full model")

**Status: complete.** Last of D1-D4's progressive fusion-input additions per CLAUDE.md (D5/D6 add
the physics-informed loss on top of this, not a further fusion-input change). Cumulative: D4's
input = D3's input (B3b's 256-dim feature + 12 growth-curve params + elapsed_days) + 5 new
previous-growth-stage features, one per raw trait.

## Design decisions (flagged as inferences, not spec text)

**"Previous growth stage" = that plant's chronologically-immediately-preceding scan's 5 raw
trait values** (height, stem_diameter, leaf_area, leaf_count, volume) — the literal reading of
"stage" as a measured STATE, not a rate (already D1's L_growth targets, not duplicated here) and
not a curve summary (D2's job).

**Missing-previous-stage handling, checked against real data first**: a plant's first scan has no
previous scan (14/223 scans project-wide); separately a previous scan can itself have a NaN trait
(8/223) or be `stem_leaf_boundary_low_confidence` flagged (5/223). Rather than fabricate a value
for a scan that never had a trustworthy one (the same category of mistake D2's first run made
with an unconstrained curve-fit parameter) or invent a new per-term input-masking mechanism
needing its own separate verification (the reasoning that ruled out learned gating during D2's
architecture fix), rows are EXCLUDED entirely when their previous scan doesn't exist or isn't
fully trustworthy — matching this project's established preference (D0 excluded low-confidence
scans from curve-fitting rather than imputing over them). **Real, reported cost**: train 72→59
(18% dropped), val 18→15 (17%), test 63→54 (14%).

Checked the 5 raw traits' value ranges (train+val vs. test) before assuming clipping was needed
like D2's growth-curve params: full overlap, 0 outliers for all 5 traits — no clipping safeguard
needed (real measured quantities, not an unconstrained curve-fit parameter).

## Per-term relevance, applied at finer granularity than D3

D3 gave `elapsed_days` to ALL 7 terms since every trait/rate is genuinely time-dependent. D4's
previous-stage features are different: the single most direct, well-justified autoregressive
relationship is a trait's OWN lag-1 value (current_height is most directly informed by
previous_height, not previous_leaf_count) — a standard panel-data assumption, and the minimal
reading absent a specified cross-trait design in the docs. Each term gets access to ONLY its own
trait's previous value: height/height_rate → prev_height; stem_diameter/stem_diameter_rate →
prev_stem_diameter; leaf_area → prev_leaf_area; leaf_count → prev_leaf_count; volume →
prev_volume. Every term's in_dim grows by exactly +1 relative to D3 (not a blanket +5).

## Smoke tests before touching real data

1. **Cross-trait structural isolation** (static forward pass, eval mode): altering leaf_area's
   OWN previous value changed its output (`7.90`, nonzero as expected); altering HEIGHT's
   previous value changed leaf_area's output by exactly `0.0` — confirms each term only ever sees
   its own trait's lag, not the other 4.
2. **Real-data join + exclusion check**: `build_scan_table_d4`'s train-split output matched the
   manually pre-computed exclusion count exactly (72→59, 8 first-scans + 5 bad-previous excluded).
3. **Synthetic training-loop + ablation**: gave leaf_area a synthetic signal depending on B3b
   dims + elapsed_days + its OWN prev_leaf_area (mirroring the real design). Trained
   `PerTermFusionHeadD4`: train R²=0.961. Post-training ablation on held-out synthetic data:
   zeroing growth-curve columns and zeroing HEIGHT's previous value both left leaf_area's val R²
   exactly unchanged (0.2742→0.2742 both times); zeroing leaf_area's OWN previous value degraded
   it substantially (0.2742→−0.1839) — confirms the model genuinely uses its own-trait lag where
   given access, and stays exactly isolated from both growth-curve columns and other traits' lags.

## Real-data result (2026-09-28)

Same architecture conventions as D1-D3 (`wd=1e-2`, `dropout=0.3`, 200 epochs), but train=59,
val=15, test=54 (see exclusion note above). Best epoch 188 (val total loss 0.9207 — notably
higher/noisier than D3's 0.1118, expected given val dropped from 18→15 samples).
`PerTermFusionHeadD4` parameter count: 1,837.

### The comparison that matters: apples-to-apples on the SAME 54 test scans

D4 evaluates on 54 test scans, not D1-D3's 63 — a naive D1/D2/D3-vs-D4 comparison would be
confounded by evaluating on different data, so D3's own trained model (unmodified, its own norm
stats) was re-evaluated restricted to exactly the 54 scans D4 also uses, before drawing any
conclusion:

| Term | D3 (all 63) | D3 (same 54 as D4) | D4 (54) |
|---|---|---|---|
| height | 0.8902 | 0.8838 | **0.9114** |
| stem_diameter | 0.1501 | 0.1462 | 0.1161 |
| leaf_area | 0.5956 | **0.5047** | **0.7382** |
| leaf_count | 0.3482 | 0.3684 | **0.4197** |
| volume | 0.1571 | **0.0622** | **0.2256** |
| height_rate | 0.7438 | 0.7494 | 0.6823 |
| stem_diameter_rate | 0.2121 | 0.2037 | 0.1416 |

**The 54-scan subset is not "easier" — for leaf_area and volume it's measurably HARDER** (D3's
own R² on this subset is lower than on the full 63: leaf_area 0.5047 vs. 0.5956, volume 0.0622
vs. 0.1571). This means D4's raw numbers, if compared naively against D1-D3's full-63 numbers,
would have UNDERSTATED its real improvement on those two terms, not overstated it — the opposite
of the naive worry going in. On the true same-subset comparison: **D4 shows large, genuine gains
on height/leaf_area/leaf_count/volume**, and **genuine regressions on stem_diameter/height_rate/
stem_diameter_rate** (all three of which involve the noisier finite-difference/segmentation-
dependent traits — plausibly the previous-stage feature adds real signal for the traits with
clean measurements and adds noise for the ones D0/D1/D2 already established are measurement-
limited, though not verified by a further ablation here).

Full RMSE/MAE/R² in `results/D4_fusion_previous_stage/run.log`.

## The Maize-volume standing diagnostic, checked as instructed (not expected to be fixed by D4)

| | D1 | D2 (corrected) | D3 | D4 |
|---|---|---|---|---|
| Maize volume test R² | −0.270 | −0.337 | −0.132 | **−0.043** |
| Tomato volume test R² | 0.674 | 0.674 | 0.690 | 0.750 |
| Maize share of test SSE | 88.6% | ~88% | 87.9% | **91.2%** |

**Still broken, continuing the same gradual-improvement trend seen since D2, now for the third
consecutive row.** Maize test R² keeps closing toward zero (−0.337→−0.132→−0.043) without ever
becoming genuinely useful, while Maize's SHARE of total test error actually ROSE slightly to
91.2% (since Tomato improved more in absolute terms this row) — the underlying few-shot problem
(4 Maize plants total in train+val, 2 held out) remains unresolved by any fusion-input addition
tried so far, exactly consistent with it being a data-scarcity limitation rather than a
missing-feature one. Not something D4 was expected to fix, and it wasn't — flagged again for
whoever picks up D5/D6 or future work, since no amount of additional fusion input appears likely
to resolve a 2-plant held-out-species problem.

## Files

- `adapters/train_d4_fusion_previous_stage.py` — training script. Defines `build_scan_table_d4`
  (extends D3's table with previous-stage lookup + exclusion), `compute_prev_stage_norm_stats`,
  `D4FusionDataset` (274-dim feat), `PerTermFusionHeadD4` (extends D3's head with each term's own
  lag-1 trait value). Reuses D1's `IOStream`/`ALL_TERMS`/`TRAIT_TERMS`/`build_scan_table`/
  `compute_norm_stats`/`masked_regression_loss`/`run_epoch`/`compute_metrics`, D2's
  `compute_feature_norm_stats`/`GROWTH_PARAM_COLS`/`CLIP_STD`/`_GROWTH_RELEVANT_SLICE`, and D3's
  `build_scan_table_d3`/`compute_temporal_norm_stats`/`_TEMPORAL_COL_IDX` unmodified.
- `results/D4_fusion_previous_stage/run.log` — full training log + final per-split metrics.

## Next

D1-D4 (the progressive fusion-input sequence) are now complete. D5/D6 add the physics-informed
growth constraint (Logistic/Gompertz `L_phys` + `L_mono`, per CLAUDE.md — `L_mono` applies only
to height and stem_diameter) on top of D4's full fusion input — same backbone/DA/augmentation,
same per-term relevance discipline established across D2-D4 should carry forward to any new
physics-loss-specific inputs if needed.
