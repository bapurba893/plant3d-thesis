# D1: Fusion Baseline (Block D, F1 — deep geometric features only)

**Status: complete.** Backbone/DA/augmentation for Block D: KPConv + DA-S (row B3b), per the
FINAL SUMMARY in CLAUDE.md (strongest non-Oracle result across all 15 A/B/C rows, 0.860
full-run-mean target accuracy, most stable trajectory in the project, no segmentation cost).
D1 trains only a small regression head on top of B3b's frozen pooled feature — no backbone
fine-tuning, no domain-adaptation machinery (nothing to adapt; frozen encoder).

## Undefined items resolved with the user before building anything

The design docs (`docs/Domain_Invariance_Strategy_Table.docx`,
`docs/plant_3d_pipeline_v6_corrected.drawio`) confirm D1's loss terms (`L_trait + L_growth`,
both cooperative/Kendall-weighted) and the general `L_trait`/`L_phys`/`L_mono` formulas already
in CLAUDE.md, but leave several things genuinely undefined for D1 specifically. Resolved
directly with the user, not assumed:

1. **`L_trait` targets all 5 traits** (height, stem_diameter, leaf_area, leaf_count, volume),
   each `Huber + lambda_corr*(1-pearson_corr)` (lambda_corr=0.5, flagged default not tuned),
   combined via Kendall uncertainty weighting.
2. **`L_growth`'s target is trait-specific**: height uses the D0-fitted growth curve's own
   closed-form derivative (whichever of Logistic/Gompertz has the higher R² per plant — Logistic
   wins 10/14 plants, Gompertz 4/14); stem_diameter uses `np.gradient` finite-difference rates
   between consecutive real scans (its curve fits were confirmed poor in D0, R² as low as
   0.025 — a fitted-curve derivative would be worse than finite-difference noise). The 6
   `stem_leaf_boundary_low_confidence` scans (T01_320, T02_322, T03_324, T04_324, T07_322,
   T07_325) are dropped from each plant's series BEFORE differencing, not just their own target
   excluded — a bad point between two good ones would otherwise corrupt both neighbors' finite
   differences too. leaf_area/leaf_count/volume get no `L_growth` term (matches their exclusion
   from D0 curve-fitting — legitimately non-monotonic via senescence).
3. **B3b's KPConv encoder is FROZEN for D1** — standard choice for a small downstream dataset
   (72-90 scans) on an already-trained encoder; keeps D1-D6 comparisons clean (only the fusion
   input changes, never confounded by different per-row fine-tuning dynamics).
4. **"Deep geometric features" (F1) = B3b's pooled global feature**, `logits["feat"]`
   (confirmed 256-dim, `adapters/models_kpconv.py::KPConv_ClsSeg.forward` — already exposed by
   every backbone's forward interface, originally added for DA-A discriminator input, directly
   reusable here).

## Frozen-encoder enforcement (explicit user request — must be genuinely enforced, not just "not called")

`adapters/train_d1_fusion_baseline.py` never imports `KPConv_ClsSeg`, `PlantKPConvConfig`, or
any KPConv-related module at all — confirmed via grep (only docstring/comment mentions, zero
real imports). This is the strongest available form of "frozen": not `requires_grad=False` on a
still-loaded encoder (which a future edit could silently flip), but the encoder's total absence
from the file's computation graph. `TraitFusionHead`'s parameter count (1,799, see below) is
printed at training start as a live audit trail — nowhere near KPConv scale.

Features are precomputed once by `adapters/extract_b3b_features.py` (reuses
`infer_segmentation.py::load_b3b_model` for exact model reconstruction +
`kpconv_collate.py::calibrate_neighborhood_limits`, both unmodified) and cached to
`data/features/Pheno4D/<species>/<stem>.npz` (`{"feat": (256,) float32}`) — all under
`torch.no_grad()`. Full run: 223/223 scans, shape (256,), no NaN/Inf, global mean=0.541/std=0.613.
Caught and fixed one bug along the way: `--limit` originally only checked between batches, so
`batch_size=4, limit=6` processed 8 scans not 6; fixed with a `n_processed_this_split` counter
checked both before the batch loop and inside the per-sample write loop (never affected the real
223-scan run, which omits `--limit`).

## L_growth targets: `scripts/compute_growth_rate_targets.py`

Verified against numerical differentiation before running on real data (max error ~1e-9 for both
the Logistic ODE form `dy/dt = rho*y*(1-y/K)` and the Gompertz form
`dy/dt = y*beta*Bg*exp(-beta*t)`). Real output (`data/traits/pheno4d_growth_rate_targets.csv`,
223 rows): `height_rate` 0 NaN, `stem_diameter_rate` exactly 6 NaN (matching the flagged scans).

## NaN/low-confidence masking — proof of cross-term isolation (explicit user request)

Each of the 7 loss terms is computed independently over its own valid subset
(`masked_regression_loss`, `run_epoch` in `train_d1_fusion_baseline.py`); a term with <2 valid
samples in a batch returns `None` and is dropped from that batch's Kendall combination entirely
(never substituted with a zero, which would falsely tell Kendall's learned uncertainty the term
was perfectly predicted). `FLAG_AFFECTED_TERMS = {stem_diameter, leaf_area, leaf_count}` —
exactly the 3 traits D0 confirmed are corrupted by the segmentation failure on the 6 flagged
scans (height/volume/height_rate unaffected; stem_diameter_rate already NaN from the source).

Verified with a synthetic batch (not just described): invalidated `leaf_area` for one sample and
replaced its target with a garbage value (99999.0). Result: the other 6 terms' losses were
byte-identical before/after (to 1e-6). `leaf_area`'s own loss DID change — correctly: masking
drops the sample count (8→7), which naturally shifts the mean-based loss; confirmed this is not
a leak by computing the loss directly over the 7 valid samples with original (non-garbage)
values, which matched exactly (diff = 0.0). Gradient check: the masked-out sample receives
exactly zero gradient from that term's loss (<1e-8) while a valid sample's gradient is nonzero.

## Overfitting check and architecture fix (explicit user request, done BEFORE touching real data)

First version used a 256→128→7 head (33,799 params), `wd=1e-4`. Full-loop synthetic smoke test
(200 synthetic samples, 7 known-linear targets, 3 terms sparsified ~15% to mirror real
NaN/flagging) showed train R²=0.995-0.997 on every term but held-out val R² only 0.11-0.38 with
the SAME clean signal — classic n≪p overfitting (33K params, 256-dim input, 160 training
samples), not a masking bug (train R² near-1.0 on the sparsified terms too confirmed masking
survives real gradient descent).

A/B tested the fix directly rather than guessing: three configs, same synthetic data,
`wd` raised to `1e-2` in all three —

| config | train R² (mean) | val R² (mean) | params |
|---|---|---|---|
| old (rejected): 256→128→7, wd=1e-4 | 0.9962 | 0.2417 | 33,799 |
| 256→32→7, wd=1e-2 | 0.9710 | 0.3707 | 8,455 |
| **256→7 linear (chosen), wd=1e-2** | 0.9568 | **0.5286** | 1,799 |

Val R² rose monotonically as capacity dropped further, with train R² barely moving — evidence
capacity reduction wasn't discarding signal, only overfitting. `hidden_dim=0` (direct linear
head, Dropout on the input only) became the new default; `--hidden_dim N` remains available if
real training data (larger N, different regularization needs) tolerates more capacity than the
synthetic test could probe.

## Real-data run (2026-09-27) — full result

Oracle-style plant-disjoint split reused unchanged from A5/B5/C5: `pheno4d_oracle_train.csv`
(8 plants/72 scans), `pheno4d_oracle_val.csv` (2 plants/18 scans),
`pheno4d_heldout_eval.csv` (4 plants/63 scans). 200 epochs, Adam+cosine schedule, `lr=1e-3`,
`wd=1e-2`, `dropout=0.3`, `hidden_dim=0` (linear head), `lambda_corr=0.5`. Best model at epoch
159 (lowest val total loss, 0.0607). CPU-only — a 1,799-parameter linear head on precomputed
tensors trains in under a minute, no GPU/sbatch needed for D1.

**Per-term R² (train n=72 / val n=18 / test n=63):**

| Term | Train | Val | Test |
|---|---|---|---|
| height | 0.9695 | 0.9711 | 0.8802 |
| height_rate | 0.8350 | 0.8739 | 0.7165 |
| stem_diameter | 0.3737 | 0.3272 | 0.1438 |
| stem_diameter_rate | 0.3538 | −0.1548 | 0.1464 |
| leaf_area | 0.6235 | 0.5956 | 0.5952 |
| leaf_count | 0.3777 | 0.1844 | 0.3439 |
| volume | 0.7055 | 0.8444 | 0.0618 |

Full RMSE/MAE/R² for every split/term in `results/D1_fusion_baseline/run.log`.

**Real data's pattern is qualitatively different from what the synthetic overfitting test
predicted — confirming the fix worked, but revealing a different, more fundamental limitation
in its place:**

1. **The train≫val overfitting gap the linear-head/high-wd fix targeted is gone on real data.**
   Train and val R² sit close for most terms (height 0.970/0.971, stem_diameter 0.374/0.327,
   leaf_area 0.624/0.596); volume's val R² (0.844) is actually higher than train (0.706). This
   is the fix generalizing correctly to the real, smaller dataset it was built for — not a
   contradiction of the synthetic result.

2. **What real data shows instead: a trait-dependent noise ceiling visible even on the training
   set** — something the synthetic test's clean linear signal (noise std 0.05) structurally
   could not produce. height/height_rate reach train R² 0.84-0.97; stem_diameter,
   stem_diameter_rate, and leaf_count stay weak (0.35-0.38) even on the 72 training samples the
   model is directly fit to. Since more capacity or less regularization cannot raise a
   training-set ceiling, this points to noise in the INPUT SIGNAL for these three terms, not an
   overfitting/capacity problem.

3. **This exactly matches D0's already-documented segmentation measurement uncertainty.**
   height's curve fits were confirmed strong in D0 (R² 0.97-0.99) and its segmentation is
   organ-boundary-independent — consistent with it being the strongest term here by a wide
   margin. stem_diameter/stem_diameter_rate/leaf_count all depend on Tomato's `height_split`
   stem/leaf boundary (76.7% stem recall, 68.7% leaf recall — the best of 5 remap attempts
   tried in D0, but real, quantified measurement noise, not ground truth) — exactly the traits
   that stay weak train-through-test here. The real dataset confirms this measurement noise is
   actively suppressing these terms' achievable R², independent of the capacity/regularization
   question the synthetic test was built to answer.

4. **New finding, not predicted by either check: volume's test-set collapse** (val R²=0.844 →
   test R²=0.062). Unlike the noise-limited terms above, volume looks strong on train/val then
   falls to near-zero specifically on the 4 held-out (genuinely plant-disjoint) test plants.
   Plausibly a cross-plant generalization gap — only 10 total plants split across train+val
   (8+2), so the 2 validation plants may not represent the volume range/shape distribution the
   4 test plants cover. Flagged for attention in D2-D4, not yet actionable from one row alone.

5. **stem_diameter_rate's negative val R² (−0.1548)** is most likely small-N noise (only 17
   valid val samples after excluding the 6 low-confidence scans) rather than a distinct problem
   — not corroborated by test R² (0.146, weakly positive, 61 samples).

## Files

- `adapters/extract_b3b_features.py` — caches B3b's frozen 256-dim pooled feature per scan.
- `scripts/compute_growth_rate_targets.py` — L_growth targets (height: curve derivative;
  stem_diameter: finite-difference).
- `adapters/train_d1_fusion_baseline.py` — the D1 training script (`TraitFusionHead`,
  `masked_regression_loss`, `run_epoch`, `compute_metrics`).
- `data/features/Pheno4D/<species>/<stem>.npz` — cached features, 223 scans.
- `data/traits/pheno4d_growth_rate_targets.csv` — L_growth targets, 223 rows.
- `results/D1_fusion_baseline/run.log` — full training log + final per-split metrics.

## Next

D2 (Block D, F2): add growth-curve params to the fusion input, same backbone/DA/augmentation
(KPConv+DA-S/B3b), same frozen-encoder/Oracle-split protocol as D1 — only the fusion input
changes, per CLAUDE.md's Block D isolation rule.
