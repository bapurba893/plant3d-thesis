# Full-workflow rerun (FW) on P4a: Block D (D1–D6 and the λ=0 control) results

Scope: the third part of the full-workflow rerun described in `jobs/fw/README.md`. Block D (D1–D6,
plus the no-physics control of D5 and of D6) was retrained with seeds 1, 2, 3 on **features from
the FW B3b backbone** (KPConv, DA-S, ALL, trained on the soil-kept preprocessing P4a,
`step_notes/FW_2_Results.md`). Block D seed `<s>` uses the backbone `results/FW_B3b_s<s>/model.pt`.
The A–C rows are in `FW_1_Results.md` (PointNet++, DGCNN) and `FW_2_Results.md` (KPConv).

What changed against the original Block D: **only the input features**. Each seed's frozen
256-number pooled B3b feature was re-extracted for all 223 Pheno4D scans through the P4a manifest
into `data/features_fw/P4a_B3b_s<s>/`. The trait and growth **targets were kept fixed**: every D
script still reads the original `data/traits/*.csv` (derived from the original B3b segmentation;
`infer_segmentation.py` was not rerun). D arguments are the originals (all defaults), plus
`--seed <s>` and `--features_dir`. The controls add `--lambda_phys 0 --lambda_mono 0`. D5, D6 and
both controls warm-start from the same seed's `results/FW_D4_s<s>/model.pt`, as the originals
warm-started from `results/D4_fusion_previous_stage/model.pt`.

Every number below was parsed from the `=== FINAL test metrics` block of
`results/FW_<row>_s<seed>/run.log` (and, for the original, `results/<original row>/run.log`), by a
throwaway parser that was not committed. Metric: **test R²** per term (Pheno4D held-out plants,
best epoch chosen by lowest val total loss; test labels never used for selection). SD = sample SD
over 3 seeds.

Test set size per term (identical for every seed and for the original):

| Rows | height | stem_diameter | leaf_area | leaf_count | volume | height_rate | stem_diameter_rate |
|---|---|---|---|---|---|---|---|
| D1–D3 | 63 | 61 | 58 | 58 | 63 | 63 | 61 |
| D4–D6, control | 54 | 52 | 51 | 51 | 54 | 54 | 52 |

D4–D6 drop every scan whose previous scan is missing or not trustworthy (train 72→59, val 18→15,
test 63→54; `adapters/train_d4_fusion_previous_stage.py` docstring).

## Jobs

| Job | What | Outcome |
|---|---|---|
| 359157 (array, tasks 1–3 = seeds 1–3, `jobs/fw/fw_D.sbatch`) | per task: feature extraction (GPU), then D1, D2, D3, D4, D5, D5 control, D6, D6 control (CPU) | every step `OK` in `results/logs/fw_D_359157_<task>.log`; tasks took 32, 19 and 36 min (2026-10-21) |

Feature extraction wrote 223 `.npz` files per seed (`feat`, shape (256,)); KPConv calibrated
`neighborhood_limits` [499, 35, 30] (seed 1), [499, 34, 30] (seeds 2, 3).

Results: `results/FW_D1_s<s>` … `FW_D6_s<s>`, controls `FW_D5_ablation_s<s>`, `FW_D6_ablation_s<s>`.
The original Block D inputs (`data/features/Pheno4D/`, `data/traits/`) and results (`results/D*/`)
were not written (no git diff).

**The D5 control and the D6 control are identical** term for term in every seed (and in the
original). Expected: with `lambda_phys = lambda_mono = 0` the two scripts train the same model from
the same warm start. They are reported once below as "control (λ=0)".

---

## 1. Per-term test R² (P4a features, 3 seeds)

Columns: FW mean ± SD | seeds 1 / 2 / 3 | original (single seed 1, original B3b features) | FW mean −
original. Best epochs (s1 / s2 / s3; original): D1 160/160/168 (159); D2 163/163/144 (164); D3
163/153/166 (169); D4 158/171/163 (188); D5 199/188/138 (172); D6 196/188/138 (172); control
148/167/138 (172).

### D1 (deep features only)

| Term | Mean ± SD | s1 / s2 / s3 | Original | Diff |
|---|---|---|---|---|
| height | 0.8853 ± 0.0309 | 0.8505 / 0.8961 / 0.9094 | 0.8802 | +0.0051 |
| stem_diameter | 0.1153 ± 0.0188 | 0.1141 / 0.1347 / 0.0972 | 0.1438 | −0.0285 |
| leaf_area | 0.4668 ± 0.1170 | 0.5482 / 0.5195 / 0.3327 | 0.5952 | −0.1284 |
| leaf_count | 0.2872 ± 0.0691 | 0.2497 / 0.2449 / 0.3669 | 0.3439 | −0.0567 |
| volume | 0.1294 ± 0.3351 | −0.0449 / −0.0826 / 0.5157 | 0.0618 | +0.0676 |
| height_rate | 0.6188 ± 0.0651 | 0.6292 / 0.5491 / 0.6780 | 0.7165 | −0.0977 |
| stem_diameter_rate | 0.1572 ± 0.0283 | 0.1894 / 0.1457 / 0.1365 | 0.1464 | +0.0108 |

### D2 (+ growth-curve parameters)

| Term | Mean ± SD | s1 / s2 / s3 | Original | Diff |
|---|---|---|---|---|
| height | 0.8897 ± 0.0284 | 0.8576 / 0.9002 / 0.9114 | 0.8891 | +0.0006 |
| stem_diameter | 0.1106 ± 0.0210 | 0.0972 / 0.1348 / 0.0999 | 0.1377 | −0.0271 |
| leaf_area | 0.4473 ± 0.0957 | 0.5334 / 0.4641 / 0.3443 | 0.5703 | −0.1230 |
| leaf_count | 0.2949 ± 0.0644 | 0.2628 / 0.2529 / 0.3690 | 0.3540 | −0.0591 |
| volume | 0.0961 ± 0.3770 | −0.1154 / −0.1277 / 0.5314 | 0.0178 | +0.0783 |
| height_rate | 0.6453 ± 0.0506 | 0.6590 / 0.5892 / 0.6876 | 0.7567 | −0.1114 |
| stem_diameter_rate | 0.2013 ± 0.0106 | 0.2099 / 0.2044 / 0.1895 | 0.1954 | +0.0059 |

### D3 (+ temporal information)

| Term | Mean ± SD | s1 / s2 / s3 | Original | Diff |
|---|---|---|---|---|
| height | 0.9022 ± 0.0243 | 0.8741 / 0.9153 / 0.9171 | 0.8902 | +0.0120 |
| stem_diameter | 0.1063 ± 0.0172 | 0.0963 / 0.1262 / 0.0964 | 0.1501 | −0.0438 |
| leaf_area | 0.5180 ± 0.0893 | 0.5835 / 0.5541 / 0.4163 | 0.5956 | −0.0776 |
| leaf_count | 0.2999 ± 0.0498 | 0.2652 / 0.2776 / 0.3569 | 0.3482 | −0.0483 |
| volume | 0.1487 ± 0.3166 | −0.0176 / −0.0501 / 0.5138 | 0.1571 | −0.0084 |
| height_rate | 0.6481 ± 0.0642 | 0.6711 / 0.5756 / 0.6976 | 0.7438 | −0.0957 |
| stem_diameter_rate | 0.2062 ± 0.0118 | 0.2177 / 0.2066 / 0.1942 | 0.2121 | −0.0059 |

### D4 (+ previous growth stage; 54-scan test set)

| Term | Mean ± SD | s1 / s2 / s3 | Original | Diff |
|---|---|---|---|---|
| height | 0.9155 ± 0.0162 | 0.8973 / 0.9211 / 0.9282 | 0.9114 | +0.0041 |
| stem_diameter | 0.0975 ± 0.0187 | 0.0801 / 0.0953 / 0.1172 | 0.1161 | −0.0186 |
| leaf_area | 0.7161 ± 0.0478 | 0.7239 / 0.7595 / 0.6648 | 0.7382 | −0.0221 |
| leaf_count | 0.3885 ± 0.0448 | 0.3462 / 0.3838 / 0.4355 | 0.4197 | −0.0312 |
| volume | 0.3352 ± 0.3063 | 0.1165 / 0.2038 / 0.6853 | 0.2256 | +0.1096 |
| height_rate | 0.6390 ± 0.0469 | 0.6455 / 0.5892 / 0.6823 | 0.6823 | −0.0433 |
| stem_diameter_rate | 0.1291 ± 0.0256 | 0.1486 / 0.1386 / 0.1001 | 0.1416 | −0.0125 |

### D5 (D4 + Logistic ODE physics loss; 54-scan test set)

| Term | Mean ± SD | s1 / s2 / s3 | Original | Diff |
|---|---|---|---|---|
| height | 0.9240 ± 0.0132 | 0.9096 / 0.9355 / 0.9268 | 0.9282 | −0.0042 |
| stem_diameter | 0.1024 ± 0.0064 | 0.0966 / 0.1014 / 0.1092 | 0.1385 | −0.0361 |
| leaf_area | 0.7966 ± 0.0334 | 0.8142 / 0.8174 / 0.7581 | 0.8000 | −0.0034 |
| leaf_count | 0.3925 ± 0.0627 | 0.3316 / 0.3892 / 0.4568 | 0.4272 | −0.0347 |
| volume | 0.5405 ± 0.2218 | 0.4067 / 0.4182 / 0.7965 | 0.4501 | +0.0904 |
| height_rate | 0.6365 ± 0.0720 | 0.6606 / 0.5555 / 0.6933 | 0.6855 | −0.0490 |
| stem_diameter_rate | 0.1337 ± 0.0238 | 0.1543 / 0.1391 / 0.1077 | 0.1614 | −0.0277 |

### D6 (D4 + Gompertz ODE physics loss; 54-scan test set)

| Term | Mean ± SD | s1 / s2 / s3 | Original | Diff |
|---|---|---|---|---|
| height | 0.9205 ± 0.0220 | 0.8951 / 0.9339 / 0.9324 | 0.9273 | −0.0068 |
| stem_diameter | 0.1024 ± 0.0065 | 0.0964 / 0.1015 / 0.1093 | 0.1386 | −0.0362 |
| leaf_area | 0.7966 ± 0.0334 | 0.8142 / 0.8174 / 0.7581 | 0.8000 | −0.0034 |
| leaf_count | 0.3925 ± 0.0627 | 0.3316 / 0.3892 / 0.4568 | 0.4272 | −0.0347 |
| volume | 0.5404 ± 0.2218 | 0.4066 / 0.4182 / 0.7965 | 0.4501 | +0.0903 |
| height_rate | 0.6365 ± 0.0720 | 0.6606 / 0.5555 / 0.6933 | 0.6855 | −0.0490 |
| stem_diameter_rate | 0.1337 ± 0.0238 | 0.1543 / 0.1391 / 0.1077 | 0.1614 | −0.0277 |

### Control, λ=0 (D5 control = D6 control; 54-scan test set)

| Term | Mean ± SD | s1 / s2 / s3 | Original | Diff |
|---|---|---|---|---|
| height | 0.9138 ± 0.0311 | 0.8784 / 0.9260 / 0.9369 | 0.9021 | +0.0117 |
| stem_diameter | 0.1018 ± 0.0118 | 0.0914 / 0.0993 / 0.1147 | 0.1418 | −0.0400 |
| leaf_area | 0.7881 ± 0.0262 | 0.8000 / 0.8063 / 0.7581 | 0.8000 | −0.0119 |
| leaf_count | 0.3921 ± 0.0617 | 0.3340 / 0.3854 / 0.4568 | 0.4272 | −0.0351 |
| volume | 0.5542 ± 0.2099 | 0.4383 / 0.4278 / 0.7965 | 0.4501 | +0.1041 |
| height_rate | 0.6411 ± 0.0713 | 0.6702 / 0.5599 / 0.6933 | 0.6855 | −0.0444 |
| stem_diameter_rate | 0.1332 ± 0.0226 | 0.1510 / 0.1408 / 0.1077 | 0.1614 | −0.0282 |

---

## 2. All-3-seeds rule

Rule (as Block PP, `FW_1_Results.md`, `FW_2_Results.md`): a variant beats a reference on a term only
if **all 3** of its seeds exceed the reference's 3-seed mean. Cells = number of variant seeds above
the reference mean; **up** = 3/3 above, **down** = 3/3 below. Per-seed values are in section 1.

| Comparison | height | stem_diameter | leaf_area | leaf_count | volume | height_rate | stem_diameter_rate |
|---|---|---|---|---|---|---|---|
| D2 vs D1 | 2/3 | 1/3 | 1/3 | 1/3 | 1/3 | 2/3 | 3/3 **up** |
| D3 vs D2 | 2/3 | 1/3 | 2/3 | 1/3 | 1/3 | 2/3 | 2/3 |
| D4 vs D3 (54 vs 63 scans, see 3a) | 2/3 | 1/3 | 3/3 **up** | 3/3 **up** | 2/3 | 1/3 | 0/3 **down** |
| D5 vs D4 | 2/3 | 2/3 | 3/3 **up** | 2/3 | 3/3 **up** | 2/3 | 2/3 |
| D6 vs D4 | 2/3 | 2/3 | 3/3 **up** | 2/3 | 3/3 **up** | 2/3 | 2/3 |
| Control vs D4 | 2/3 | 2/3 | 3/3 **up** | 1/3 | 3/3 **up** | 2/3 | 2/3 |
| D5 vs control | 2/3 | 1/3 | 2/3 | 1/3 | 1/3 | 2/3 | 2/3 |
| D6 vs control | 2/3 | 1/3 | 2/3 | 1/3 | 1/3 | 2/3 | 2/3 |

Paired per-seed difference against the control, on the two terms the physics/monotonicity loss acts
on (s1 / s2 / s3; original single seed in brackets):

| | height | stem_diameter |
|---|---|---|
| D5 − control | +0.0312 / +0.0095 / −0.0101 (orig +0.0261) | +0.0052 / +0.0021 / −0.0055 (orig −0.0033) |
| D6 − control | +0.0167 / +0.0079 / −0.0045 (orig +0.0252) | +0.0050 / +0.0022 / −0.0054 (orig −0.0032) |

---

## 3. Findings and caveats

Each statement is labelled **[measured]** (read directly from the logs above) or **[inferred]**.

- **No step from D1 to D4 passes the rule on height, stem_diameter, volume or height_rate.**
  [measured] The only 3/3 results are: D2 > D1 on stem_diameter_rate; D4 > D3 on leaf_area and
  leaf_count; D4 < D3 on stem_diameter_rate (3/3 below); D5, D6 and the control > D4 on leaf_area
  and volume.
- **a. D4 vs D3 is not like-for-like.** [measured] D4–D6 are tested on 54 scans (and trained on 59),
  D1–D3 on 63 (trained on 72), because D4 drops scans without a trustworthy previous scan. The D4 vs
  D3 differences (leaf_area and leaf_count up, stem_diameter_rate down) mix the effect of the
  previous-stage input with the change of scan set. [inferred] The two effects have not been
  separated; that would need D3 retrained and tested on the same 54/59-scan subset.
- **b. The gain of D5 and D6 over D4 on leaf_area and volume is not caused by the physics loss.**
  [measured] The λ=0 control also beats D4 on leaf_area and volume on all 3 seeds, and D5/D6 do not
  beat the control on either term (2/3 and 1/3). [inferred, not tested] The likely cause is the
  extra training: D5, D6 and the control all warm-start from D4's checkpoint and train 200 more
  epochs. No run separates the warm start from the extra epochs.
- **No physics-loss benefit passes the rule on any term.** [measured] D5 and D6 vs the control: at
  most 2/3 on every term. On height, D5 is above the control in seeds 1 and 2 and below it in
  seed 3 (+0.0312 / +0.0095 / −0.0101); D6 the same pattern (+0.0167 / +0.0079 / −0.0045). The
  original single-seed result (D5 − control on height = +0.026) is therefore **not confirmed** by the
  rerun: the sign holds in 2 of 3 seeds, not all 3.
- **c. Correction to the earlier "only height differs from the control" statement.** [measured] That
  statement holds only when the physics run and the control select the same best epoch. That happens
  in seed 3 (both epoch 138), as in the original (both 172): there, all terms except height and
  stem_diameter are identical to 4 decimals. (Strictly, stem_diameter differs too, in seed 3:
  0.1092 vs 0.1147 for D5; it also differed in the original, 0.1385 vs 0.1418. Those are the two
  terms the physics and monotonicity losses act on.) In seeds 1 and 2 the best epochs differ (D5
  199 and 188, D6 196 and 188, control 148 and 167), so **every term differs** from the control,
  including the five terms the physics loss cannot reach. [inferred] Those differences come from
  the different selected epoch, not from the physics loss acting on those terms.
- **d. Volume has a large seed spread.** [measured] Volume R² SD is 0.21–0.38 in every row, and
  seed 3 is far above seeds 1 and 2 in every row (by +0.36 to +0.65; e.g. D1 −0.0449 / −0.0826 /
  0.5157, D4 0.1165 / 0.2038 / 0.6853). Volume means and every volume difference above are
  dominated by seed 3. [inferred] The reason is not known (see f).
- **e. stem_diameter R² is about 0.10 in every row.** [measured] Means 0.0975–0.1153 across all
  rows, against 0.12–0.15 in the original runs. [inferred] Consistent with D0's finding that the
  stem_diameter targets are noisy (`step_notes/D0_Trait_Extraction_Pipeline.md`); no D row predicts
  it usefully.
- **f. The Block D seed and the backbone seed move together.** [measured] Block D seed `<s>` uses
  features from `FW_B3b_s<s>`, and B3b seed 3 was the weak backbone seed (full-run mean target
  accuracy 0.7114, against 0.8754 and 0.9079; `FW_2_Results.md`). So the three seeds vary both the
  backbone (and its features) and the D training seed. [inferred] The SDs above measure backbone
  sensitivity as well as training sensitivity. This has **not been separated** (it would need, for
  example, several D seeds on one backbone's features). The weak B3b seed is also the seed with the
  much higher volume R²; whether that is linked is not known.
- **g. The trait targets come from our own segmentation, not hand measurements.** [measured] Every
  target in `data/traits/*.csv` was computed from the original B3b's predicted segmentation
  (`step_notes/D0_Trait_Extraction_Pipeline.md`), and the rerun kept these targets fixed. R² here is
  agreement with those derived traits, not with measured plant traits. [inferred] Errors of the
  original segmentation are part of the targets.
- **Against the original single seed.** [measured] On P4a features, D1–D3 lose on leaf_area
  (−0.08 to −0.13), leaf_count and height_rate (−0.05 to −0.11), and height is within ±0.012 in every
  row. Volume is higher on P4a in most rows, but that comes from seed 3 (see d). As in
  `FW_2_Results.md`, the original is one seed, so a few points' difference can be run-to-run noise.

## 4. Not done yet

- The λ=0 control for other backbones or adaptation conditions (Block D was only run on B3b
  features).
- Extra seeds, including seeds that separate the backbone seed from the Block D training seed
  (see f).
- Block D on the soil-removed target (no Block D run and no FW row has been evaluated on the
  soil-removed target cache).
