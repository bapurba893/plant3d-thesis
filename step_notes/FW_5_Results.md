# Full-workflow rerun (FW) on P4a: Block D with causal growth-curve inputs (D2–D6, control)

Scope: a rerun of Block D rows D2–D6 and the λ=0 control from `FW_3_Results.md`, after fixing a
within-plant leak in the growth-curve inputs. Everything else matches FW_3: same P4a B3b features
(`data/features_fw/P4a_B3b_s<s>/`, not re-extracted), same seeds 1, 2, 3, same split, same
arguments, same fixed trait and growth targets. D1 has no growth-curve input and was not rerun; its
FW_3 numbers are reused below.

## 0. The leak and the fix

**Leak.** [measured, from the code] In FW_3 (and every earlier Block D run), D2–D6 received each
plant's growth-curve parameters from `data/traits/pheno4d_growth_curves.csv`. That file holds one
Logistic and one Gompertz fit per plant and trait (height, stem_diameter), fitted on the plant's
**whole** trajectory. The same 12 numbers were given to every scan of the plant, held-out test
plants included. A test scan's inputs were therefore fitted on its own trait value and on that
plant's later scans (for example, the Logistic K is roughly the plant's final height). The plant
split stayed disjoint, so nothing leaked across plants. The leak was within each plant.
A second, smaller leak: D5/D6 initialised their trainable ODE parameters (ρ, K / α, β) from the
mean of those fits over all 14 plants, val and test included.

**Fix** (code changes uncommitted at the time of writing):

- `scripts/growth_curves.py --causal` (`fit_all_causal`) writes
  `data/traits/pheno4d_growth_curves_causal.csv`: for every scan and trait, Logistic and Gompertz
  are fitted on that plant's scans **strictly earlier** than the scan (`elapsed_days < t`). D0's
  rules are unchanged: stem_diameter skips low-confidence scans, and a fit needs at least 4 points.
- D2 (`build_growth_feature_table`, `build_scan_table_d2`) merges these per scan, for train, val
  and test alike. If a scan has fewer than 4 earlier points, or either family fails to converge,
  that trait's 6 parameters are the **per-species mean of the 8 training plants'**
  whole-trajectory fits, and a new 0/1 `<trait>_fit_available` column is 0. Each trait's block is
  now 7 columns, 14 in total. Input widths: D2 270 (was 268), D3 271 (269), D4 276 (274). The
  growth-aware heads (height, stem_diameter and their rates) went from 262/263/264 to
  263/264/265 inputs in D2/D3/D4. The leaf_area, leaf_count and volume heads still never see
  growth columns.
- D3 and D4 take the shared layout from D2 (`growth_feature_vector`, `_N_GROWTH_TOTAL`). Before the
  fix, `_N_GROWTH_PER_TRAIT` and `_TEMPORAL_COL_IDX` were computed from the parameter count and
  would have kept their old values (6 and 268) after adding the indicators, so D3 and D4 would
  have read the wrong columns without any error. They are now computed from the new 7-column
  block.
- D5/D6 `load_d0_population_init` averages the 8 training plants only (M01, M02, M05, M06, T02,
  T04, T05, T06). New initial values (old 14-plant values in brackets): Logistic height
  ρ 0.231 (0.225), K 488.8 (499.7); stem_diameter ρ 0.226 (0.253), K 138.8 (201.2). Gompertz
  height α 0.742 (0.714), β 0.121 (0.115); stem_diameter α 0.490 (0.510), β 0.150 (0.128).
- `data/traits/pheno4d_growth_curves.csv` is **unchanged**. It is still the D0 record, the source
  of the fallback (training plants only) and of the D5/D6 starting values, and the source of the
  `height_rate` targets (see limitation 4g).

**Causal fit counts** [measured] (`pheno4d_growth_curves_causal.csv`, 446 rows = 223 scans × 2
traits): 334 fitted, 112 too few earlier points (each plant's first 4 scans), 0 failed to converge.
Per split, from the D2 log (identical in all 3 seeds). "Poor" is D0's flag: R² < 0.8, or K/A near
its bound.

| Split | Trait | Scans | Causal fits | Fallback | Poor, Logistic | Poor, Gompertz |
|---|---|---|---|---|---|---|
| train | height | 72 | 56 | 16 | 32/56 | 39/56 |
| train | stem_diameter | 72 | 56 | 16 | 54/56 | 55/56 |
| val | height | 18 | 14 | 4 | 9/14 | 9/14 |
| val | stem_diameter | 18 | 14 | 4 | 13/14 | 13/14 |
| test | height | 63 | 47 | 16 | 23/47 | 28/47 |
| test | stem_diameter | 63 | 47 | 16 | 47/47 | 47/47 |

## Jobs

| Job | What | Outcome |
|---|---|---|
| smoke test (login node, 2026-10-27) | D2, seed 1, into `results/FW_D_causal_smoke_s1/D2` | finished; test metrics match the array job's seed-1 D2 (R² identical to 4 decimals) |
| 366279 (array, tasks 1–3 = seeds 1–3, `jobs/fw/fw_D_causal.sbatch`) | per task: D2, D3, D4, D5, D5 control, D6, D6 control (CPU; D5/D6 and controls warm-start from the same seed's new D4) | all 3 tasks `COMPLETED`, exit 0, every step `OK` in `results/logs/fw_D_causal_366279_<task>.log`; 8m14s, 8m13s, 4m55s |

Results: `results/FW_D_causal_s<s>/<row>/`. Job 359157's folders (`results/FW_D<n>_s<s>/`,
`FW_D5_ablation_s<s>/`, `FW_D6_ablation_s<s>/`; 24 folders, 60 files) were checksummed (sha256)
before submission and again after: **unchanged**.

As in FW_3, the D5 control and the D6 control are identical term for term in every seed and are
reported once as "control (λ=0)".

---

## 1. Per-term test R², old (FW_3, job 359157) next to new (causal, job 366279)

Metric: test R² per term (Pheno4D held-out plants; best epoch by lowest val total loss; test labels
never used for selection). SD = sample SD over 3 seeds. Test set sizes are as in FW_3 (D1–D3: 63 / 61
/ 58 / 58 / 63 / 63 / 61; D4–D6 and control: 54 / 52 / 51 / 51 / 54 / 54 / 52, in the term order
below); verified identical between old and new. Numbers were parsed from the
`=== FINAL test metrics` blocks with a throwaway parser that was not committed.

Best epochs (s1 / s2 / s3), old → new: D2 163/163/144 → 187/194/181; D3 163/153/166 → 155/161/138;
D4 158/171/163 → 154/172/186; D5 199/188/138 → 187/181/151; D6 196/188/138 → 187/181/151; control
148/167/138 → 187/181/151.

### D1 (deep features only; not rerun, FW_3 numbers)

| Term | Mean ± SD | s1 / s2 / s3 |
|---|---|---|
| height | 0.8853 ± 0.0309 | 0.8505 / 0.8961 / 0.9094 |
| stem_diameter | 0.1153 ± 0.0188 | 0.1141 / 0.1347 / 0.0972 |
| leaf_area | 0.4668 ± 0.1170 | 0.5482 / 0.5195 / 0.3327 |
| leaf_count | 0.2872 ± 0.0691 | 0.2497 / 0.2449 / 0.3669 |
| volume | 0.1294 ± 0.3351 | −0.0449 / −0.0826 / 0.5157 |
| height_rate | 0.6188 ± 0.0651 | 0.6292 / 0.5491 / 0.6780 |
| stem_diameter_rate | 0.1572 ± 0.0283 | 0.1894 / 0.1457 / 0.1365 |

### D2 (+ growth-curve parameters)

| Term | Old mean ± SD | Old s1 / s2 / s3 | New mean ± SD | New s1 / s2 / s3 | New − old |
|---|---|---|---|---|---|
| height | 0.8897 ± 0.0284 | 0.8576 / 0.9002 / 0.9114 | 0.8842 ± 0.0322 | 0.8510 / 0.8862 / 0.9153 | −0.0056 |
| stem_diameter | 0.1106 ± 0.0210 | 0.0972 / 0.1348 / 0.0999 | 0.1384 ± 0.0156 | 0.1233 / 0.1544 / 0.1374 | +0.0277 |
| leaf_area | 0.4473 ± 0.0957 | 0.5334 / 0.4641 / 0.3443 | 0.4587 ± 0.0974 | 0.5278 / 0.5010 / 0.3473 | +0.0114 |
| leaf_count | 0.2949 ± 0.0644 | 0.2628 / 0.2529 / 0.3690 | 0.2914 ± 0.0649 | 0.2616 / 0.2467 / 0.3659 | −0.0035 |
| volume | 0.0961 ± 0.3770 | −0.1154 / −0.1277 / 0.5314 | 0.0932 ± 0.3686 | −0.1298 / −0.1092 / 0.5187 | −0.0029 |
| height_rate | 0.6453 ± 0.0506 | 0.6590 / 0.5892 / 0.6876 | 0.6575 ± 0.0551 | 0.6822 / 0.5944 / 0.6960 | +0.0123 |
| stem_diameter_rate | 0.2013 ± 0.0106 | 0.2099 / 0.2044 / 0.1895 | 0.1116 ± 0.0316 | 0.1418 / 0.1141 / 0.0788 | −0.0897 |

### D3 (+ temporal information)

| Term | Old mean ± SD | Old s1 / s2 / s3 | New mean ± SD | New s1 / s2 / s3 | New − old |
|---|---|---|---|---|---|
| height | 0.9022 ± 0.0243 | 0.8741 / 0.9153 / 0.9171 | 0.8956 ± 0.0273 | 0.8688 / 0.8947 / 0.9233 | −0.0066 |
| stem_diameter | 0.1063 ± 0.0172 | 0.0963 / 0.1262 / 0.0964 | 0.1421 ± 0.0258 | 0.1179 / 0.1692 / 0.1392 | +0.0358 |
| leaf_area | 0.5180 ± 0.0893 | 0.5835 / 0.5541 / 0.4163 | 0.5025 ± 0.0785 | 0.5431 / 0.5524 / 0.4120 | −0.0155 |
| leaf_count | 0.2999 ± 0.0498 | 0.2652 / 0.2776 / 0.3569 | 0.3073 ± 0.0645 | 0.2574 / 0.2845 / 0.3801 | +0.0074 |
| volume | 0.1487 ± 0.3166 | −0.0176 / −0.0501 / 0.5138 | 0.1308 ± 0.3369 | −0.0263 / −0.0989 / 0.5175 | −0.0179 |
| height_rate | 0.6481 ± 0.0642 | 0.6711 / 0.5756 / 0.6976 | 0.6816 ± 0.0610 | 0.7144 / 0.6112 / 0.7193 | +0.0335 |
| stem_diameter_rate | 0.2062 ± 0.0118 | 0.2177 / 0.2066 / 0.1942 | 0.1328 ± 0.0317 | 0.1679 / 0.1242 / 0.1062 | −0.0734 |

### D4 (+ previous growth stage; 54-scan test set)

| Term | Old mean ± SD | Old s1 / s2 / s3 | New mean ± SD | New s1 / s2 / s3 | New − old |
|---|---|---|---|---|---|
| height | 0.9155 ± 0.0162 | 0.8973 / 0.9211 / 0.9282 | 0.9074 ± 0.0243 | 0.8881 / 0.8994 / 0.9347 | −0.0081 |
| stem_diameter | 0.0975 ± 0.0187 | 0.0801 / 0.0953 / 0.1172 | 0.1072 ± 0.0066 | 0.1101 / 0.0996 / 0.1119 | +0.0097 |
| leaf_area | 0.7161 ± 0.0478 | 0.7239 / 0.7595 / 0.6648 | 0.7190 ± 0.0423 | 0.7632 / 0.7150 / 0.6789 | +0.0030 |
| leaf_count | 0.3885 ± 0.0448 | 0.3462 / 0.3838 / 0.4355 | 0.3937 ± 0.0353 | 0.3624 / 0.3867 / 0.4320 | +0.0052 |
| volume | 0.3352 ± 0.3063 | 0.1165 / 0.2038 / 0.6853 | 0.2959 ± 0.3588 | 0.2245 / −0.0218 / 0.6850 | −0.0393 |
| height_rate | 0.6390 ± 0.0469 | 0.6455 / 0.5892 / 0.6823 | 0.6287 ± 0.0333 | 0.6633 / 0.5969 / 0.6259 | −0.0103 |
| stem_diameter_rate | 0.1291 ± 0.0256 | 0.1486 / 0.1386 / 0.1001 | 0.1258 ± 0.0606 | 0.1800 / 0.1370 / 0.0603 | −0.0033 |

### D5 (D4 + Logistic ODE physics loss; 54-scan test set)

| Term | Old mean ± SD | Old s1 / s2 / s3 | New mean ± SD | New s1 / s2 / s3 | New − old |
|---|---|---|---|---|---|
| height | 0.9240 ± 0.0132 | 0.9096 / 0.9355 / 0.9268 | 0.9231 ± 0.0149 | 0.9061 / 0.9336 / 0.9297 | −0.0008 |
| stem_diameter | 0.1024 ± 0.0064 | 0.0966 / 0.1014 / 0.1092 | 0.1137 ± 0.0070 | 0.1076 / 0.1213 / 0.1123 | +0.0113 |
| leaf_area | 0.7966 ± 0.0334 | 0.8142 / 0.8174 / 0.7581 | 0.8006 ± 0.0309 | 0.8149 / 0.8218 / 0.7652 | +0.0041 |
| leaf_count | 0.3925 ± 0.0627 | 0.3316 / 0.3892 / 0.4568 | 0.4010 ± 0.0523 | 0.3469 / 0.4047 / 0.4514 | +0.0085 |
| volume | 0.5405 ± 0.2218 | 0.4067 / 0.4182 / 0.7965 | 0.5349 ± 0.2116 | 0.4801 / 0.3561 / 0.7686 | −0.0055 |
| height_rate | 0.6365 ± 0.0720 | 0.6606 / 0.5555 / 0.6933 | 0.6446 ± 0.0303 | 0.6662 / 0.6100 / 0.6575 | +0.0081 |
| stem_diameter_rate | 0.1337 ± 0.0238 | 0.1543 / 0.1391 / 0.1077 | 0.1423 ± 0.0608 | 0.1863 / 0.1678 / 0.0729 | +0.0086 |

### D6 (D4 + Gompertz ODE physics loss; 54-scan test set)

| Term | Old mean ± SD | Old s1 / s2 / s3 | New mean ± SD | New s1 / s2 / s3 | New − old |
|---|---|---|---|---|---|
| height | 0.9205 ± 0.0220 | 0.8951 / 0.9339 / 0.9324 | 0.9189 ± 0.0251 | 0.8901 / 0.9311 / 0.9356 | −0.0015 |
| stem_diameter | 0.1024 ± 0.0065 | 0.0964 / 0.1015 / 0.1093 | 0.1138 ± 0.0070 | 0.1075 / 0.1214 / 0.1126 | +0.0114 |
| leaf_area | 0.7966 ± 0.0334 | 0.8142 / 0.8174 / 0.7581 | 0.8006 ± 0.0309 | 0.8149 / 0.8218 / 0.7652 | +0.0041 |
| leaf_count | 0.3925 ± 0.0627 | 0.3316 / 0.3892 / 0.4568 | 0.4010 ± 0.0523 | 0.3469 / 0.4047 / 0.4514 | +0.0085 |
| volume | 0.5404 ± 0.2218 | 0.4066 / 0.4182 / 0.7965 | 0.5349 ± 0.2116 | 0.4801 / 0.3561 / 0.7686 | −0.0055 |
| height_rate | 0.6365 ± 0.0720 | 0.6606 / 0.5555 / 0.6933 | 0.6446 ± 0.0303 | 0.6662 / 0.6100 / 0.6575 | +0.0081 |
| stem_diameter_rate | 0.1337 ± 0.0238 | 0.1543 / 0.1391 / 0.1077 | 0.1423 ± 0.0608 | 0.1863 / 0.1678 / 0.0729 | +0.0086 |

### Control, λ=0 (D5 control = D6 control; 54-scan test set)

| Term | Old mean ± SD | Old s1 / s2 / s3 | New mean ± SD | New s1 / s2 / s3 | New − old |
|---|---|---|---|---|---|
| height | 0.9138 ± 0.0311 | 0.8784 / 0.9260 / 0.9369 | 0.9077 ± 0.0326 | 0.8716 / 0.9162 / 0.9352 | −0.0061 |
| stem_diameter | 0.1018 ± 0.0118 | 0.0914 / 0.0993 / 0.1147 | 0.1145 ± 0.0060 | 0.1077 / 0.1190 / 0.1168 | +0.0127 |
| leaf_area | 0.7881 ± 0.0262 | 0.8000 / 0.8063 / 0.7581 | 0.8006 ± 0.0309 | 0.8149 / 0.8218 / 0.7652 | +0.0125 |
| leaf_count | 0.3921 ± 0.0617 | 0.3340 / 0.3854 / 0.4568 | 0.4010 ± 0.0523 | 0.3469 / 0.4047 / 0.4514 | +0.0089 |
| volume | 0.5542 ± 0.2099 | 0.4383 / 0.4278 / 0.7965 | 0.5349 ± 0.2116 | 0.4801 / 0.3561 / 0.7686 | −0.0193 |
| height_rate | 0.6411 ± 0.0713 | 0.6702 / 0.5599 / 0.6933 | 0.6446 ± 0.0303 | 0.6662 / 0.6100 / 0.6575 | +0.0034 |
| stem_diameter_rate | 0.1332 ± 0.0226 | 0.1510 / 0.1408 / 0.1077 | 0.1423 ± 0.0608 | 0.1863 / 0.1678 / 0.0729 | +0.0092 |

---

## 2. All-3-seeds rule, old next to new

Rule as in FW_3: a variant beats a reference on a term only if **all 3** of its seeds exceed the
reference's 3-seed mean. Cells: old (FW_3) → new. **up** = 3/3 above, **down** = 0/3 above (all 3
below). D1 is the same runs in both.

| Comparison | height | stem_diameter | leaf_area | leaf_count | volume | height_rate | stem_diameter_rate |
|---|---|---|---|---|---|---|---|
| D2 vs D1 | 2/3 → 2/3 | 1/3 → 3/3 **up** | 1/3 → 2/3 | 1/3 → 1/3 | 1/3 → 1/3 | 2/3 → 2/3 | 3/3 **up** → 0/3 **down** |
| D3 vs D2 | 2/3 → 2/3 | 1/3 → 2/3 | 2/3 → 2/3 | 1/3 → 1/3 | 1/3 → 1/3 | 2/3 → 2/3 | 2/3 → 2/3 |
| D4 vs D3 (54 vs 63 scans) | 2/3 → 2/3 | 1/3 → 0/3 **down** | 3/3 **up** → 3/3 **up** | 3/3 **up** → 3/3 **up** | 2/3 → 2/3 | 1/3 → 0/3 **down** | 0/3 **down** → 2/3 |
| D5 vs D4 | 2/3 → 2/3 | 2/3 → 3/3 **up** | 3/3 **up** → 3/3 **up** | 2/3 → 2/3 | 3/3 **up** → 3/3 **up** | 2/3 → 2/3 | 2/3 → 2/3 |
| D6 vs D4 | 2/3 → 2/3 | 2/3 → 3/3 **up** | 3/3 **up** → 3/3 **up** | 2/3 → 2/3 | 3/3 **up** → 3/3 **up** | 2/3 → 2/3 | 2/3 → 2/3 |
| Control vs D4 | 2/3 → 2/3 | 2/3 → 3/3 **up** | 3/3 **up** → 3/3 **up** | 1/3 → 2/3 | 3/3 **up** → 3/3 **up** | 2/3 → 2/3 | 2/3 → 2/3 |
| D5 vs control | 2/3 → 2/3 | 1/3 → 1/3 | 2/3 → 2/3 | 1/3 → 2/3 | 1/3 → 1/3 | 2/3 → 2/3 | 2/3 → 2/3 |
| D6 vs control | 2/3 → 2/3 | 1/3 → 1/3 | 2/3 → 2/3 | 1/3 → 2/3 | 1/3 → 1/3 | 2/3 → 2/3 | 2/3 → 2/3 |

Paired per-seed difference against the control on the two terms the physics/monotonicity loss acts
on (s1 / s2 / s3):

| | height, old | height, new | stem_diameter, old | stem_diameter, new |
|---|---|---|---|---|
| D5 − control | +0.0312 / +0.0095 / −0.0101 | +0.0345 / +0.0174 / −0.0055 | +0.0052 / +0.0021 / −0.0055 | −0.0001 / +0.0023 / −0.0045 |
| D6 − control | +0.0167 / +0.0079 / −0.0045 | +0.0185 / +0.0149 / +0.0004 | +0.0050 / +0.0022 / −0.0054 | −0.0002 / +0.0024 / −0.0042 |

---

## 3. Which FW_3 conclusions change

Each statement is labelled **[measured]** (read from the tables above) or **[inferred]**.

1. **Changes: D2's only 3/3 gain over D1 (stem_diameter_rate) was produced by the leak.** [measured]
   Old D2 vs D1 on stem_diameter_rate: 3/3 up (0.2013 vs 0.1572). New: 0/3 (0.1116, all 3 seeds
   below D1's mean). D3 shows the same drop (0.2062 → 0.1328). [inferred] stem_diameter_rate
   targets are finite differences of the plant's own stem_diameter series
   (`compute_growth_rate_targets.py`), and the old whole-trajectory stem_diameter fit summarised
   that same series, future scans included. With causal fits that information is gone, and D2/D3
   now do worse than D1 on this term. Why they fall *below* D1, rather than back to D1's level, is
   not known. Possible causes: the extra 7 noisy columns (every causal stem_diameter fit is flagged
   poor), and the fallback rows. Not tested.
2. **Changes: there is now no gain from growth-curve parameters on any growth term.** FW_3 already
   found no D2 > D1 effect on height, stem_diameter, volume or height_rate. The new run removes the
   one positive result FW_3 had (point 1), so **"+ growth curve params" has no supported benefit**
   on the causal inputs. [measured] D2's height mean is now 0.8842, below D1's 0.8853 (old D2
   0.8897).
3. **New: D2 and D3 beat D1 on stem_diameter on all 3 seeds** (D2 0.1384, D3 0.1421 vs D1 0.1153;
   D2 vs D1 3/3 up). [measured] The absolute level is still low (R² ≈ 0.14), and FW_3's point e
   (no row predicts stem_diameter usefully) still holds. [inferred] The cause is not known. All
   causal stem_diameter fits are flagged poor, so this is unlikely to be the curves carrying real
   growth information. It could come from the fit-available indicator (a proxy for "early scan")
   or from the training-plant fallback values. Not tested; treat as unexplained.
4. **Unchanged: D4 vs D3 leaf_area and leaf_count up on all 3 seeds.** [measured] Still mixed with
   the change of scan set (54 vs 63), as FW_3 point a says. What changes: the old "D4 < D3 on
   stem_diameter_rate" is no longer 3/3 (now 2/3). Two new 3/3-below results appear instead:
   stem_diameter and height_rate. [inferred] The first comes from D3 being higher on stem_diameter
   (point 3), not from D4 getting worse. All three D4-vs-D3 differences are still confounded by
   the scan set.
5. **Unchanged: D5, D6 and the control beat D4 on leaf_area and volume, and the physics loss is not
   the cause.** [measured] The control again does the same (3/3 on both), and on the causal inputs
   all three also beat D4 on stem_diameter (3/3). FW_3 point b stands: the gain comes with the
   warm start plus 200 more epochs, not with L_phys/L_mono.
6. **Unchanged: no physics-loss benefit passes the rule on any term.** [measured] D5 and D6 vs the
   control are at most 2/3 on every term. One detail is different. On height, the paired D6 −
   control difference is now positive in all 3 seeds (+0.0185 / +0.0149 / +0.0004), but the third
   is essentially zero, and the rule still fails: D6 seed 1 (0.8901) is below the control's mean
   (0.9077). D5 − control on height is +0.0345 / +0.0174 / −0.0055, the same 2-of-3 pattern as
   FW_3. So the conclusion stays "not confirmed". [inferred] If anything, D6's
   height result is the closest Block D has come to a physics-loss effect, and would be the one to
   test with more seeds.
7. **New, minor: D5, D6 and the control now select the same best epoch in every seed** (187 / 181 /
   151). [measured] As a result, the five terms the physics loss cannot reach are identical across
   D5, D6 and the control in all three seeds. Only height and stem_diameter differ, which is FW_3
   point c's "same epoch" case, now in every seed. This makes the paired differences above a
   cleaner comparison than in FW_3.
8. **Unchanged: volume is dominated by seed 3** (SD 0.21–0.37 in every row), **and the Block D seed
   still moves together with the backbone seed** (FW_3 points d, f). [measured]
9. **Size of the leak elsewhere.** [measured] Apart from stem_diameter_rate (D2/D3) and
   stem_diameter (D2–D6), every new mean is within ±0.04 of its old mean. Most rule cells are
   unchanged. [inferred] Outside stem_diameter_rate, the leaked inputs were not driving the FW_3
   results. The D5/D6 initialisation change (training plants only) has no visible effect beyond
   seed noise.

## 4. Limitations

- **g. height_rate targets still use whole-trajectory fits.** [measured] `height_rate` targets
  come from `compute_growth_rate_targets.py`, which evaluates each plant's best whole-trajectory
  curve (`pheno4d_growth_curves.csv`) at each scan's time. This is how the label is defined, not a
  model input, and it was left unchanged on purpose. The label for a test scan still depends on
  that plant's later scans. The stem_diameter_rate targets are finite differences of neighbouring
  scans (`np.gradient`), which also use the next scan.
- Early-scan fits are short: windows of 4–6 points make up a large share of the causal fits, and
  more than half of the height fits and almost all stem_diameter fits are flagged poor (table in
  section 0).
- The fallback uses per-species means, so species is an input of D2–D6 for the first 4 scans of
  each plant (it was implicitly one before, through each plant's own fit). [inferred] The B3b
  features already separate the two species, so this adds little new information.
- Three seeds, with the backbone seed and the D seed moving together (FW_3 point f).
- FW_3's results (job 359157) are kept as they were, and so are the step notes that cite them
  (`D2_Fusion_Growth_Curves.md`, `D5_D6_Physics_Loss.md`, `FW_3_Results.md`). Those numbers used
  the leaking inputs.
