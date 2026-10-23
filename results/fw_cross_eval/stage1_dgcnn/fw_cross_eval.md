# FW soil-cue cross-evaluation (P4a-trained models on the soil-removed P4 target)

Written by `scripts/16_fw_cross_eval.py`. No retraining: each run's selected checkpoint (`results/FW_<row>_s<seed>/model.pt`; C1 = `results/PP_P4a_s<seed>/model.pt`) is evaluated on the 63 Pheno4D held-out clouds of the P4a cache (own, soil kept) and of the P4 cache (soil removed): same clouds, same order (checked per run).

**Caveat: P4 is normalised after the soil cut.** The unit-sphere centre and scale are computed on the plant only, the voxel size is height-relative, and all 4096 FPS points fall on the plant. So on P4 the plant's scale, centre and point density also differ from P4a (P4/P4a norm_scale median 0.78 over the 63 clouds). A drop from P4a to P4 is therefore **not only the loss of the soil cue**.

**KPConv:** neighborhood_limits are the training-time limits printed in each run's own `run.log`; they were not recalibrated on P4 or P4a.

**Repeats:** PointNet++ and KPConv evaluation is not repeatable (random FPS start; random subsampling-grid rotation), so A/B runs are evaluated 3 times per cache (seeds 0, 1, 2): value = mean, [min–max] of the 3 repeats. DGCNN: one evaluation.

Majority baseline (always Tomato) = 40/63 = 0.635. SD = sample SD over seeds. T→M = tomato clouds predicted maize, M→T = maize clouds predicted tomato (of 40 tomato, 23 maize; repeat mean for A/B).

## Adaptation rows (non-Oracle): per row (3 seeds)

| Row | Backbone / DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Δ bal. acc | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | DGCNN / DA-0 / ALL (= PP_P4a) | 0.8836 ± 0.0367 | 0.6508 ± 0.0966 | -0.2328 | 0.8652 ± 0.0253 | 0.7250 ± 0.0760 | -0.1402 | 0.6032 / 0.7619 / 0.5873 | 2/3 | 2.7 / 22.0 | 4.7 / 0.0 |
| C2 | DGCNN / DA-A / ALL | 0.7989 ± 0.1972 | 0.5714 ± 0.1672 | -0.2275 | 0.8109 ± 0.1294 | 0.6625 ± 0.1317 | -0.1484 | 0.3968 / 0.5873 / 0.7302 | 2/3 | 9.3 / 27.0 | 3.3 / 0.0 |
| C3 | DGCNN / DA-S / ALL | 0.9312 ± 0.0242 | 0.5503 ± 0.0601 | -0.3810 | 0.9181 ± 0.0261 | 0.6458 ± 0.0473 | -0.2723 | 0.5079 / 0.5238 / 0.6190 | 3/3 | 1.3 / 28.3 | 3.0 / 0.0 |
| C4 | DGCNN / DA-A / L-N | 0.6508 ± 0.0840 | 0.4021 ± 0.0330 | -0.2487 | 0.6973 ± 0.0650 | 0.5292 ± 0.0260 | -0.1681 | 0.4127 / 0.3651 / 0.4286 | 3/3 | 19.0 / 37.7 | 3.0 / 0.0 |
| C6 | DGCNN / DA-D / ALL | 0.9365 ± 0.0420 | 0.5503 ± 0.0750 | -0.3862 | 0.9223 ± 0.0419 | 0.6458 ± 0.0591 | -0.2764 | 0.5238 / 0.4921 / 0.6349 | 2/3 | 1.0 / 28.3 | 3.0 / 0.0 |

## Oracle rows (A5, B5, C5; upper bound, not pooled): per row (3 seeds)

| Row | Backbone / DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Δ bal. acc | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C5 | DGCNN / DA-O / ALL | 0.9524 ± 0.0420 | 0.8360 ± 0.0242 | -0.1164 | 0.9625 ± 0.0331 | 0.8708 ± 0.0191 | -0.0917 | 0.8095 / 0.8413 / 0.8571 | 0/3 | 3.0 / 10.3 | 0.0 / 0.0 |

## Per run

| Run | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | T→M P4a / P4 | M→T P4a / P4 | Logged FINAL acc / bal | Own-cache check |
|---|---|---|---|---|---|---|---|---|---|
| PP_P4a_s1 | 0.8413 | 0.6032 | -0.2381 | 0.8380 | 0.6875 | 6.0 / 25.0 | 4.0 / 0.0 | 0.8413 / 0.8380 | MATCH |
| PP_P4a_s2 | 0.9048 | 0.7619 | -0.1429 | 0.8696 | 0.8125 | 0.0 / 15.0 | 6.0 / 0.0 | 0.9048 / 0.8696 | MATCH |
| PP_P4a_s3 | 0.9048 | 0.5873 | -0.3175 | 0.8880 | 0.6750 | 2.0 / 26.0 | 4.0 / 0.0 | 0.9048 / 0.8880 | MATCH |
| FW_C2_s1 | 0.5714 | 0.3968 | -0.1746 | 0.6625 | 0.5250 | 27.0 / 38.0 | 0.0 / 0.0 | 0.5714 / 0.6625 | MATCH |
| FW_C2_s2 | 0.9206 | 0.5873 | -0.3333 | 0.9005 | 0.6750 | 1.0 / 26.0 | 4.0 / 0.0 | 0.9206 / 0.9005 | MATCH |
| FW_C2_s3 | 0.9048 | 0.7302 | -0.1746 | 0.8696 | 0.7875 | 0.0 / 17.0 | 6.0 / 0.0 | 0.9048 / 0.8696 | MATCH |
| FW_C3_s1 | 0.9048 | 0.5079 | -0.3968 | 0.8880 | 0.6125 | 2.0 / 31.0 | 4.0 / 0.0 | 0.9048 / 0.8880 | MATCH |
| FW_C3_s2 | 0.9365 | 0.5238 | -0.4127 | 0.9315 | 0.6250 | 2.0 / 30.0 | 2.0 / 0.0 | 0.9365 / 0.9315 | MATCH |
| FW_C3_s3 | 0.9524 | 0.6190 | -0.3333 | 0.9348 | 0.7000 | 0.0 / 24.0 | 3.0 / 0.0 | 0.9524 / 0.9348 | MATCH |
| FW_C4_s1 | 0.5873 | 0.4127 | -0.1746 | 0.6565 | 0.5375 | 24.0 / 37.0 | 2.0 / 0.0 | 0.5873 / 0.6565 | MATCH |
| FW_C4_s2 | 0.6190 | 0.3651 | -0.2540 | 0.6630 | 0.5000 | 20.0 / 40.0 | 4.0 / 0.0 | 0.6190 / 0.6630 | MATCH |
| FW_C4_s3 | 0.7460 | 0.4286 | -0.3175 | 0.7723 | 0.5500 | 13.0 / 36.0 | 3.0 / 0.0 | 0.7460 / 0.7723 | MATCH |
| FW_C5_s1 (Oracle) | 1.0000 | 0.8095 | -0.1905 | 1.0000 | 0.8500 | 0.0 / 12.0 | 0.0 / 0.0 | 1.0000 / 1.0000 | MATCH |
| FW_C5_s2 (Oracle) | 0.9365 | 0.8413 | -0.0952 | 0.9500 | 0.8750 | 4.0 / 10.0 | 0.0 / 0.0 | 0.9365 / 0.9500 | MATCH |
| FW_C5_s3 (Oracle) | 0.9206 | 0.8571 | -0.0635 | 0.9375 | 0.8875 | 5.0 / 9.0 | 0.0 / 0.0 | 0.9206 / 0.9375 | MATCH |
| FW_C6_s1 | 0.9683 | 0.5238 | -0.4444 | 0.9565 | 0.6250 | 0.0 / 30.0 | 2.0 / 0.0 | 0.9683 / 0.9565 | MATCH |
| FW_C6_s2 | 0.8889 | 0.4921 | -0.3968 | 0.8755 | 0.6000 | 3.0 / 32.0 | 4.0 / 0.0 | 0.8889 / 0.8755 | MATCH |
| FW_C6_s3 | 0.9524 | 0.6349 | -0.3175 | 0.9348 | 0.7125 | 0.0 / 23.0 | 3.0 / 0.0 | 0.9524 / 0.9348 | MATCH |

## C1 vs Block PP cross-evaluation (`results/pp_summary/cross_eval_runs.csv`)

- PP_P4a_s1 on P4a: acc 0.841270 / bal 0.838043 vs Block PP 0.841270 / 0.838043 -> MATCH
- PP_P4a_s1 on P4: acc 0.603175 / bal 0.687500 vs Block PP 0.603175 / 0.687500 -> MATCH
- PP_P4a_s2 on P4a: acc 0.904762 / bal 0.869565 vs Block PP 0.904762 / 0.869565 -> MATCH
- PP_P4a_s2 on P4: acc 0.761905 / bal 0.812500 vs Block PP 0.761905 / 0.812500 -> MATCH
- PP_P4a_s3 on P4a: acc 0.904762 / bal 0.888043 vs Block PP 0.904762 / 0.888043 -> MATCH
- PP_P4a_s3 on P4: acc 0.587302 / bal 0.675000 vs Block PP 0.587302 / 0.675000 -> MATCH
