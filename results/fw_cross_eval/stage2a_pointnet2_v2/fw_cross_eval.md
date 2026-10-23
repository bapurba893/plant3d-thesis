# FW soil-cue cross-evaluation (P4a-trained models on the soil-removed P4 target)

**Status: complete.**

Written by `scripts/16_fw_cross_eval.py`. No retraining: each run's selected checkpoint (`results/FW_<row>_s<seed>/model.pt`; C1 = `results/PP_P4a_s<seed>/model.pt`) is evaluated on the 63 Pheno4D held-out clouds of the P4a cache (own, soil kept) and of the P4 cache (soil removed): same clouds, same order (checked per run).

**Caveat: P4 is normalised after the soil cut.** The unit-sphere centre and scale are computed on the plant only, the voxel size is height-relative, and all 4096 FPS points fall on the plant. So on P4 the plant's scale, centre and point density also differ from P4a (P4/P4a norm_scale median 0.78 over the 63 clouds). A drop from P4a to P4 is therefore **not only the loss of the soil cue**.

**KPConv:** neighborhood_limits are the training-time limits printed in each run's own `run.log`; they were not recalibrated on P4 or P4a.

**Repeats:** PointNet++ and KPConv evaluation is not repeatable (random FPS start; random subsampling-grid rotation), so A/B runs are evaluated 3 times per cache (seeds 0, 1, 2): value = mean, [min–max] of the 3 repeats. DGCNN: one evaluation.

Majority baseline (always Tomato) = 40/63 = 0.635. SD = sample SD over seeds. T→M = tomato clouds predicted maize, M→T = maize clouds predicted tomato (of 40 tomato, 23 maize; repeat mean for A/B).

## Adaptation rows (non-Oracle): per row (3 seeds)

| Row | Backbone / DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Δ bal. acc | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | PointNet++ / DA-0 / ALL | 0.4286 ± 0.0420 | 0.6384 ± 0.0291 | +0.2099 | 0.5500 ± 0.0331 | 0.7153 ± 0.0229 | +0.1653 | 0.6402 / 0.6085 / 0.6667 | 1/3 | 36.0 / 22.8 | 0.0 / 0.0 |
| A2 | PointNet++ / DA-A / ALL | 0.5291 ± 0.1244 | 0.5203 ± 0.0186 | -0.0088 | 0.6292 ± 0.0980 | 0.6222 ± 0.0146 | -0.0069 | 0.5026 / 0.5397 / 0.5185 | 3/3 | 29.7 / 30.2 | 0.0 / 0.0 |
| A3 | PointNet++ / DA-S / ALL | 0.7601 ± 0.2610 | 0.6296 ± 0.2021 | -0.1305 | 0.8111 ± 0.2055 | 0.7083 ± 0.1592 | -0.1028 | 0.6984 / 0.4021 / 0.7884 | 1/3 | 15.1 / 23.3 | 0.0 / 0.0 |
| A4 | PointNet++ / DA-A / L-N | 0.5079 ± 0.1041 | 0.5573 ± 0.1565 | +0.0494 | 0.6125 ± 0.0820 | 0.6514 ± 0.1232 | +0.0389 | 0.3862 / 0.6931 / 0.5926 | 2/3 | 31.0 / 27.9 | 0.0 / 0.0 |
| A6 | PointNet++ / DA-D / ALL | 0.7372 ± 0.1123 | 0.6631 ± 0.0401 | -0.0741 | 0.7931 ± 0.0884 | 0.7347 ± 0.0315 | -0.0583 | 0.6455 / 0.7090 / 0.6349 | 0/3 | 16.6 / 21.2 | 0.0 / 0.0 |

## Oracle rows (A5, B5, C5; upper bound, not pooled): per row (3 seeds)

| Row | Backbone / DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Δ bal. acc | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A5 | PointNet++ / DA-O / ALL | 0.9947 ± 0.0053 | 0.8642 ± 0.0492 | -0.1305 | 0.9958 ± 0.0042 | 0.8931 ± 0.0387 | -0.1028 | 0.9206 / 0.8307 / 0.8413 | 0/3 | 0.3 / 8.6 | 0.0 / 0.0 |

## Per run

| Run | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | T→M P4a / P4 | M→T P4a / P4 | Logged FINAL acc / bal | Own-cache check |
|---|---|---|---|---|---|---|---|---|---|
| FW_A1_s1 | 0.4127 [0.4127–0.4127] | 0.6402 [0.6349–0.6508] | +0.2275 | 0.5375 [0.5375–0.5375] | 0.7167 [0.7125–0.7250] | 37.0 / 22.7 | 0.0 / 0.0 | 0.4127 / 0.5375 | logged inside repeat range |
| FW_A1_s2 | 0.3968 [0.3968–0.3968] | 0.6085 [0.5873–0.6190] | +0.2116 | 0.5250 [0.5250–0.5250] | 0.6917 [0.6750–0.7000] | 38.0 / 24.7 | 0.0 / 0.0 | 0.3968 / 0.5250 | logged inside repeat range |
| FW_A1_s3 | 0.4762 [0.4603–0.4921] | 0.6667 [0.6508–0.6825] | +0.1905 | 0.5875 [0.5750–0.6000] | 0.7375 [0.7250–0.7500] | 33.0 / 21.0 | 0.0 / 0.0 | 0.4603 / 0.5750 | logged inside repeat range |
| FW_A2_s1 | 0.6561 [0.6508–0.6667] | 0.5026 [0.4921–0.5079] | -0.1534 | 0.7292 [0.7250–0.7375] | 0.6083 [0.6000–0.6125] | 21.7 / 31.3 | 0.0 / 0.0 | 0.6349 / 0.7125 | logged OUTSIDE repeat range |
| FW_A2_s2 | 0.4074 [0.3968–0.4286] | 0.5397 [0.5079–0.5714] | +0.1323 | 0.5333 [0.5250–0.5500] | 0.6375 [0.6125–0.6625] | 37.3 / 29.0 | 0.0 / 0.0 | 0.3968 / 0.5250 | logged inside repeat range |
| FW_A2_s3 | 0.5238 [0.4921–0.5397] | 0.5185 [0.4762–0.5397] | -0.0053 | 0.6250 [0.6000–0.6375] | 0.6208 [0.5875–0.6375] | 30.0 / 30.3 | 0.0 / 0.0 | 0.5079 / 0.6125 | logged inside repeat range |
| FW_A3_s1 | 0.8836 [0.8730–0.9048] | 0.6984 [0.6667–0.7302] | -0.1852 | 0.9083 [0.9000–0.9250] | 0.7625 [0.7375–0.7875] | 7.3 / 19.0 | 0.0 / 0.0 | 0.8889 / 0.9125 | logged inside repeat range |
| FW_A3_s2 | 0.4603 [0.4444–0.4762] | 0.4021 [0.3968–0.4127] | -0.0582 | 0.5750 [0.5625–0.5875] | 0.5292 [0.5250–0.5375] | 34.0 / 37.7 | 0.0 / 0.0 | 0.4762 / 0.5875 | logged inside repeat range |
| FW_A3_s3 | 0.9365 [0.9365–0.9365] | 0.7884 [0.7778–0.7937] | -0.1481 | 0.9500 [0.9500–0.9500] | 0.8333 [0.8250–0.8375] | 4.0 / 13.3 | 0.0 / 0.0 | 0.9524 / 0.9625 | logged OUTSIDE repeat range |
| FW_A4_s1 | 0.3968 [0.3810–0.4127] | 0.3862 [0.3810–0.3968] | -0.0106 | 0.5250 [0.5125–0.5375] | 0.5167 [0.5125–0.5250] | 38.0 / 38.7 | 0.0 / 0.0 | 0.3968 / 0.5250 | logged inside repeat range |
| FW_A4_s2 | 0.6032 [0.5873–0.6190] | 0.6931 [0.6667–0.7302] | +0.0899 | 0.6875 [0.6750–0.7000] | 0.7583 [0.7375–0.7875] | 25.0 / 19.3 | 0.0 / 0.0 | 0.6032 / 0.6875 | logged inside repeat range |
| FW_A4_s3 | 0.5238 [0.5238–0.5238] | 0.5926 [0.5873–0.6032] | +0.0688 | 0.6250 [0.6250–0.6250] | 0.6792 [0.6750–0.6875] | 30.0 / 25.7 | 0.0 / 0.0 | 0.5556 / 0.6500 | logged OUTSIDE repeat range |
| FW_A5_s1 (Oracle) | 0.9894 [0.9841–1.0000] | 0.9206 [0.9206–0.9206] | -0.0688 | 0.9917 [0.9875–1.0000] | 0.9375 [0.9375–0.9375] | 0.7 / 5.0 | 0.0 / 0.0 | 0.9841 / 0.9875 | logged inside repeat range |
| FW_A5_s2 (Oracle) | 0.9947 [0.9841–1.0000] | 0.8307 [0.8254–0.8413] | -0.1640 | 0.9958 [0.9875–1.0000] | 0.8667 [0.8625–0.8750] | 0.3 / 10.7 | 0.0 / 0.0 | 0.9841 / 0.9875 | logged inside repeat range |
| FW_A5_s3 (Oracle) | 1.0000 [1.0000–1.0000] | 0.8413 [0.8413–0.8413] | -0.1587 | 1.0000 [1.0000–1.0000] | 0.8750 [0.8750–0.8750] | 0.0 / 10.0 | 0.0 / 0.0 | 1.0000 / 1.0000 | logged inside repeat range |
| FW_A6_s1 | 0.6085 [0.5397–0.6508] | 0.6455 [0.6190–0.6825] | +0.0370 | 0.6917 [0.6375–0.7250] | 0.7208 [0.7000–0.7500] | 24.7 / 22.3 | 0.0 / 0.0 | 0.6825 / 0.7500 | logged OUTSIDE repeat range |
| FW_A6_s2 | 0.7884 [0.7778–0.8095] | 0.7090 [0.6984–0.7143] | -0.0794 | 0.8333 [0.8250–0.8500] | 0.7708 [0.7625–0.7750] | 13.3 / 18.3 | 0.0 / 0.0 | 0.8254 / 0.8625 | logged OUTSIDE repeat range |
| FW_A6_s3 | 0.8148 [0.7937–0.8413] | 0.6349 [0.6190–0.6508] | -0.1799 | 0.8542 [0.8375–0.8750] | 0.7125 [0.7000–0.7250] | 11.7 / 23.0 | 0.0 / 0.0 | 0.8095 / 0.8500 | logged inside repeat range |
