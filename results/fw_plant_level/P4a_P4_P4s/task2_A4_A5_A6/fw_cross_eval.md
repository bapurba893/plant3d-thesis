# FW cross-evaluation of the P4a-trained models on caches P4a, P4, P4s

**Status: complete.**

Written by `scripts/16_fw_cross_eval.py`. No retraining: each run's selected checkpoint (`results/FW_<row>_s<seed>/model.pt`; C1 = `results/PP_P4a_s<seed>/model.pt`) is evaluated on the 63 Pheno4D held-out clouds of each cache: same clouds, same order (checked per run). Δ = cache − P4a. Per-cloud predictions: `fw_cross_eval_clouds.csv`.

**Caveat: P4 is normalised after the soil cut.** The unit-sphere centre and scale are computed on the plant only, the voxel size is height-relative, and all 4096 FPS points fall on the plant. So on P4 the plant's scale, centre and point density also differ from P4a (P4/P4a norm_scale median 0.78 over the 63 clouds). A drop from P4a to P4 is therefore **not only the loss of the soil cue**.

**KPConv:** neighborhood_limits are the training-time limits printed in each run's own `run.log`; they were not recalibrated on any cache.

**Repeats:** PointNet++ and KPConv evaluation is not repeatable (random FPS start; random subsampling-grid rotation), so A/B runs are evaluated 3 times per cache (seeds 0, 1, 2): value = mean, [min–max] of the 3 repeats. DGCNN: one evaluation.

Majority baseline (always Tomato) = 40/63 = 0.635. SD = sample SD over seeds. T→M = tomato clouds predicted maize, M→T = maize clouds predicted tomato (of 40 tomato, 23 maize; repeat mean for A/B).

## Adaptation rows (non-Oracle): per row (3 seeds)

| Row | Backbone / DA / Aug | Acc P4a | Acc P4 | Acc P4s | Δ acc P4 | Δ acc P4s | Bal. acc P4a | Bal. acc P4 | Bal. acc P4s | Acc P4a per seed | Acc P4 per seed | Acc P4s per seed | Seeds < 0.635 P4a | Seeds < 0.635 P4 | Seeds < 0.635 P4s | T→M P4a | T→M P4 | T→M P4s | M→T P4a | M→T P4 | M→T P4s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A4 | PointNet++ / DA-A / L-N | 0.5079 ± 0.1041 | 0.5573 ± 0.1565 | 0.5397 ± 0.1440 | +0.0494 | +0.0317 | 0.6125 ± 0.0820 | 0.6514 ± 0.1232 | 0.6375 ± 0.1134 | 0.3968 / 0.6032 / 0.5238 | 0.3862 / 0.6931 / 0.5926 | 0.3757 / 0.6455 / 0.5979 | 3/3 | 2/3 | 2/3 | 31.0 | 27.9 | 29.0 | 0.0 | 0.0 | 0.0 |
| A6 | PointNet++ / DA-D / ALL | 0.7372 ± 0.1123 | 0.6631 ± 0.0401 | 0.6596 ± 0.0272 | -0.0741 | -0.0776 | 0.7931 ± 0.0884 | 0.7347 ± 0.0315 | 0.7319 ± 0.0214 | 0.6085 / 0.7884 / 0.8148 | 0.6455 / 0.7090 / 0.6349 | 0.6667 / 0.6825 / 0.6296 | 1/3 | 0/3 | 1/3 | 16.6 | 21.2 | 21.4 | 0.0 | 0.0 | 0.0 |

## Oracle rows (A5, B5, C5; upper bound, not pooled): per row (3 seeds)

| Row | Backbone / DA / Aug | Acc P4a | Acc P4 | Acc P4s | Δ acc P4 | Δ acc P4s | Bal. acc P4a | Bal. acc P4 | Bal. acc P4s | Acc P4a per seed | Acc P4 per seed | Acc P4s per seed | Seeds < 0.635 P4a | Seeds < 0.635 P4 | Seeds < 0.635 P4s | T→M P4a | T→M P4 | T→M P4s | M→T P4a | M→T P4 | M→T P4s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A5 | PointNet++ / DA-O / ALL | 0.9947 ± 0.0053 | 0.8642 ± 0.0492 | 0.8325 ± 0.0670 | -0.1305 | -0.1623 | 0.9958 ± 0.0042 | 0.8931 ± 0.0387 | 0.8681 ± 0.0528 | 0.9894 / 0.9947 / 1.0000 | 0.9206 / 0.8307 / 0.8413 | 0.9048 / 0.8201 / 0.7725 | 0/3 | 0/3 | 0/3 | 0.3 | 8.6 | 10.6 | 0.0 | 0.0 | 0.0 |

## Per run

| Run | Acc P4a | Acc P4 | Acc P4s | Δ acc P4 | Δ acc P4s | Bal. acc P4a | Bal. acc P4 | Bal. acc P4s | T→M P4a | T→M P4 | T→M P4s | M→T P4a | M→T P4 | M→T P4s | Logged FINAL acc / bal | Own-cache check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FW_A4_s1 | 0.3968 [0.3810–0.4127] | 0.3862 [0.3810–0.3968] | 0.3757 [0.3651–0.3968] | -0.0106 | -0.0212 | 0.5250 [0.5125–0.5375] | 0.5167 [0.5125–0.5250] | 0.5083 [0.5000–0.5250] | 38.0 | 38.7 | 39.3 | 0.0 | 0.0 | 0.0 | 0.3968 / 0.5250 | logged inside repeat range |
| FW_A4_s2 | 0.6032 [0.5873–0.6190] | 0.6931 [0.6667–0.7302] | 0.6455 [0.6349–0.6508] | +0.0899 | +0.0423 | 0.6875 [0.6750–0.7000] | 0.7583 [0.7375–0.7875] | 0.7208 [0.7125–0.7250] | 25.0 | 19.3 | 22.3 | 0.0 | 0.0 | 0.0 | 0.6032 / 0.6875 | logged inside repeat range |
| FW_A4_s3 | 0.5238 [0.5238–0.5238] | 0.5926 [0.5873–0.6032] | 0.5979 [0.5873–0.6032] | +0.0688 | +0.0741 | 0.6250 [0.6250–0.6250] | 0.6792 [0.6750–0.6875] | 0.6833 [0.6750–0.6875] | 30.0 | 25.7 | 25.3 | 0.0 | 0.0 | 0.0 | 0.5556 / 0.6500 | logged OUTSIDE repeat range |
| FW_A5_s1 (Oracle) | 0.9894 [0.9841–1.0000] | 0.9206 [0.9206–0.9206] | 0.9048 [0.9048–0.9048] | -0.0688 | -0.0847 | 0.9917 [0.9875–1.0000] | 0.9375 [0.9375–0.9375] | 0.9250 [0.9250–0.9250] | 0.7 | 5.0 | 6.0 | 0.0 | 0.0 | 0.0 | 0.9841 / 0.9875 | logged inside repeat range |
| FW_A5_s2 (Oracle) | 0.9947 [0.9841–1.0000] | 0.8307 [0.8254–0.8413] | 0.8201 [0.8095–0.8254] | -0.1640 | -0.1746 | 0.9958 [0.9875–1.0000] | 0.8667 [0.8625–0.8750] | 0.8583 [0.8500–0.8625] | 0.3 | 10.7 | 11.3 | 0.0 | 0.0 | 0.0 | 0.9841 / 0.9875 | logged inside repeat range |
| FW_A5_s3 (Oracle) | 1.0000 [1.0000–1.0000] | 0.8413 [0.8413–0.8413] | 0.7725 [0.7619–0.7778] | -0.1587 | -0.2275 | 1.0000 [1.0000–1.0000] | 0.8750 [0.8750–0.8750] | 0.8208 [0.8125–0.8250] | 0.0 | 10.0 | 14.3 | 0.0 | 0.0 | 0.0 | 1.0000 / 1.0000 | logged inside repeat range |
| FW_A6_s1 | 0.6085 [0.5397–0.6508] | 0.6455 [0.6190–0.6825] | 0.6667 [0.6508–0.6825] | +0.0370 | +0.0582 | 0.6917 [0.6375–0.7250] | 0.7208 [0.7000–0.7500] | 0.7375 [0.7250–0.7500] | 24.7 | 22.3 | 21.0 | 0.0 | 0.0 | 0.0 | 0.6825 / 0.7500 | logged OUTSIDE repeat range |
| FW_A6_s2 | 0.7884 [0.7778–0.8095] | 0.7090 [0.6984–0.7143] | 0.6825 [0.6825–0.6825] | -0.0794 | -0.1058 | 0.8333 [0.8250–0.8500] | 0.7708 [0.7625–0.7750] | 0.7500 [0.7500–0.7500] | 13.3 | 18.3 | 20.0 | 0.0 | 0.0 | 0.0 | 0.8254 / 0.8625 | logged OUTSIDE repeat range |
| FW_A6_s3 | 0.8148 [0.7937–0.8413] | 0.6349 [0.6190–0.6508] | 0.6296 [0.6190–0.6349] | -0.1799 | -0.1852 | 0.8542 [0.8375–0.8750] | 0.7125 [0.7000–0.7250] | 0.7083 [0.7000–0.7125] | 11.7 | 23.0 | 23.3 | 0.0 | 0.0 | 0.0 | 0.8095 / 0.8500 | logged inside repeat range |
