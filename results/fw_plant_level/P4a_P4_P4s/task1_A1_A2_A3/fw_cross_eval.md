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
| A1 | PointNet++ / DA-0 / ALL | 0.4286 ± 0.0420 | 0.6384 ± 0.0291 | 0.6790 ± 0.0133 | +0.2099 | +0.2504 | 0.5500 ± 0.0331 | 0.7153 ± 0.0229 | 0.7472 ± 0.0105 | 0.4127 / 0.3968 / 0.4762 | 0.6402 / 0.6085 / 0.6667 | 0.6931 / 0.6772 / 0.6667 | 3/3 | 1/3 | 0/3 | 36.0 | 22.8 | 20.2 | 0.0 | 0.0 | 0.0 |
| A2 | PointNet++ / DA-A / ALL | 0.5291 ± 0.1244 | 0.5203 ± 0.0186 | 0.5556 ± 0.0212 | -0.0088 | +0.0265 | 0.6292 ± 0.0980 | 0.6222 ± 0.0146 | 0.6500 ± 0.0167 | 0.6561 / 0.4074 / 0.5238 | 0.5026 / 0.5397 / 0.5185 | 0.5344 / 0.5767 / 0.5556 | 2/3 | 3/3 | 3/3 | 29.7 | 30.2 | 28.0 | 0.0 | 0.0 | 0.0 |
| A3 | PointNet++ / DA-S / ALL | 0.7601 ± 0.2610 | 0.6296 ± 0.2021 | 0.6296 ± 0.2090 | -0.1305 | -0.1305 | 0.8111 ± 0.2055 | 0.7083 ± 0.1592 | 0.7083 ± 0.1646 | 0.8836 / 0.4603 / 0.9365 | 0.6984 / 0.4021 / 0.7884 | 0.7143 / 0.3915 / 0.7831 | 1/3 | 1/3 | 1/3 | 15.1 | 23.3 | 23.3 | 0.0 | 0.0 | 0.0 |

## Per run

| Run | Acc P4a | Acc P4 | Acc P4s | Δ acc P4 | Δ acc P4s | Bal. acc P4a | Bal. acc P4 | Bal. acc P4s | T→M P4a | T→M P4 | T→M P4s | M→T P4a | M→T P4 | M→T P4s | Logged FINAL acc / bal | Own-cache check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FW_A1_s1 | 0.4127 [0.4127–0.4127] | 0.6402 [0.6349–0.6508] | 0.6931 [0.6667–0.7302] | +0.2275 | +0.2804 | 0.5375 [0.5375–0.5375] | 0.7167 [0.7125–0.7250] | 0.7583 [0.7375–0.7875] | 37.0 | 22.7 | 19.3 | 0.0 | 0.0 | 0.0 | 0.4127 / 0.5375 | logged inside repeat range |
| FW_A1_s2 | 0.3968 [0.3968–0.3968] | 0.6085 [0.5873–0.6190] | 0.6772 [0.6508–0.6984] | +0.2116 | +0.2804 | 0.5250 [0.5250–0.5250] | 0.6917 [0.6750–0.7000] | 0.7458 [0.7250–0.7625] | 38.0 | 24.7 | 20.3 | 0.0 | 0.0 | 0.0 | 0.3968 / 0.5250 | logged inside repeat range |
| FW_A1_s3 | 0.4762 [0.4603–0.4921] | 0.6667 [0.6508–0.6825] | 0.6667 [0.6667–0.6667] | +0.1905 | +0.1905 | 0.5875 [0.5750–0.6000] | 0.7375 [0.7250–0.7500] | 0.7375 [0.7375–0.7375] | 33.0 | 21.0 | 21.0 | 0.0 | 0.0 | 0.0 | 0.4603 / 0.5750 | logged inside repeat range |
| FW_A2_s1 | 0.6561 [0.6508–0.6667] | 0.5026 [0.4921–0.5079] | 0.5344 [0.5238–0.5397] | -0.1534 | -0.1217 | 0.7292 [0.7250–0.7375] | 0.6083 [0.6000–0.6125] | 0.6333 [0.6250–0.6375] | 21.7 | 31.3 | 29.3 | 0.0 | 0.0 | 0.0 | 0.6349 / 0.7125 | logged OUTSIDE repeat range |
| FW_A2_s2 | 0.4074 [0.3968–0.4286] | 0.5397 [0.5079–0.5714] | 0.5767 [0.5556–0.5873] | +0.1323 | +0.1693 | 0.5333 [0.5250–0.5500] | 0.6375 [0.6125–0.6625] | 0.6667 [0.6500–0.6750] | 37.3 | 29.0 | 26.7 | 0.0 | 0.0 | 0.0 | 0.3968 / 0.5250 | logged inside repeat range |
| FW_A2_s3 | 0.5238 [0.4921–0.5397] | 0.5185 [0.4762–0.5397] | 0.5556 [0.5238–0.5714] | -0.0053 | +0.0317 | 0.6250 [0.6000–0.6375] | 0.6208 [0.5875–0.6375] | 0.6500 [0.6250–0.6625] | 30.0 | 30.3 | 28.0 | 0.0 | 0.0 | 0.0 | 0.5079 / 0.6125 | logged inside repeat range |
| FW_A3_s1 | 0.8836 [0.8730–0.9048] | 0.6984 [0.6667–0.7302] | 0.7143 [0.6984–0.7302] | -0.1852 | -0.1693 | 0.9083 [0.9000–0.9250] | 0.7625 [0.7375–0.7875] | 0.7750 [0.7625–0.7875] | 7.3 | 19.0 | 18.0 | 0.0 | 0.0 | 0.0 | 0.8889 / 0.9125 | logged inside repeat range |
| FW_A3_s2 | 0.4603 [0.4444–0.4762] | 0.4021 [0.3968–0.4127] | 0.3915 [0.3810–0.3968] | -0.0582 | -0.0688 | 0.5750 [0.5625–0.5875] | 0.5292 [0.5250–0.5375] | 0.5208 [0.5125–0.5250] | 34.0 | 37.7 | 38.3 | 0.0 | 0.0 | 0.0 | 0.4762 / 0.5875 | logged inside repeat range |
| FW_A3_s3 | 0.9365 [0.9365–0.9365] | 0.7884 [0.7778–0.7937] | 0.7831 [0.7778–0.7937] | -0.1481 | -0.1534 | 0.9500 [0.9500–0.9500] | 0.8333 [0.8250–0.8375] | 0.8292 [0.8250–0.8375] | 4.0 | 13.3 | 13.7 | 0.0 | 0.0 | 0.0 | 0.9524 / 0.9625 | logged OUTSIDE repeat range |
