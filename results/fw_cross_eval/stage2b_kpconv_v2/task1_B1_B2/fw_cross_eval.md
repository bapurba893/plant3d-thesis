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
| B1 | KPConv / DA-0 / ALL | 0.6984 ± 0.0399 | 0.7090 ± 0.0295 | +0.0106 | 0.7502 ± 0.0372 | 0.7708 ± 0.0232 | +0.0207 | 0.7354 / 0.6772 / 0.7143 | 0/3 | 17.7 / 18.3 | 1.3 / 0.0 |
| B2 | KPConv / DA-A / ALL | 0.6755 ± 0.0456 | 0.4550 ± 0.0826 | -0.2205 | 0.7075 ± 0.0359 | 0.5708 ± 0.0651 | -0.1367 | 0.4127 / 0.5503 / 0.4021 | 3/3 | 16.4 / 34.3 | 4.0 / 0.0 |

## Per run

| Run | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | T→M P4a / P4 | M→T P4a / P4 | Logged FINAL acc / bal | Own-cache check |
|---|---|---|---|---|---|---|---|---|---|
| FW_B1_s1 | 0.6561 [0.6032–0.6984] | 0.7354 [0.7302–0.7460] | +0.0794 | 0.7076 [0.6598–0.7533] | 0.7917 [0.7875–0.8000] | 19.3 / 16.7 | 2.3 / 0.0 | 0.6508 / 0.7065 | logged inside repeat range |
| FW_B1_s2 | 0.7354 [0.7143–0.7619] | 0.6772 [0.6667–0.6825] | -0.0582 | 0.7763 [0.7473–0.8125] | 0.7458 [0.7375–0.7500] | 15.0 / 20.3 | 1.7 / 0.0 | 0.7619 / 0.7848 | logged inside repeat range |
| FW_B1_s3 | 0.7037 [0.6825–0.7302] | 0.7143 [0.7143–0.7143] | +0.0106 | 0.7667 [0.7500–0.7875] | 0.7750 [0.7750–0.7750] | 18.7 / 18.0 | 0.0 / 0.0 | 0.7460 / 0.8000 | logged OUTSIDE repeat range |
| FW_B2_s1 | 0.7249 [0.7143–0.7302] | 0.4127 [0.3968–0.4286] | -0.3122 | 0.7464 [0.7380–0.7505] | 0.5375 [0.5250–0.5500] | 13.3 / 37.0 | 4.0 / 0.0 | 0.6984 / 0.7255 | logged OUTSIDE repeat range |
| FW_B2_s2 | 0.6667 [0.6508–0.6825] | 0.5503 [0.5238–0.5714] | -0.1164 | 0.7005 [0.6880–0.7130] | 0.6458 [0.6250–0.6625] | 17.0 / 28.3 | 4.0 / 0.0 | 0.6667 / 0.7005 | logged inside repeat range |
| FW_B2_s3 | 0.6349 [0.6190–0.6508] | 0.4021 [0.3968–0.4127] | -0.2328 | 0.6755 [0.6630–0.6880] | 0.5292 [0.5250–0.5375] | 19.0 / 37.7 | 4.0 / 0.0 | 0.6508 / 0.6880 | logged inside repeat range |
