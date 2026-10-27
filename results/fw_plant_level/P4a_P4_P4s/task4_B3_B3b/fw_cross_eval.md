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
| B3 | KPConv / DA-D / ALL | 0.6667 ± 0.0881 | 0.6261 ± 0.0680 | 0.6437 ± 0.0480 | -0.0406 | -0.0229 | 0.7098 ± 0.0603 | 0.7056 ± 0.0536 | 0.7194 ± 0.0378 | 0.7302 / 0.5661 / 0.7037 | 0.5979 / 0.7037 / 0.5767 | 0.6085 / 0.6984 / 0.6243 | 1/3 | 2/3 | 2/3 | 18.0 | 23.6 | 22.4 | 3.0 | 0.0 | 0.0 |
| B3b | KPConv / DA-S / ALL | 0.7072 ± 0.2298 | 0.6825 ± 0.0596 | 0.6720 ± 0.0790 | -0.0247 | -0.0353 | 0.7479 ± 0.1569 | 0.7500 ± 0.0470 | 0.7417 ± 0.0622 | 0.7619 / 0.9048 / 0.4550 | 0.7196 / 0.7143 / 0.6138 | 0.7302 / 0.7037 / 0.5820 | 1/3 | 1/3 | 1/3 | 16.1 | 20.0 | 20.7 | 2.3 | 0.0 | 0.0 |

## Per run

| Run | Acc P4a | Acc P4 | Acc P4s | Δ acc P4 | Δ acc P4s | Bal. acc P4a | Bal. acc P4 | Bal. acc P4s | T→M P4a | T→M P4 | T→M P4s | M→T P4a | M→T P4 | M→T P4s | Logged FINAL acc / bal | Own-cache check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FW_B3_s1 | 0.7302 [0.7143–0.7460] | 0.5979 [0.5714–0.6190] | 0.6085 [0.5873–0.6190] | -0.1323 | -0.1217 | 0.7659 [0.7565–0.7783] | 0.6833 [0.6625–0.7000] | 0.6917 [0.6750–0.7000] | 14.7 | 25.3 | 24.7 | 2.3 | 0.0 | 0.0 | 0.7302 / 0.7783 | logged inside repeat range |
| FW_B3_s2 | 0.5661 [0.5397–0.5873] | 0.7037 [0.6667–0.7302] | 0.6984 [0.6825–0.7143] | +0.1376 | +0.1323 | 0.6460 [0.6190–0.6658] | 0.7667 [0.7375–0.7875] | 0.7625 [0.7500–0.7750] | 26.0 | 18.7 | 19.0 | 1.3 | 0.0 | 0.0 | 0.5714 / 0.6625 | logged inside repeat range |
| FW_B3_s3 | 0.7037 [0.6667–0.7302] | 0.5767 [0.5556–0.6032] | 0.6243 [0.6032–0.6349] | -0.1270 | -0.0794 | 0.7174 [0.6913–0.7321] | 0.6667 [0.6500–0.6875] | 0.7042 [0.6875–0.7125] | 13.3 | 26.7 | 23.7 | 5.3 | 0.0 | 0.0 | 0.6349 / 0.6663 | logged OUTSIDE repeat range |
| FW_B3b_s1 | 0.7619 [0.7302–0.7778] | 0.7196 [0.6984–0.7302] | 0.7302 [0.7302–0.7302] | -0.0423 | -0.0317 | 0.8033 [0.7690–0.8250] | 0.7792 [0.7625–0.7875] | 0.7875 [0.7875–0.7875] | 14.0 | 17.7 | 17.0 | 1.0 | 0.0 | 0.0 | 0.7937 / 0.8283 | logged OUTSIDE repeat range |
| FW_B3b_s2 | 0.9048 [0.9048–0.9048] | 0.7143 [0.6984–0.7302] | 0.7037 [0.6984–0.7143] | -0.1905 | -0.2011 | 0.8696 [0.8696–0.8696] | 0.7750 [0.7625–0.7875] | 0.7667 [0.7625–0.7750] | 0.0 | 18.0 | 18.7 | 6.0 | 0.0 | 0.0 | 0.9048 / 0.8696 | logged inside repeat range |
| FW_B3b_s3 | 0.4550 [0.4444–0.4603] | 0.6138 [0.6032–0.6349] | 0.5820 [0.5556–0.6032] | +0.1587 | +0.1270 | 0.5708 [0.5625–0.5750] | 0.6958 [0.6875–0.7125] | 0.6708 [0.6500–0.6875] | 34.3 | 24.3 | 26.3 | 0.0 | 0.0 | 0.0 | 0.4921 / 0.6000 | logged OUTSIDE repeat range |
