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
| B4 | KPConv / DA-0 / L-D | 0.6384 ± 0.1912 | 0.6614 ± 0.0140 | +0.0229 | 0.6978 ± 0.1207 | 0.7333 ± 0.0110 | +0.0355 | 0.6561 / 0.6772 / 0.6508 | 0/3 | 20.9 / 21.3 | 1.9 / 0.0 |

## Oracle rows (A5, B5, C5; upper bound, not pooled): per row (3 seeds)

| Row | Backbone / DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Δ bal. acc | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B5 | KPConv / DA-O / ALL | 1.0000 ± 0.0000 | 0.8113 ± 0.0291 | -0.1887 | 1.0000 ± 0.0000 | 0.8514 ± 0.0229 | -0.1486 | 0.7778 / 0.8254 / 0.8307 | 0/3 | 0.0 / 11.9 | 0.0 / 0.0 |

## Per run

| Run | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | T→M P4a / P4 | M→T P4a / P4 | Logged FINAL acc / bal | Own-cache check |
|---|---|---|---|---|---|---|---|---|---|
| FW_B4_s1 | 0.8571 [0.8571–0.8571] | 0.6561 [0.6349–0.6667] | -0.2011 | 0.8351 [0.8321–0.8413] | 0.7292 [0.7125–0.7375] | 3.3 / 21.7 | 5.7 / 0.0 | 0.8413 / 0.8288 | logged OUTSIDE repeat range |
| FW_B4_s2 | 0.5026 [0.4762–0.5238] | 0.6772 [0.6508–0.6984] | +0.1746 | 0.6083 [0.5875–0.6250] | 0.7458 [0.7250–0.7625] | 31.3 / 20.3 | 0.0 / 0.0 | 0.4921 / 0.6000 | logged inside repeat range |
| FW_B4_s3 | 0.5556 [0.5238–0.5873] | 0.6508 [0.6508–0.6508] | +0.0952 | 0.6500 [0.6250–0.6750] | 0.7250 [0.7250–0.7250] | 28.0 / 22.0 | 0.0 / 0.0 | 0.5556 / 0.6500 | logged inside repeat range |
| FW_B5_s1 (Oracle) | 1.0000 [1.0000–1.0000] | 0.7778 [0.7619–0.7937] | -0.2222 | 1.0000 [1.0000–1.0000] | 0.8250 [0.8125–0.8375] | 0.0 / 14.0 | 0.0 / 0.0 | 0.9841 / 0.9875 | logged OUTSIDE repeat range |
| FW_B5_s2 (Oracle) | 1.0000 [1.0000–1.0000] | 0.8254 [0.7937–0.8571] | -0.1746 | 1.0000 [1.0000–1.0000] | 0.8625 [0.8375–0.8875] | 0.0 / 11.0 | 0.0 / 0.0 | 1.0000 / 1.0000 | logged inside repeat range |
| FW_B5_s3 (Oracle) | 1.0000 [1.0000–1.0000] | 0.8307 [0.8095–0.8571] | -0.1693 | 1.0000 [1.0000–1.0000] | 0.8667 [0.8500–0.8875] | 0.0 / 10.7 | 0.0 / 0.0 | 1.0000 / 1.0000 | logged inside repeat range |
