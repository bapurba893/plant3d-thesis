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
| B4 | KPConv / DA-0 / L-D | 0.6384 ± 0.1912 | 0.6614 ± 0.0140 | 0.6614 ± 0.0370 | +0.0229 | +0.0229 | 0.6978 ± 0.1207 | 0.7333 ± 0.0110 | 0.7333 ± 0.0292 | 0.8571 / 0.5026 / 0.5556 | 0.6561 / 0.6772 / 0.6508 | 0.6984 / 0.6614 / 0.6243 | 2/3 | 0/3 | 1/3 | 20.9 | 21.3 | 21.3 | 1.9 | 0.0 | 0.0 |

## Oracle rows (A5, B5, C5; upper bound, not pooled): per row (3 seeds)

| Row | Backbone / DA / Aug | Acc P4a | Acc P4 | Acc P4s | Δ acc P4 | Δ acc P4s | Bal. acc P4a | Bal. acc P4 | Bal. acc P4s | Acc P4a per seed | Acc P4 per seed | Acc P4s per seed | Seeds < 0.635 P4a | Seeds < 0.635 P4 | Seeds < 0.635 P4s | T→M P4a | T→M P4 | T→M P4s | M→T P4a | M→T P4 | M→T P4s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B5 | KPConv / DA-O / ALL | 1.0000 ± 0.0000 | 0.8113 ± 0.0291 | 0.8131 ± 0.0500 | -0.1887 | -0.1869 | 1.0000 ± 0.0000 | 0.8514 ± 0.0229 | 0.8528 ± 0.0394 | 1.0000 / 1.0000 / 1.0000 | 0.7778 / 0.8254 / 0.8307 | 0.7566 / 0.8519 / 0.8307 | 0/3 | 0/3 | 0/3 | 0.0 | 11.9 | 11.8 | 0.0 | 0.0 | 0.0 |

## Per run

| Run | Acc P4a | Acc P4 | Acc P4s | Δ acc P4 | Δ acc P4s | Bal. acc P4a | Bal. acc P4 | Bal. acc P4s | T→M P4a | T→M P4 | T→M P4s | M→T P4a | M→T P4 | M→T P4s | Logged FINAL acc / bal | Own-cache check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FW_B4_s1 | 0.8571 [0.8571–0.8571] | 0.6561 [0.6349–0.6667] | 0.6984 [0.6825–0.7143] | -0.2011 | -0.1587 | 0.8351 [0.8321–0.8413] | 0.7292 [0.7125–0.7375] | 0.7625 [0.7500–0.7750] | 3.3 | 21.7 | 19.0 | 5.7 | 0.0 | 0.0 | 0.8413 / 0.8288 | logged OUTSIDE repeat range |
| FW_B4_s2 | 0.5026 [0.4762–0.5238] | 0.6772 [0.6508–0.6984] | 0.6614 [0.6508–0.6825] | +0.1746 | +0.1587 | 0.6083 [0.5875–0.6250] | 0.7458 [0.7250–0.7625] | 0.7333 [0.7250–0.7500] | 31.3 | 20.3 | 21.3 | 0.0 | 0.0 | 0.0 | 0.4921 / 0.6000 | logged inside repeat range |
| FW_B4_s3 | 0.5556 [0.5238–0.5873] | 0.6508 [0.6508–0.6508] | 0.6243 [0.6032–0.6349] | +0.0952 | +0.0688 | 0.6500 [0.6250–0.6750] | 0.7250 [0.7250–0.7250] | 0.7042 [0.6875–0.7125] | 28.0 | 22.0 | 23.7 | 0.0 | 0.0 | 0.0 | 0.5556 / 0.6500 | logged inside repeat range |
| FW_B5_s1 (Oracle) | 1.0000 [1.0000–1.0000] | 0.7778 [0.7619–0.7937] | 0.7566 [0.7460–0.7619] | -0.2222 | -0.2434 | 1.0000 [1.0000–1.0000] | 0.8250 [0.8125–0.8375] | 0.8083 [0.8000–0.8125] | 0.0 | 14.0 | 15.3 | 0.0 | 0.0 | 0.0 | 0.9841 / 0.9875 | logged OUTSIDE repeat range |
| FW_B5_s2 (Oracle) | 1.0000 [1.0000–1.0000] | 0.8254 [0.7937–0.8571] | 0.8519 [0.8254–0.8730] | -0.1746 | -0.1481 | 1.0000 [1.0000–1.0000] | 0.8625 [0.8375–0.8875] | 0.8833 [0.8625–0.9000] | 0.0 | 11.0 | 9.3 | 0.0 | 0.0 | 0.0 | 1.0000 / 1.0000 | logged inside repeat range |
| FW_B5_s3 (Oracle) | 1.0000 [1.0000–1.0000] | 0.8307 [0.8095–0.8571] | 0.8307 [0.7778–0.8571] | -0.1693 | -0.1693 | 1.0000 [1.0000–1.0000] | 0.8667 [0.8500–0.8875] | 0.8667 [0.8250–0.8875] | 0.0 | 10.7 | 10.7 | 0.0 | 0.0 | 0.0 | 1.0000 / 1.0000 | logged inside repeat range |
