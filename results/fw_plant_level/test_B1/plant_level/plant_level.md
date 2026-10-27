# FW plant-level accuracy and plant-level bootstrap

Written by `scripts/17_fw_plant_level.py` from: `results/fw_plant_level/test_B1/fw_cross_eval_clouds.csv`

**Warning: 4 held-out plants are too few for these bootstrap intervals to be reliable: resampling 4 plants with replacement gives only 35 distinct resamples (10 of them contain a single species), so the intervals are coarse and do not reach their nominal 95% coverage. They show how much each result depends on which plants are in the test set, not how a new set of plants would score.**

Held-out plants: M03 (Maize, 11 clouds), M04 (Maize, 12 clouds), T01 (Tomato, 20 clouds), T03 (Tomato, 20 clouds).

Per-plant accuracy = fraction of that plant's clouds predicted correctly, per seed (mean of the evaluation repeats for PointNet++/KPConv), then mean ± sample SD over seeds. Row mean = all clouds pooled, mean over seeds. Bootstrap: resample the plants with replacement (each drawn plant brings all of its clouds), 10000 resamples, percentile 95% interval, numpy seed 0.

## Cache P4a

| Row | M03 (Maize) | M04 (Maize) | T01 (Tomato) | T03 (Tomato) | Row mean (pooled) | Plant-bootstrap 95% interval | Seeds |
|---|---|---|---|---|---|---|---|
| B1 | 0.949 ± 0.046 | 0.935 ± 0.058 | 0.606 ± 0.111 | 0.511 ± 0.129 | 0.698 | [0.558, 0.942] | 3 |

## Cache P4

| Row | M03 (Maize) | M04 (Maize) | T01 (Tomato) | T03 (Tomato) | Row mean (pooled) | Plant-bootstrap 95% interval | Seeds |
|---|---|---|---|---|---|---|---|
| B1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.589 ± 0.059 | 0.494 ± 0.038 | 0.709 | [0.542, 1.000] | 3 |

