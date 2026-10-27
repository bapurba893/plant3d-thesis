# Full-workflow rerun (FW): plant-level evaluation, checkpoint reporting, and the P4s cache

Scope: three additions to the FW evaluation of the 54 selected A–C checkpoints (18 rows × seeds
1, 2, 3; C1 = `PP_P4a_s<seed>`), all trained on the soil-kept preprocessing P4a. Nothing was
retrained.

1. **Plant-level evaluation**: per-cloud predictions, per-plant accuracy for every row, and a
   bootstrap over plants (section 2).
2. **Checkpoint reporting**: final-epoch, best-epoch, validation-selected and full-run-mean target
   accuracy from the training logs (section 3).
3. **The P4s cache**: the same checkpoints evaluated on a third target cache (section 4).

Every statement is labelled **[measured]** (read from a file or command output named here) or
**[inferred]** (an interpretation, not tested).

Three standing caveats, which apply to every number below:

- **The models were trained with soil.** [measured] Every checkpoint was trained on P4a, which
  keeps soil and pot (`step_notes/FW_1_Results.md`, `FW_2_Results.md`). Evaluating on P4 or P4s
  shows what happens when soil is taken away at test time. It does not show what a model trained
  without soil could do.
- **4 plants are too few for reliable intervals.** [measured] The held-out set is 63 clouds from
  4 plants. Resampling 4 plants with replacement gives only 35 distinct resamples, and 10 of them
  contain a single species. [inferred] The bootstrap intervals are therefore coarse and do not
  reach their nominal 95% coverage. They show how much a result depends on which plants are in
  the test set, not how a new set of plants would score.
- **P4s keeps the P4a centre and scale, but still changes point density.** [measured] See
  section 4.

## 0. Code and jobs

- `scripts/16_fw_cross_eval.py` (changed):
  - New output `fw_cross_eval_clouds.csv`: one line per run × cache × repeat × held-out cloud,
    with run, seed, cloud id, plant id, species, scan date, true label, predicted label and
    correct. The predictions come from each training script's own `evaluate_cls_only`, which
    already returned them; no model code changed.
  - Each cloud is paired with its row of `data/pheno4d_heldout_eval.csv` (the plant-id mapping)
    by position. The script stops if the file name or label at any position does not match.
  - New `--cache` option: a cache name under `data/preprocessed_variants/` or a folder path. The
    first cache listed is the reference for the Δ columns. New `--seeds` option.
  - The per-run Δ column is now called `d_acc_<cache>` (it was `d_acc`).
  - Kept unchanged: 3 evaluation repeats for PointNet++ and KPConv (their evaluation is not
    repeatable), 1 for DGCNN.
- `scripts/17_fw_plant_level.py` (new): per-plant accuracy and the plant bootstrap.
- `scripts/18_fw_epoch_accuracy.py` (new): checkpoint reporting from run.log. It reuses
  `15_fw_summarize.py`'s parser, which is not modified.
- `jobs/fw_plant_level.sbatch` (new): a 6-task array.

| Job | What | Outcome |
|---|---|---|
| 366306 (task 3 only, rows = B1) | single-row test on caches P4a and P4 | finished; all 18 evaluations identical to the earlier B1 evaluation (job 361787) |
| 366319 (tasks 1–5, A and B rows) | caches P4a, P4, P4s | all tasks `FAILED` (exit 1) from the run.log self-check only; every output file written. Times: 4m07s, 4m08s, 15m10s, 14m29s, 12m21s |
| 366332 (task 6, C rows) | caches P4a, P4, P4s | `COMPLETED`; all checks passed. Task 6 was submitted separately because of the dgx QOS limit of 5 submitted jobs per user |

Checks [measured]:
- The 216 A/B evaluations on P4a and P4 are identical to `results/fw_cross_eval/stage2a_pointnet2_v2`
  and `stage2b_kpconv_v2`: same accuracy and the same confusion counts.
- The run.log check fails for 12 of the 36 A/B runs. These are the same 12 runs as in
  `FW_4_Results.md`, each 1 or 2 clouds outside the 3-repeat range.
- DGCNN matches every logged FINAL value, and C1 matches Block PP on P4a and P4.
- In every evaluation, the accuracy recomputed from the per-cloud lines equals the summary
  accuracy.

Outputs: `results/fw_plant_level/P4a_P4_P4s/task<n>_<rows>/` and
`results/fw_plant_level/P4a_P4_P4s/plant_level/` (`plant_level.md`, `plant_level_per_plant.csv`,
`plant_level_bootstrap.csv`). The single-row test is in `results/fw_plant_level/test_B1/`.

## 1. Held-out plants

[measured] From `data/pheno4d_heldout_eval.csv` (columns `plant_id`, `species`):

| Plant | Species | Clouds |
|---|---|---|
| M03 | Maize | 11 |
| M04 | Maize | 12 |
| T01 | Tomato | 20 |
| T03 | Tomato | 20 |

## 2. Plant-level evaluation

Definitions:
- **Per-plant accuracy** = the fraction of that plant's clouds predicted correctly in one seed
  (mean of the 3 repeats for PointNet++/KPConv), then mean ± sample SD over the 3 seeds.
- **Row mean** = all 63 clouds pooled, mean over seeds. This is the same number as the "Acc"
  column of `fw_cross_eval.md`.
- **Bootstrap** = the 4 plants resampled with replacement, each drawn plant bringing all of its
  clouds. 10,000 resamples, percentile 95% interval, numpy seed 0.

**Warning (written into every output of script 17):** 4 held-out plants are too few for these
bootstrap intervals to be reliable. Resampling 4 plants with replacement gives only 35 distinct
resamples (10 of them contain a single species), so the intervals are coarse and do not reach
their nominal 95% coverage. They show how much each result depends on which plants are in the
test set, not how a new set of plants would score.

### 2a. Cache P4a (training cache, soil kept) [measured]

| Row | M03 (Maize) | M04 (Maize) | T01 (Tomato) | T03 (Tomato) | Row mean | Plant bootstrap 95% |
|---|---|---|---|---|---|---|
| A1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.156 ± 0.059 | 0.044 ± 0.077 | 0.429 | [0.100, 1.000] |
| A2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.389 ± 0.267 | 0.128 ± 0.125 | 0.529 | [0.258, 1.000] |
| A3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.750 ± 0.433 | 0.494 ± 0.394 | 0.760 | [0.573, 1.000] |
| A4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.256 ± 0.160 | 0.194 ± 0.168 | 0.508 | [0.225, 1.000] |
| A5 (Oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.994 ± 0.010 | 0.989 ± 0.019 | 0.995 | [0.991, 1.000] |
| A6 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.650 ± 0.240 | 0.522 ± 0.123 | 0.737 | [0.586, 1.000] |
| B1 | 0.949 ± 0.046 | 0.935 ± 0.058 | 0.606 ± 0.111 | 0.511 ± 0.129 | 0.698 | [0.558, 0.942] |
| B2 | 0.818 ± 0.000 | 0.833 ± 0.000 | 0.522 ± 0.067 | 0.656 ± 0.082 | 0.675 | [0.568, 0.826] |
| B3 | 0.879 ± 0.091 | 0.861 ± 0.096 | 0.606 ± 0.236 | 0.494 ± 0.113 | 0.667 | [0.550, 0.870] |
| B3b | 0.909 ± 0.157 | 0.889 ± 0.127 | 0.628 ± 0.435 | 0.567 ± 0.433 | 0.707 | [0.597, 0.899] |
| B4 | 0.909 ± 0.157 | 0.926 ± 0.128 | 0.461 ± 0.372 | 0.494 ± 0.395 | 0.638 | [0.478, 0.918] |
| B5 (Oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | [1.000, 1.000] |
| C1 | 0.788 ± 0.052 | 0.806 ± 0.048 | 0.967 ± 0.058 | 0.900 ± 0.173 | 0.884 | [0.797, 0.940] |
| C2 | 0.848 ± 0.139 | 0.861 ± 0.127 | 0.850 ± 0.260 | 0.683 ± 0.506 | 0.799 | [0.713, 0.857] |
| C3 | 0.848 ± 0.052 | 0.889 ± 0.048 | 1.000 ± 0.000 | 0.933 ± 0.058 | 0.931 | [0.870, 0.981] |
| C4 | 0.848 ± 0.052 | 0.889 ± 0.048 | 0.533 ± 0.029 | 0.517 ± 0.257 | 0.651 | [0.525, 0.870] |
| C5 (Oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.850 ± 0.132 | 0.952 | [0.875, 1.000] |
| C6 | 0.848 ± 0.052 | 0.889 ± 0.048 | 1.000 ± 0.000 | 0.950 ± 0.087 | 0.937 | [0.870, 0.981] |

### 2b. Cache P4 (soil removed, re-centred and re-scaled on the plant) [measured]

| Row | M03 (Maize) | M04 (Maize) | T01 (Tomato) | T03 (Tomato) | Row mean | Plant bootstrap 95% |
|---|---|---|---|---|---|---|
| A1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.528 ± 0.051 | 0.333 ± 0.117 | 0.638 | [0.431, 1.000] |
| A2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.361 ± 0.025 | 0.128 ± 0.035 | 0.520 | [0.244, 1.000] |
| A3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.489 ± 0.323 | 0.344 ± 0.320 | 0.630 | [0.417, 1.000] |
| A4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.289 ± 0.217 | 0.317 ± 0.280 | 0.557 | [0.303, 1.000] |
| A5 (Oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.856 ± 0.126 | 0.717 ± 0.029 | 0.864 | [0.761, 1.000] |
| A6 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.550 ± 0.083 | 0.389 ± 0.079 | 0.663 | [0.469, 1.000] |
| B1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.589 ± 0.059 | 0.494 ± 0.038 | 0.709 | [0.542, 1.000] |
| B2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.183 ± 0.115 | 0.100 ± 0.145 | 0.455 | [0.142, 1.000] |
| B3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.494 ± 0.086 | 0.328 ± 0.199 | 0.626 | [0.411, 1.000] |
| B3b | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.528 ± 0.142 | 0.472 ± 0.051 | 0.683 | [0.500, 1.000] |
| B4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.422 ± 0.025 | 0.511 ± 0.059 | 0.661 | [0.467, 1.000] |
| B5 (Oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.839 ± 0.079 | 0.567 ± 0.017 | 0.811 | [0.635, 1.000] |
| C1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.567 ± 0.115 | 0.333 ± 0.189 | 0.651 | [0.437, 1.000] |
| C2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.333 ± 0.252 | 0.317 ± 0.284 | 0.571 | [0.325, 1.000] |
| C3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.400 ± 0.050 | 0.183 ± 0.153 | 0.550 | [0.292, 1.000] |
| C4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.050 ± 0.050 | 0.067 ± 0.076 | 0.402 | [0.058, 1.000] |
| C5 (Oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.933 ± 0.029 | 0.550 ± 0.087 | 0.836 | [0.625, 1.000] |
| C6 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.433 ± 0.058 | 0.150 ± 0.180 | 0.550 | [0.282, 1.000] |

### 2c. Cache P4s (soil removed, P4a centre and scale kept) [measured]

| Row | M03 (Maize) | M04 (Maize) | T01 (Tomato) | T03 (Tomato) | Row mean | Plant bootstrap 95% |
|---|---|---|---|---|---|---|
| A1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.550 ± 0.044 | 0.439 ± 0.019 | 0.679 | [0.494, 1.000] |
| A2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.406 ± 0.067 | 0.194 ± 0.010 | 0.556 | [0.300, 1.000] |
| A3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.467 ± 0.332 | 0.367 ± 0.333 | 0.630 | [0.417, 1.000] |
| A4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.283 ± 0.233 | 0.267 ± 0.220 | 0.540 | [0.275, 1.000] |
| A5 (Oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.844 ± 0.150 | 0.628 ± 0.063 | 0.832 | [0.685, 1.000] |
| A6 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.583 ± 0.000 | 0.344 ± 0.086 | 0.660 | [0.446, 1.000] |
| B1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.583 ± 0.076 | 0.489 ± 0.079 | 0.705 | [0.536, 1.000] |
| B2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.217 ± 0.145 | 0.061 ± 0.106 | 0.453 | [0.139, 1.000] |
| B3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.544 ± 0.051 | 0.333 ± 0.196 | 0.644 | [0.437, 1.000] |
| B3b | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.506 ± 0.155 | 0.461 ± 0.096 | 0.672 | [0.483, 1.000] |
| B4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.439 ± 0.067 | 0.494 ± 0.067 | 0.661 | [0.467, 1.000] |
| B5 (Oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.867 ± 0.117 | 0.544 ± 0.042 | 0.813 | [0.620, 1.000] |
| C1 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.217 ± 0.076 | 0.117 ± 0.161 | 0.471 | [0.167, 1.000] |
| C2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.133 ± 0.153 | 0.150 ± 0.132 | 0.455 | [0.142, 1.000] |
| C3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.117 ± 0.029 | 0.067 ± 0.076 | 0.423 | [0.092, 1.000] |
| C4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.050 ± 0.050 | 0.017 ± 0.029 | 0.386 | [0.033, 1.000] |
| C5 (Oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.600 ± 0.180 | 0.400 ± 0.000 | 0.683 | [0.493, 1.000] |
| C6 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.117 ± 0.076 | 0.067 ± 0.076 | 0.423 | [0.092, 1.000] |

### 2d. Findings

- **Without soil, every model gets every maize cloud right.** [measured] On P4 and P4s, M03 and
  M04 are 1.000 ± 0.000 for all 18 rows. All differences between rows on these caches come from
  the two tomato plants.
- **Most of these intervals have an upper bound of 1.000, which tells us little.** [measured]
  That covers every row on P4 and P4s, and every P4a row with perfect maize (all PointNet++ rows
  and the Oracle rows). [inferred] Any resample made only of maize plants scores 1.0 for such a
  row. This is the single-species resampling the warning refers to.
- **With soil (P4a), DGCNN errs differently from the other two backbones.** [measured] C1, C3
  and C6 are 0.90–1.00 on the tomato plants but 0.79–0.89 on the maize plants. PointNet++ is the
  reverse: maize 1.000, tomato mostly 0.04–0.75. The only non-Oracle rows with intervals that
  exclude the always-tomato level of 0.635 on P4a are DGCNN rows: C1 [0.797, 0.940],
  C2 [0.713, 0.857], C3 [0.870, 0.981] and C6 [0.870, 0.981].
- **The two tomato plants differ.** [measured] T03 is below T01 in most rows and caches (for
  example C1 on P4: 0.333 vs 0.567). [inferred] With only two plants per species, one hard plant
  moves the row mean a lot, which is part of why the intervals are wide.

## 3. Checkpoint reporting (script 18)

[measured] From the 54 FW run.logs (`results/fw_epoch_accuracy/`). Target = Pheno4D held-out, 63
clouds, logged during training on P4a. Mean ± SD over 3 seeds.

- **Best epoch uses the target test labels to choose the epoch.** It is an upper bound (an oracle
  choice of epoch), not a result any method could report.
- **Validation-selected** = the checkpoint at the lowest source validation loss (Oracle rows:
  Oracle validation loss), chosen without target labels.

| Row | Final epoch | Best epoch (uses target labels: upper bound) | Validation-selected | Full-run mean | Best − selected |
|---|---|---|---|---|---|
| A1 | 0.4286 ± 0.0550 | 0.9471 ± 0.0243 | 0.4233 ± 0.0330 | 0.4968 ± 0.0202 | 0.5238 |
| A2 | 0.5027 ± 0.1475 | 0.9418 ± 0.0330 | 0.5132 ± 0.1191 | 0.5651 ± 0.0674 | 0.4286 |
| A3 | 0.6614 ± 0.2737 | 0.9629 ± 0.0367 | 0.7725 ± 0.2586 | 0.7313 ± 0.1024 | 0.1904 |
| A4 | 0.5132 ± 0.1283 | 0.9048 ± 0.0728 | 0.5185 ± 0.1081 | 0.5436 ± 0.0387 | 0.3862 |
| A5 (Oracle) | 0.9894 ± 0.0092 | 1.0000 ± 0.0000 | 0.9894 ± 0.0092 | 0.9275 ± 0.0113 | 0.0106 |
| A6 | 0.7725 ± 0.1350 | 0.9471 ± 0.0330 | 0.7725 ± 0.0783 | 0.6961 ± 0.0699 | 0.1746 |
| B1 | 0.6666 ± 0.0420 | 0.9365 ± 0.0000 | 0.7196 ± 0.0601 | 0.6729 ± 0.0107 | 0.2169 |
| B2 | 0.6613 ± 0.0733 | 0.9365 ± 0.0000 | 0.6720 ± 0.0242 | 0.6844 ± 0.0631 | 0.2645 |
| B3 | 0.6349 ± 0.0824 | 0.9312 ± 0.0092 | 0.6455 ± 0.0799 | 0.7020 ± 0.0196 | 0.2857 |
| B3b | 0.7672 ± 0.2247 | 0.9418 ± 0.0092 | 0.7302 ± 0.2136 | 0.8316 ± 0.1053 | 0.2116 |
| B4 | 0.8519 ± 0.0661 | 0.9524 ± 0.0159 | 0.6297 ± 0.1860 | 0.7866 ± 0.0402 | 0.3227 |
| B5 (Oracle) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9947 ± 0.0092 | 0.9543 ± 0.0059 | 0.0053 |
| C1 | 0.8942 ± 0.0184 | 0.9524 ± 0.0275 | 0.8836 ± 0.0367 | 0.8632 ± 0.0155 | 0.0688 |
| C2 | 0.4656 ± 0.0399 | 0.9418 ± 0.0243 | 0.7989 ± 0.1972 | 0.6429 ± 0.0471 | 0.1429 |
| C3 | 0.9312 ± 0.0331 | 0.9683 ± 0.0000 | 0.9312 ± 0.0242 | 0.8525 ± 0.0121 | 0.0371 |
| C4 | 0.5661 ± 0.0601 | 0.9100 ± 0.0330 | 0.6508 ± 0.0840 | 0.6317 ± 0.0279 | 0.2593 |
| C5 (Oracle) | 0.9524 ± 0.0420 | 1.0000 ± 0.0000 | 0.9524 ± 0.0420 | 0.8914 ± 0.0044 | 0.0476 |
| C6 | 0.9206 ± 0.0420 | 0.9577 ± 0.0184 | 0.9365 ± 0.0420 | 0.8687 ± 0.0410 | 0.0212 |

Per-run values, with the epoch of each number, are in `results/fw_epoch_accuracy/fw_epoch_accuracy.md`.

- **The full-run means match the published FW values.** [measured] For example C1 0.8632, B3b
  0.8316 and B1 0.6729, as in `FW_1_Results.md` and `FW_2_Results.md`.
- **Every non-Oracle row reaches at least 0.90 at some epoch** (best epoch 0.90–0.97). [measured]
  The validation-selected checkpoint is far below that for most rows: the gap is 0.52 for A1,
  0.43 for A2 and 0.32 for B4, and only 0.02–0.07 for C1, C3 and C6. [inferred] So most rows do
  pass through a good epoch, but source-validation selection cannot find it. This agrees with the
  earlier finding that the selection signal is unreliable under domain shift (CLAUDE.md, final
  summary point 4). The best-epoch numbers must never be reported as results.
- **Final epoch vs validation-selected.** [measured] These are within about 0.1 for most rows.
  The two clear exceptions go in opposite directions: B4's final epoch (0.8519) is well above its
  validation-selected checkpoint (0.6297), and C2's final epoch (0.4656) is well below it (0.7989).

## 4. The P4s evaluation

### 4a. What P4s is

All of the following is [measured] by comparing `data/preprocessed_variants/{P4a,P4s,P4}/preprocessed_manifest.csv`:

- 531 clouds, 4096 points each, with the same voxel rule (height/200) and FPS sampling as P4a
  and P4. All 63 held-out clouds and the Crops3D train/val clouds resolve.
- **Same centre and scale as P4a.** Across all 531 clouds, `norm_scale` and `norm_center_x/y/z`
  differ from P4a by at most 3e-14. P4, for comparison, differs by up to 74 in scale.
- **Soil removed, as in P4.** On all 63 held-out clouds, `n_soil_pot_removed` > 0 in P4s and P4,
  and 0 in P4a.
- **Point density still changes.** On the held-out clouds, the median number of points before
  FPS is 107,873 in P4a and 37,591 in P4s (the same as P4). The 4096 FPS points are therefore
  drawn from the plant only, so the plant is sampled more densely than in P4a. `voxel_size_mm`
  also differs from P4a by up to 0.35 mm, the same difference P4 shows.
- **How P4s was built is not recorded here.** No script in the repo mentions P4s other than
  `16_fw_cross_eval.py`'s usage example. [inferred] It was presumably generated outside this
  session; its build steps should be recorded next to the other variants.

[inferred] So P4s separates "soil removed" from "re-centred and re-scaled", but not from "point
density changed". A P4a → P4s difference is the soil plus the density change, not the soil alone.

### 4b. Accuracy per row (3-seed mean ± SD, selected checkpoints) [measured]

Δ = cache − P4a. "Below 0.635" = seeds below the always-tomato level on P4s. T→M and M→T are the
mean number of P4s clouds misclassified as each species (of 40 tomato and 23 maize).

| Row | Acc P4a | Acc P4 | Acc P4s | Δ P4 | Δ P4s | P4s per seed | Below 0.635 (P4s) | T→M P4s | M→T P4s |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 0.4286 ± 0.0420 | 0.6384 ± 0.0291 | 0.6790 ± 0.0133 | +0.2099 | +0.2504 | 0.6931 / 0.6772 / 0.6667 | 0/3 | 20.2 | 0.0 |
| A2 | 0.5291 ± 0.1244 | 0.5203 ± 0.0186 | 0.5556 ± 0.0212 | −0.0088 | +0.0265 | 0.5344 / 0.5767 / 0.5556 | 3/3 | 28.0 | 0.0 |
| A3 | 0.7601 ± 0.2610 | 0.6296 ± 0.2021 | 0.6296 ± 0.2090 | −0.1305 | −0.1305 | 0.7143 / 0.3915 / 0.7831 | 1/3 | 23.3 | 0.0 |
| A4 | 0.5079 ± 0.1041 | 0.5573 ± 0.1565 | 0.5397 ± 0.1440 | +0.0494 | +0.0317 | 0.3757 / 0.6455 / 0.5979 | 2/3 | 29.0 | 0.0 |
| A5 (Oracle) | 0.9947 ± 0.0053 | 0.8642 ± 0.0492 | 0.8325 ± 0.0670 | −0.1305 | −0.1623 | 0.9048 / 0.8201 / 0.7725 | 0/3 | 10.6 | 0.0 |
| A6 | 0.7372 ± 0.1123 | 0.6631 ± 0.0401 | 0.6596 ± 0.0272 | −0.0741 | −0.0776 | 0.6667 / 0.6825 / 0.6296 | 1/3 | 21.4 | 0.0 |
| B1 | 0.6984 ± 0.0399 | 0.7090 ± 0.0295 | 0.7055 ± 0.0492 | +0.0106 | +0.0071 | 0.7460 / 0.6508 / 0.7196 | 0/3 | 18.6 | 0.0 |
| B2 | 0.6755 ± 0.0456 | 0.4550 ± 0.0826 | 0.4533 ± 0.0796 | −0.2205 | −0.2222 | 0.4127 / 0.5450 / 0.4021 | 3/3 | 34.4 | 0.0 |
| B3 | 0.6667 ± 0.0881 | 0.6261 ± 0.0680 | 0.6437 ± 0.0480 | −0.0406 | −0.0229 | 0.6085 / 0.6984 / 0.6243 | 2/3 | 22.4 | 0.0 |
| B3b | 0.7072 ± 0.2298 | 0.6825 ± 0.0596 | 0.6720 ± 0.0790 | −0.0247 | −0.0353 | 0.7302 / 0.7037 / 0.5820 | 1/3 | 20.7 | 0.0 |
| B4 | 0.6384 ± 0.1912 | 0.6614 ± 0.0140 | 0.6614 ± 0.0370 | +0.0229 | +0.0229 | 0.6984 / 0.6614 / 0.6243 | 1/3 | 21.3 | 0.0 |
| B5 (Oracle) | 1.0000 ± 0.0000 | 0.8113 ± 0.0291 | 0.8131 ± 0.0500 | −0.1887 | −0.1869 | 0.7566 / 0.8519 / 0.8307 | 0/3 | 11.8 | 0.0 |
| C1 | 0.8836 ± 0.0367 | 0.6508 ± 0.0966 | 0.4709 ± 0.0733 | −0.2328 | −0.4127 | 0.4286 / 0.5556 / 0.4286 | 3/3 | 33.3 | 0.0 |
| C2 | 0.7989 ± 0.1972 | 0.5714 ± 0.1672 | 0.4550 ± 0.0874 | −0.2275 | −0.3439 | 0.3651 / 0.4603 / 0.5397 | 3/3 | 34.3 | 0.0 |
| C3 | 0.9312 ± 0.0242 | 0.5503 ± 0.0601 | 0.4233 ± 0.0330 | −0.3810 | −0.5079 | 0.3968 / 0.4127 / 0.4603 | 3/3 | 36.3 | 0.0 |
| C4 | 0.6508 ± 0.0840 | 0.4021 ± 0.0330 | 0.3862 ± 0.0242 | −0.2487 | −0.2646 | 0.3651 / 0.3810 / 0.4127 | 3/3 | 38.7 | 0.0 |
| C5 (Oracle) | 0.9524 ± 0.0420 | 0.8360 ± 0.0242 | 0.6825 ± 0.0572 | −0.1164 | −0.2698 | 0.6190 / 0.6984 / 0.7302 | 1/3 | 20.0 | 0.0 |
| C6 | 0.9365 ± 0.0420 | 0.5503 ± 0.0750 | 0.4233 ± 0.0458 | −0.3862 | −0.5132 | 0.3968 / 0.3968 / 0.4762 | 3/3 | 36.3 | 0.0 |

### 4c. Findings

- **On P4s no model calls any maize cloud tomato.** [measured] M→T is 0.0 for every row, the
  same as on P4. Every P4s error is a tomato cloud predicted as maize.
- **DGCNN falls further on P4s than on P4.** [measured] All six DGCNN rows are lower on P4s than
  on P4. C1 drops from 0.8836 (P4a) to 0.6508 (P4) to 0.4709 (P4s), and C3 and C6 drop to
  0.4233. All non-Oracle DGCNN rows have all 3 seeds below the always-tomato level on P4s.
- **PointNet++ and KPConv are about the same on P4s as on P4.** [measured] Every A and B row is
  within 0.04 of its P4 value, except A1 (+0.041).
- [inferred, not tested] Keeping the P4a centre and scale does not soften DGCNN's drop; it
  deepens it. So DGCNN's soil-kept accuracy does not depend only on the re-scaling that P4 adds.
  What remains in P4s is the soil removal plus the density change (section 4a), and these two are
  not separated. One reading: DGCNN relies on soil or point-density cues more than the other two
  backbones do. That would need, for example, a cache with soil kept but density matched, or a
  model trained without soil.
- **Few rows score higher on P4s than on P4a.** [measured] The five that do are A1
  (0.4286 → 0.6790), A2 (+0.0265), A4 (+0.0317), B1 (+0.0071) and B4 (+0.0229). Every A1, A2 and
  A4 seed is still below 0.70 on P4s. [inferred] A1's rise matches the change of bias FW_4
  described for P4 (with soil, A1 called almost every tomato maize), not a real improvement.

## 5. Not done yet

- A DGCNN C1 trained on P4s (proposed separately; not run at the time of writing).
- A cache that removes soil without changing point density, to separate those two effects (4a).
- The evaluation pooled over all rows of one backbone, or a bootstrap that resamples within
  species (with 2 plants per species, it would have only 3 distinct resamples per species).
