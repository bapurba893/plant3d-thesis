# Full-workflow rerun (FW): soil-removed cross-evaluation of the A–C rows

Scope: the fourth part of the full-workflow rerun described in `jobs/fw/README.md`. Every FW A–C
row (A1–A6, B1–B5 plus B3b, C1–C6; 3 seeds each, 54 runs) was trained on the soil-kept
preprocessing P4a (`FW_1_Results.md`, `FW_2_Results.md`). Here each run's **selected checkpoint**
is evaluated, without retraining, on the 63 Pheno4D held-out clouds (40 tomato, 23 maize) of two
target caches: **P4a** (its own cache, soil and pot kept) and **P4** (soil removed). The question
is how much of the P4a target accuracy depends on the soil/pot cue (`PP_3_Results.md`;
`FW_1_Results.md` section 3.3). Block D is not part of this test.

Every number below is recomputed from `results/fw_cross_eval/*/fw_cross_eval_rows.csv` and
`fw_cross_eval_runs.csv` (written by `scripts/16_fw_cross_eval.py`) by a throwaway script that was
not committed. Majority-class reference (always Tomato) = 40/63 = 0.635. SD = sample SD over 3
seeds.

## Jobs

All on Prajna (`dgx`), logs in `results/logs/` (not tracked).

| Job | Rows | Job file | Output folder | Outcome |
|---|---|---|---|---|
| 361654 | stage 1: C1–C6 (DGCNN) | `jobs/fw_cross_eval.sbatch` | `results/fw_cross_eval/stage1_dgcnn/` | COMPLETED, 3 min 45 s, "all checks passed" |
| 361785 | stage 2a: A1–A6 (PointNet++) | `jobs/fw_cross_eval.sbatch` | `results/fw_cross_eval/stage2a_pointnet2_v2/` | FAILED (exit 1), 5 min 20 s, 18/18 runs written |
| 361787 (array, 3 tasks) | stage 2b: task 1 B1, B2 (guard on B1 s1); task 2 B3, B3b; task 3 B4, B5 | `jobs/fw_cross_eval_kpconv_array.sbatch` | `results/fw_cross_eval/stage2b_kpconv_v2/task{1_B1_B2,2_B3_B3b,3_B4_B5}/` | each task FAILED (exit 1), 10–12 min, 6/6 runs written per task |

Earlier attempts: 361669 (PointNet++ stage 2a, `--time=01:00:00`) timed out after 8 of 18 runs, and
the KPConv stage-2b job 361676 (`--dependency=afterok:361669`, all six B rows in one job) was
cancelled before it started. They were replaced by
361785 and the array 361787. The 361669 outputs were not kept: `results/fw_cross_eval/stage2a_pointnet2/`
is empty and is not committed.

**Why the state is FAILED.** `scripts/16_fw_cross_eval.py` writes all four output files after every
run, prints the tables, and then exits with code 1 if any run failed the own-cache check (section 2).
All four FAILED jobs/tasks ended this way: no traceback in any log, and every run's line is in
the CSVs (18 + 6 + 6 + 6).

## How it was done

- **Checkpoint**: `results/FW_<row>_s<seed>/model.pt` (C1: `results/PP_P4a_s<seed>/model.pt`, which
  is FW C1), i.e. the model each training script saved at its lowest source validation loss
  (Oracle rows: lowest Oracle validation loss). Target labels were never used for selection.
- **Caches**: P4a and P4, same 63 clouds in the same order (checked per run,
  `data/pheno4d_heldout_eval.csv`).
- Model construction, target DataLoader settings and the evaluation function are taken from each
  backbone's own training script; dropout and test batch size are read from each run's `run.log`.
- **Repeats**: PointNet++ (random farthest-point-sampling start) and KPConv (random
  subsampling-grid rotation) evaluation is not repeatable, so A and B runs were evaluated **3 times
  per cache** (torch/numpy seeded with the repeat index 0, 1, 2); the per-run value is the mean of
  the 3 repeats. DGCNN: **one evaluation** per cache.
- **KPConv neighbourhood limits**: the training-time limits printed in each run's own `run.log`;
  **not recalibrated** on P4a or P4.
- Run on the GPU (A100), as the training runs were.

---

## 1. Per-row tables (selected checkpoint, 3 seeds)

Acc and balanced acc = mean ± SD over 3 seeds. Δ acc = P4 − P4a. T→M = tomato clouds predicted
maize, M→T = maize clouds predicted tomato (3-seed mean; for A/B each seed is the mean of its 3
repeats). "Seeds < 0.635 on P4" counts seeds below the always-Tomato reference.

### PointNet++

| Row | DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | DA-0 / ALL | 0.4286 ± 0.0420 | 0.6384 ± 0.0291 | +0.2099 | 0.5500 ± 0.0331 | 0.7153 ± 0.0229 | 0.6402 / 0.6085 / 0.6667 | 1/3 | 36.0 / 22.8 | 0.0 / 0.0 |
| A2 | DA-A / ALL | 0.5291 ± 0.1244 | 0.5203 ± 0.0186 | -0.0088 | 0.6292 ± 0.0980 | 0.6222 ± 0.0146 | 0.5026 / 0.5397 / 0.5185 | 3/3 | 29.7 / 30.2 | 0.0 / 0.0 |
| A3 | DA-S / ALL | 0.7601 ± 0.2610 | 0.6296 ± 0.2021 | -0.1305 | 0.8111 ± 0.2055 | 0.7083 ± 0.1592 | 0.6984 / 0.4021 / 0.7884 | 1/3 | 15.1 / 23.3 | 0.0 / 0.0 |
| A4 | DA-A / L-N | 0.5079 ± 0.1041 | 0.5573 ± 0.1565 | +0.0494 | 0.6125 ± 0.0820 | 0.6514 ± 0.1232 | 0.3862 / 0.6931 / 0.5926 | 2/3 | 31.0 / 27.9 | 0.0 / 0.0 |
| A6 | DA-D / ALL | 0.7372 ± 0.1123 | 0.6631 ± 0.0401 | -0.0741 | 0.7931 ± 0.0884 | 0.7347 ± 0.0315 | 0.6455 / 0.7090 / 0.6349 | 0/3 | 16.6 / 21.2 | 0.0 / 0.0 |

Oracle (upper bound, not a candidate):

| Row | DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|
| A5 | DA-O / ALL | 0.9947 ± 0.0053 | 0.8642 ± 0.0492 | -0.1305 | 0.9958 ± 0.0042 | 0.8931 ± 0.0387 | 0.9206 / 0.8307 / 0.8413 | 0/3 | 0.3 / 8.6 | 0.0 / 0.0 |

### KPConv

| Row | DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | DA-0 / ALL | 0.6984 ± 0.0399 | 0.7090 ± 0.0295 | +0.0106 | 0.7502 ± 0.0372 | 0.7708 ± 0.0232 | 0.7354 / 0.6772 / 0.7143 | 0/3 | 17.7 / 18.3 | 1.3 / 0.0 |
| B2 | DA-A / ALL | 0.6755 ± 0.0456 | 0.4550 ± 0.0826 | -0.2205 | 0.7075 ± 0.0359 | 0.5708 ± 0.0651 | 0.4127 / 0.5503 / 0.4021 | 3/3 | 16.4 / 34.3 | 4.0 / 0.0 |
| B3 | DA-D / ALL | 0.6667 ± 0.0881 | 0.6261 ± 0.0680 | -0.0406 | 0.7098 ± 0.0603 | 0.7056 ± 0.0536 | 0.5979 / 0.7037 / 0.5767 | 2/3 | 18.0 / 23.6 | 3.0 / 0.0 |
| B3b | DA-S / ALL | 0.7072 ± 0.2298 | 0.6825 ± 0.0596 | -0.0247 | 0.7479 ± 0.1569 | 0.7500 ± 0.0470 | 0.7196 / 0.7143 / 0.6138 | 1/3 | 16.1 / 20.0 | 2.3 / 0.0 |
| B4 | DA-0 / L-D | 0.6384 ± 0.1912 | 0.6614 ± 0.0140 | +0.0229 | 0.6978 ± 0.1207 | 0.7333 ± 0.0110 | 0.6561 / 0.6772 / 0.6508 | 0/3 | 20.9 / 21.3 | 1.9 / 0.0 |

Oracle (upper bound, not a candidate):

| Row | DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|
| B5 | DA-O / ALL | 1.0000 ± 0.0000 | 0.8113 ± 0.0291 | -0.1887 | 1.0000 ± 0.0000 | 0.8514 ± 0.0229 | 0.7778 / 0.8254 / 0.8307 | 0/3 | 0.0 / 11.9 | 0.0 / 0.0 |

### DGCNN

| Row | DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | DA-0 / ALL (= PP_P4a) | 0.8836 ± 0.0367 | 0.6508 ± 0.0966 | -0.2328 | 0.8652 ± 0.0253 | 0.7250 ± 0.0760 | 0.6032 / 0.7619 / 0.5873 | 2/3 | 2.7 / 22.0 | 4.7 / 0.0 |
| C2 | DA-A / ALL | 0.7989 ± 0.1972 | 0.5714 ± 0.1672 | -0.2275 | 0.8109 ± 0.1294 | 0.6625 ± 0.1317 | 0.3968 / 0.5873 / 0.7302 | 2/3 | 9.3 / 27.0 | 3.3 / 0.0 |
| C3 | DA-S / ALL | 0.9312 ± 0.0242 | 0.5503 ± 0.0601 | -0.3810 | 0.9181 ± 0.0261 | 0.6458 ± 0.0473 | 0.5079 / 0.5238 / 0.6190 | 3/3 | 1.3 / 28.3 | 3.0 / 0.0 |
| C4 | DA-A / L-N | 0.6508 ± 0.0840 | 0.4021 ± 0.0330 | -0.2487 | 0.6973 ± 0.0650 | 0.5292 ± 0.0260 | 0.4127 / 0.3651 / 0.4286 | 3/3 | 19.0 / 37.7 | 3.0 / 0.0 |
| C6 | DA-D / ALL | 0.9365 ± 0.0420 | 0.5503 ± 0.0750 | -0.3862 | 0.9223 ± 0.0419 | 0.6458 ± 0.0591 | 0.5238 / 0.4921 / 0.6349 | 2/3 | 1.0 / 28.3 | 3.0 / 0.0 |

Oracle (upper bound, not a candidate):

| Row | DA / Aug | Acc P4a | Acc P4 | Δ acc | Bal. acc P4a | Bal. acc P4 | Acc P4 per seed | Seeds < 0.635 on P4 | T→M P4a / P4 | M→T P4a / P4 |
|---|---|---|---|---|---|---|---|---|---|---|
| C5 | DA-O / ALL | 0.9524 ± 0.0420 | 0.8360 ± 0.0242 | -0.1164 | 0.9625 ± 0.0331 | 0.8708 ± 0.0191 | 0.8095 / 0.8413 / 0.8571 | 0/3 | 3.0 / 10.3 | 0.0 / 0.0 |


---

## 2. Own-cache check

Each run's own-cache (P4a) accuracy is compared with the FINAL target accuracy in its `run.log`
(4 decimals).

- **DGCNN: all 18 runs MATCH** exactly. C1's P4a and P4 numbers also match the Block PP
  cross-evaluation (`results/pp_summary/cross_eval_runs.csv`) on every seed.
- **PointNet++ and KPConv: 24 of 36 runs have the logged value inside their 3-repeat range** on
  P4a (to within 4-decimal rounding).
- **12 runs are outside by 1 or 2 clouds** (8 by 1 cloud, 4 by 2 clouds, none by more): 5
  PointNet++, 7 KPConv; 7 above the range and 5 below, so there is no common direction.

| Run | Logged FINAL acc | P4a repeats min – max | In clouds: logged / min–max | Outside by |
|---|---|---|---|---|
| FW_A4_s3 | 0.5556 | 0.5238 – 0.5238 | 35 / 33–33 | 2 clouds above |
| FW_A6_s1 | 0.6825 | 0.5397 – 0.6508 | 43 / 34–41 | 2 clouds above |
| FW_B3_s3 | 0.6349 | 0.6667 – 0.7302 | 40 / 42–46 | 2 clouds below |
| FW_B3b_s3 | 0.4921 | 0.4444 – 0.4603 | 31 / 28–29 | 2 clouds above |
| FW_A2_s1 | 0.6349 | 0.6508 – 0.6667 | 40 / 41–42 | 1 cloud below |
| FW_A3_s3 | 0.9524 | 0.9365 – 0.9365 | 60 / 59–59 | 1 cloud above |
| FW_A6_s2 | 0.8254 | 0.7778 – 0.8095 | 52 / 49–51 | 1 cloud above |
| FW_B1_s3 | 0.7460 | 0.6825 – 0.7302 | 47 / 43–46 | 1 cloud above |
| FW_B2_s1 | 0.6984 | 0.7143 – 0.7302 | 44 / 45–46 | 1 cloud below |
| FW_B3b_s1 | 0.7937 | 0.7302 – 0.7778 | 50 / 46–49 | 1 cloud above |
| FW_B4_s1 | 0.8413 | 0.8571 – 0.8571 | 53 / 54–54 | 1 cloud below |
| FW_B5_s1 | 0.9841 | 1.0000 – 1.0000 | 62 / 63–63 | 1 cloud below |

[inferred] With only 3 repeats the min–max range is narrower than the true spread of a
non-repeatable evaluation; four of these runs (A3 s3, A4 s3, B4 s1, B5 s1) gave the same value on
all 3 repeats, so their range has width 0. A 1–2 cloud difference is of the size of the
repeat spread seen elsewhere (largest P4a range among passing runs: PointNet++ 3 clouds, KPConv 6
clouds, B1 s1; among the failing runs A6 s1 has 7). The check is reported, not used to drop runs.

- **Guard (B1 seed 1, task 1 of 361787): PASS.** Printed:
  `GUARD FW_B1_s1 on P4a: repeats 0.6032 / 0.6667 / 0.6984 (min 0.6032, max 0.6984, tolerance 1 cloud = 0.0159); logged FINAL 0.6508 -> PASS`.

Largest repeat range (max − min, clouds out of 63) among the 24 passing A/B runs:

| Backbone | P4a | P4 |
|---|---|---|
| PointNet++ | 3 (A2 s3, A6 s3) | 4 (A2 s2, A2 s3, A3 s1, A4 s2) |
| KPConv | 6 (B1 s1) | 4 (B3 s2, B5 s2) |
| DGCNN | — (one evaluation) | — |

---

## 3. Findings

- **DGCNN: every row falls on every seed.** [measured] All 18 per-seed changes P4a → P4 are
  negative, from −0.06 (C5) to −0.44 (C6); non-Oracle row means fall by 0.23 to 0.39. On P4,
  tomato-called-maize rises (e.g. C3 1.3 → 28.3, C6 1.0 → 28.3). [inferred] DGCNN's P4a target
  accuracy depends heavily on what differs between the caches (soil/pot and the renormalisation,
  see caveats), not only on plant shape.
- **PointNet++: mixed.** [measured] A3 and A5 fall on all 3 seeds; A1 rises on all 3 (+0.19 to
  +0.23); A2, A4 and A6 are mixed across seeds (A2 2 down/1 up, A4 1 down/2 up, A6 2 down/1 up).
- **KPConv: mixed.** [measured] B2 and B5 fall on all 3 seeds (B2 −0.12 to −0.31). B1, B3, B3b and
  B4 are mixed across seeds; their row means change by −0.04 to +0.02, smaller than their seed SD
  on P4a (up to 0.23). [inferred] For these four rows the P4a seed spread is too large for 3 seeds
  to show a soil effect either way.
- **No maize cloud is called tomato on P4 in any run.** [measured] M→T on P4 is 0 in all 54 runs
  (and in every repeat). On P4a it is 0–6. Tomato-called-maize is higher on P4 than on P4a in 32 of
  the 45 non-Oracle runs. [inferred] Without soil (and after renormalisation) the models lean
  towards maize.
- **Non-Oracle accuracies on P4 bunch around the 0.635 majority baseline.** [measured] Over the 45
  non-Oracle runs, P4 accuracy spans 0.365–0.788 (SD 0.115) against 0.397–0.968 on P4a (SD 0.181);
  28 of 45 lie within ±0.10 of 0.635 and 27 of 45 are below it. Seeds that are high on P4a tend to
  fall and seeds near or below 0.635 on P4a tend to rise. Some rows sit well below the baseline on
  every seed (C4 0.37–0.43, B2 0.40–0.55, A2 0.50–0.54).
- **The Oracles fall by 0.12 to 0.19.** [measured] A5 −0.131, B5 −0.189, C5 −0.116 (row means);
  they fall on all 3 seeds and stay the best rows on P4 (0.81–0.86). The Oracles were trained on
  P4a Pheno4D clouds, which include soil and pot.

## 4. Caveats

- **P4 differs from P4a in more than soil.** The P4 cloud is normalised after the soil cut: centre
  and unit-sphere scale are computed on the plant only, the voxel size is height-relative, and all
  4096 FPS points fall on the plant (P4/P4a norm_scale median 0.78 over the 63 clouds, from the
  `fw_cross_eval.md` header). A P4a → P4 change is therefore **not only the loss of the soil cue**.
- **Train/test mismatch, not a soil-free model.** Every model was trained on soil-kept data (P4a
  source, and P4a target for the adaptation and Oracle rows). This measures how these models react
  when the soil is taken away at test time; it does not measure what a model trained without soil
  could reach.
- **A1's rise on P4 is a different bias, not a recovery.** [inferred] On P4a A1 calls almost every
  tomato maize (T→M 36 of 40, accuracy 0.43, below always-Tomato). On P4 T→M drops to 22.8 and
  accuracy reaches 0.638, about the majority baseline; it is still far from the other rows' P4a
  accuracy.
- **Selected-checkpoint values, not full-run means.** The P4a accuracies here are of the one
  selected checkpoint, so they are not comparable with the full-run mean target accuracy that is
  the primary metric in `FW_1_Results.md` and `FW_2_Results.md`.
- **Only 3 repeats** for A/B runs, and 3 seeds per row. See section 2 for the repeat spread.

## 5. Not done yet

- A target cache with the **soil removed but P4a's centre and scale** (normalisation computed
  before the soil cut) does not exist. It would separate the soil cue from the scale/centre/density
  change; without it the P4a → P4 drop cannot be attributed to soil alone.
- No FW row was trained on soil-removed data (only Block PP's `results/PP_P4_s*` runs were).
- Block D was not evaluated on the soil-removed target.

## Reproducing

```bash
sbatch --time=00:30:00 --export=ALL,FW_XEVAL_ROWS="C1 C2 C3 C4 C5 C6",FW_XEVAL_OUT=results/fw_cross_eval/stage1_dgcnn jobs/fw_cross_eval.sbatch
sbatch --time=04:00:00 --export=ALL,FW_XEVAL_ROWS="A1 A2 A3 A4 A5 A6",FW_XEVAL_OUT=results/fw_cross_eval/stage2a_pointnet2_v2 jobs/fw_cross_eval.sbatch
sbatch jobs/fw_cross_eval_kpconv_array.sbatch
```
These are the submit lines of 361654, 361785 and 361787 (from `sacct`). An output folder that already has results is refused unless `--overwrite` is passed to the script.
