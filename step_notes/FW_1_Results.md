# Full-workflow rerun (FW) on P4a: PointNet++ (A1–A6) and DGCNN (C1–C6) results

Scope: the first part of the full-workflow rerun described in `jobs/fw/README.md`. Every Block A
and Block C row was retrained with seeds 1, 2, 3 on the soil-kept preprocessing P4a
(`data/preprocessed_variants/P4a/preprocessed_manifest.csv`), with each row's original arguments.
P4a was chosen as the best baseline in Block PP (`step_notes/PP_2_Results.md`, `PP_3_Results.md`).
KPConv rows (B1–B5, B3b) and Block D had not been run on P4a when this was written; Block D was run
later (job 359157, `step_notes/FW_3_Results.md`).

Every number below comes from a file in the repo:

- `results/pp_summary/fw_summary.md` / `fw_by_row.csv`: A1–A6 and C2–C6, written by
  `scripts/15_fw_summarize.py` from `results/FW_<row>_s<seed>/run.log` (last regenerated in commit
  f46be17, "FW: summary after C3").
- `results/pp_summary/pp_summary.md` / `pp_runs.csv`: C1, written by `scripts/10_pp_summarize.py`
  from `results/PP_<pipeline>_s<seed>/run.log`. FW C1 is **not** a separate run: `PP_P4a_s{1,2,3}`
  uses the same script, arguments, seeds and manifest, so it is reused as FW C1 (as planned in
  `jobs/fw/README.md`).

Target = Pheno4D held-out set, 63 clouds (40 tomato, 23 maize). Majority-class reference (always
Tomato) = 0.635.

Metrics: **full-run mean target accuracy** (mean of the per-epoch target accuracy over the 100
epochs; primary metric), full-run mean balanced accuracy, and selected-checkpoint target accuracy
(lowest source validation loss; for the Oracle rows A5/C5 the lowest Oracle validation loss).
Target test labels are never used for selection. Std = sample std over 3 seeds.

## Jobs

All on Prajna (`dgx`), every seed finished with an `OK` line in its job log
(`results/logs/fw_<row>_<jobid>.log`).

| Round | Rows (job id) |
|---|---|
| 1 | A1 (352559), A5 (352560), A2 (352561), A4 (352562), A6 (352563) |
| 2 | C4 (353254), C2 (353255), C6 (353256), C5 (353257), A3 (353258) |
| 3 (partial) | C3 (353781) |
| C1 | reused from Block PP: `results/PP_P4a_s{1,2,3}` |

PointNet++ eval is not deterministic (random farthest-point-sampling start), so for 9 PointNet++
runs the FINAL re-evaluation of the selected checkpoint differs by one or two clouds from that
epoch's logged accuracy. The FINAL value is used, as in all earlier summaries (list in
`fw_summary.md`, explanation in `step_notes/PointNet2_Eval_Nondeterminism.md`).

---

## 1. Per-row table (P4a, 3-seed mean ± std)

| Row | Backbone / DA / Aug | Full-run mean target acc | Full-run mean target bal. acc | Selected-checkpoint target acc |
|---|---|---|---|---|
| A1 | PointNet++ / DA-0 / ALL | 0.4968 ± 0.0202 | 0.6035 ± 0.0159 | 0.4233 ± 0.0330 |
| A2 | PointNet++ / DA-A / ALL | 0.5651 ± 0.0674 | 0.6555 ± 0.0526 | 0.5132 ± 0.1191 |
| A3 | PointNet++ / DA-S / ALL | 0.7313 ± 0.1024 | 0.7743 ± 0.0803 | 0.7725 ± 0.2586 |
| A4 | PointNet++ / DA-A / L-N | 0.5436 ± 0.0387 | 0.6399 ± 0.0312 | 0.5185 ± 0.1081 |
| A5 | PointNet++ / DA-O / ALL (Oracle) | 0.9275 ± 0.0113 | 0.9420 ± 0.0087 | 0.9894 ± 0.0092 |
| A6 | PointNet++ / DA-D / ALL | 0.6961 ± 0.0699 | 0.7601 ± 0.0550 | 0.7725 ± 0.0783 |
| C1 | DGCNN / DA-0 / ALL (= `PP_P4a`) | 0.8632 ± 0.0155 | 0.8462 ± 0.0139 | 0.8836 ± 0.0367 |
| C2 | DGCNN / DA-A / ALL | 0.6429 ± 0.0471 | 0.6964 ± 0.0347 | 0.7989 ± 0.1972 |
| C3 | DGCNN / DA-S / ALL | 0.8525 ± 0.0121 | 0.8600 ± 0.0137 | 0.9312 ± 0.0242 |
| C4 | DGCNN / DA-A / L-N | 0.6317 ± 0.0279 | 0.6895 ± 0.0237 | 0.6508 ± 0.0840 |
| C5 | DGCNN / DA-O / ALL (Oracle) | 0.8914 ± 0.0044 | 0.9104 ± 0.0015 | 0.9524 ± 0.0420 |
| C6 | DGCNN / DA-D / ALL | 0.8687 ± 0.0410 | 0.8659 ± 0.0366 | 0.9365 ± 0.0420 |

Per-seed values are in `fw_summary.md` (A2–A6, C2–C6) and `pp_runs.csv` (C1: full-run mean
0.8467 / 0.8656 / 0.8773; selected 0.8413 / 0.9048 / 0.9048).

---

## 2. Comparison with the original single-seed runs (old preprocessing P0)

Original = the earlier strategy-table run: **seed 1 on P0** (`data/preprocessed/...`), read from
its own `run.log` with the same parser. FW = **seeds 1–3 on P4a**. For every row except C1, both
the data and the number of seeds differ, so a difference is not attributable to the preprocessing
alone.

| Row | Full-run mean: original (s1, P0) | FW mean (P4a) | FW − orig | FW seeds above orig | Selected: original | FW mean | FW − orig |
|---|---|---|---|---|---|---|---|
| A1 | 0.5986 | 0.4968 | −0.1017 | 0/3 | 0.4603 | 0.4233 | −0.0370 |
| A2 | 0.5538 | 0.5651 | +0.0113 | 2/3 | 0.4444 | 0.5132 | +0.0688 |
| A3 | 0.7997 | 0.7313 | −0.0684 | 0/3 | 0.8889 | 0.7725 | −0.1164 |
| A4 | 0.5862 | 0.5436 | −0.0426 | 0/3 | 0.5397 | 0.5185 | −0.0212 |
| A5 | 0.9156 | 0.9275 | +0.0120 | 2/3 | 1.0000 | 0.9894 | −0.0106 |
| A6 | 0.6494 | 0.6961 | +0.0468 | 3/3 | 0.5714 | 0.7725 | +0.2011 |
| C1 | 0.7913 | 0.8632 | +0.0719 | 3/3 | 0.7460 | 0.8836 | +0.1376 |
| C2 | 0.6327 | 0.6429 | +0.0101 | 1/3 | 0.6667 | 0.7989 | +0.1322 |
| C3 | 0.7030 | 0.8525 | +0.1495 | 3/3 | 0.5714 | 0.9312 | +0.3598 |
| C4 | 0.6127 | 0.6317 | +0.0190 | 2/3 | 0.5556 | 0.6508 | +0.0952 |
| C5 | 0.9251 | 0.8914 | −0.0336 | 0/3 | 1.0000 | 0.9524 | −0.0476 |
| C6 | 0.8160 | 0.8687 | +0.0526 | 3/3 | 0.8571 | 0.9365 | +0.0794 |

C1's row is computed from `pp_runs.csv` (P0 seed 1 and P4a seeds 1–3); the others are copied from
`fw_summary.md`.

**Only C1 is a like-for-like preprocessing comparison**, because only C1 also has P0 run with 3
seeds (Block PP). The single-seed originals of the other rows are one draw each, and seed-to-seed
spread on P4a is large for several rows (A3 std 0.102, A6 0.070, A2 0.067), so a change of a few
points against one original seed is within run-to-run noise. Large changes in both directions
remain notable as observations, not as effects of preprocessing: A1 −0.10 (0/3 seeds above), A3
−0.07 (0/3), C3 +0.15 (3/3).

### 2.1 C1: P4a against P0, 3 seeds each (Block PP)

| | Full-run mean | Seeds | Selected-checkpoint |
|---|---|---|---|
| P0 | 0.8451 ± 0.0482 | 0.7913 / 0.8599 / 0.8843 | 0.8624 ± 0.1021 |
| P4a | 0.8632 ± 0.0155 | 0.8467 / 0.8656 / 0.8773 | 0.8836 ± 0.0367 |

Under the plan's rule (a pipeline beats P0 only if all 3 of its seeds exceed P0's 3-seed mean,
0.8451), **P4a beats P0**: all three P4a seeds are above 0.8451. **The margin is small**: +0.018 on
the mean, smaller than P0's own seed std (0.048), and two of P0's three seeds (0.8599, 0.8843) are
at or above P4a's lowest seed. What clearly differs is the spread (std 0.016 against 0.048). See
`step_notes/PP_2_Results.md` section 2.3.

---

## 3. Findings

### 3.1 No adaptation row clearly beats C1 on P4a

Rule used: the same all-3-seeds rule as Block PP, against C1's P4a mean (0.8632).

| Row | Full-run mean | Seeds above 0.8632 |
|---|---|---|
| C6 (DA-D) | 0.8687 ± 0.0410 | 2/3 (0.8754, 0.9059; 0.8248 below) |
| C3 (DA-S) | 0.8525 ± 0.0121 | 1/3 (0.8649; 0.8406 and 0.8519 below) |
| C2 (DA-A) | 0.6429 ± 0.0471 | 0/3 |
| C4 (DA-A, L-N) | 0.6317 ± 0.0279 | 0/3 |
| A1–A4, A6 | 0.497–0.731 | 0/3 each |

- **C3 is about equal to C1 on P4a (0.8525 against 0.8632), not better.** Its large gain over the
  original C3 (+0.15) comes from comparing against a single P0 seed (0.7030) that was itself below
  the old C1 seed (0.7913). On the same data, DA-S lands 0.011 below DA-0 on the primary metric.
  It is higher on the two secondary metrics (balanced 0.8600 against 0.8462; selected 0.9312
  against 0.8836); the selected-accuracy gap (0.048) is about the size of C1's seed std (0.037),
  so this is not a clear difference either way with 3 seeds.
- **C6 (DA-D)** has the highest DGCNN non-Oracle mean (0.8687), but only 2/3 seeds above C1 and a
  seed std (0.041) far larger than its margin (+0.006). Not a clear win.
- **DA-A (C2, C4)** stays about 22–23 points below C1 on P4a, the same direction as the original
  runs.
- **PointNet++ is far below DGCNN on P4a.** A1 (DA-0) is 0.4968, **below the majority baseline
  (0.635)** in all three seeds (0.5014 / 0.5143 / 0.4748). The best PointNet++ non-Oracle row is A3
  (DA-S, 0.7313), but with a std of 0.102 (seed 2 at 0.6130), followed by A6 (DA-D, 0.6961). Within
  Block A, DA-S and DA-D both beat A1 in all three seeds; that only says they beat a baseline that is
  itself below the majority reference.

### 3.2 C5 and A5 are the Oracle upper bound, not candidates

C5 (0.8914 ± 0.0044) and A5 (0.9275 ± 0.0113) train on labelled Pheno4D. They are reference ceilings
only (CLAUDE.md, DA-O) and are not counted in 3.1. On P4a, C5's ceiling is 0.028 above C1. The
original gap was 0.13 (0.9251 against 0.7913), but that used one P0 seed. C5 is lower on P4a than
its original (−0.034, 0/3 seeds above). Not investigated; one difference is that the Oracle rows now
train on P4a Pheno4D clouds, which include pot and soil.

### 3.3 Soil-cue caveat (P4a keeps soil)

P4a keeps soil and pot in both datasets. Block PP found (`PP_2_Results.md` 3.6, `PP_3_Results.md`)
that soil-kept models drop to about the majority baseline when target soil is removed at test time
(P4a: 0.651), and that early-season clouds are close to species-blind under both P4 and P4a. **All
P4a numbers above may therefore partly reflect a soil/pot cue rather than plant shape.** This rerun
did not test that at the time; it was tested later on the soil-removed target cache (`step_notes/FW_4_Results.md`). The comparisons in 3.1
are between rows that all see the same soil-kept data, so the cue is shared by both sides of each
comparison, but it can still inflate the absolute accuracies and the C1–C5 gap.

---

## 4. Not done yet

- KPConv rows B1–B5 and B3b on P4a (rounds 3 and 4 of `jobs/fw/README.md`, except C3): not started.
- Block D on P4a: run later (job 359157), see `step_notes/FW_3_Results.md`.
- Cross-evaluation of the FW rows on the soil-removed target cache (tests 3.3): done later, see `step_notes/FW_4_Results.md`.
- Segmentation mIoU: not reported (not comparable across rows).

## Reproducing

```bash
python scripts/15_fw_summarize.py   # reads results/FW_*/run.log -> results/pp_summary/fw_summary.md, fw_by_row.csv
python scripts/10_pp_summarize.py   # C1 = PP_P4a, P0 reference -> results/pp_summary/pp_*.{csv,md,txt}
```
