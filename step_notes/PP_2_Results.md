# Block PP, Stage 4: baseline preprocessing comparison results

Scope: Stage 4 of `docs/PREPROCESSING_COMPARISON_PLAN.md`. This note reports the C1 baseline
(DGCNN, DA-0, ALL augmentation, `L_cls + L_seg`) trained once per preprocessing pipeline and seed
(9 pipelines × seeds 1, 2, 3 = 27 runs, all on Prajna, all with the fixed setup in plan section 4).
The pipelines themselves are defined in plan section 5.

Every number below comes from a file in the repo:

- `results/pp_summary/pp_summary.md` / `pp_by_pipeline.csv`: the per-pipeline table (mean ± std
  over seeds), written by `scripts/10_pp_summarize.py` from `results/PP_<ID>_s<seed>/run.log`.
- `results/pp_summary/pp_runs.csv`: one row per run.
- `results/pp_summary/shortcut_probe.md` / `.csv`: the shortcut probe (plan section 7, item 5),
  written by `scripts/09_shortcut_probe.py`. It is not a training metric.

Completeness, checked in `pp_runs.csv`: all 27 runs are `complete=True` with 100 epochs, and every
run used the same splits (263 Crops3D train / 45 Crops3D val / 63 Pheno4D held-out). P0 seed 1
**reproduces the existing C1 result exactly**: full-run mean 0.7913, selected epoch 95, selected
accuracy 0.746, Tomato mIoU 0.3248, Maize mIoU 0.3427. These match `results/C1_*` and CLAUDE.md, so
the `--manifest` change to the C1 script did not change the old default behaviour.

---

## 1. Per-pipeline table

Copied from `results/pp_summary/pp_summary.md`. Mean ± std over seeds 1, 2, 3. Target = Pheno4D
held-out, 63 clouds.

| Pipeline | Seeds | Full-run mean target acc | Full-run mean target bal. acc | Selected-checkpoint target acc | Source seg mIoU, tomato | Source seg mIoU, maize* | Decision vs P0 |
|---|---|---|---|---|---|---|---|
| P0 | 1,2,3 | 0.845 ± 0.048 | 0.826 ± 0.042 | 0.862 ± 0.102 | 0.308 ± 0.018 | 0.321 ± 0.019 | reference |
| P1 | 1,2,3 | 0.864 ± 0.009 | 0.839 ± 0.006 | 0.915 ± 0.018 | 0.307 ± 0.026 | 0.325 ± 0.029 | beats P0 |
| P2 | 1,2,3 | 0.717 ± 0.010 | 0.777 ± 0.008 | 0.772 ± 0.024 | 0.317 ± 0.024 | 0.177 ± 0.013 | no clear difference |
| P3 | 1,2,3 | 0.703 ± 0.014 | 0.766 ± 0.010 | 0.730 ± 0.069 | 0.322 ± 0.011 | 0.191 ± 0.020 | no clear difference |
| P4 | 1,2,3 | 0.703 ± 0.014 | 0.766 ± 0.010 | 0.730 ± 0.069 | 0.322 ± 0.011 | 0.191 ± 0.020 | no clear difference |
| P4a | 1,2,3 | 0.863 ± 0.015 | 0.846 ± 0.014 | 0.884 ± 0.037 | 0.333 ± 0.022 | 0.345 ± 0.004 | beats P0 |
| P4b | 1,2,3 | 0.678 ± 0.006 | 0.747 ± 0.005 | 0.688 ± 0.072 | 0.295 ± 0.042 | 0.188 ± 0.017 | no clear difference |
| P4c | 1,2,3 | 0.703 ± 0.014 | 0.766 ± 0.010 | 0.730 ± 0.069 | 0.322 ± 0.011 | 0.191 ± 0.020 | no clear difference |
| P4d | 1,2,3 | 0.571 ± 0.021 | 0.662 ± 0.017 | 0.561 ± 0.033 | 0.298 ± 0.022 | 0.149 ± 0.017 | no clear difference |
| *majority baseline (40/63)* | – | 0.635 | – | – | – | – | – |
| *existing P0 single-seed C1* | – | 0.791 | – | – | – | – | – |

Decision rule (plan section 7): a pipeline beats P0 only if all 3 of its seeds have a higher
full-run mean target accuracy than P0's 3-seed mean (0.845). Selected checkpoint = lowest source
validation loss; target labels are never used for selection.

\* Maize mIoU is **not comparable across pipelines**; see caveat 2.

Note on the "Decision vs P0" column: "no clear difference" is the plan's fixed wording for
"does not beat P0 under the rule". It is not a claim that the pipeline is equal to P0. P2, P3/P4,
P4b and P4d are all clearly *lower* than P0 (see the findings).

---

## 2. Findings

### 2.1 Soil/pot removal lowers target accuracy by about 16 points

P4 and P4a differ in exactly one step: P4a skips soil/pot removal (plan section 5, ablation
table). Full-run mean target accuracy:

- P4a (soil kept): **0.863 ± 0.015**
- P4 (soil removed): **0.703 ± 0.014**
- Difference: **−0.160**

The drop holds seed by seed (`pp_runs.csv`): P4a 0.847 / 0.866 / 0.877 against P4 0.715 / 0.688 /
0.707. Selected-checkpoint accuracy drops the same way (0.884 → 0.730). Every pipeline that
removes soil (P2, P3/P4, P4b, P4d) lands between 0.571 and 0.717. Every pipeline that keeps soil
(P0, P1, P4a) lands between 0.845 and 0.864. No pipeline overlaps the two groups.

What this does **not** show: why removing soil hurts. See caveat 4 (soil/pot may itself be a
species cue present in both datasets).

### 2.2 P4d (normalise before soil removal) is the worst pipeline, below the majority baseline

P4d is P4 with normalisation done before soil removal, so the soil is included in the centre and
scale. It gives **0.571 ± 0.021** full-run mean target accuracy and 0.561 ± 0.033 at the selected
checkpoint. Both are **below the 0.635 majority baseline** (always predicting Tomato, 40/63).
Every seed is below the baseline (0.548 / 0.590 / 0.574). Compared with P4 (0.703) it loses a
further 13 points.

Inference, not tested: removing soil after normalising leaves the plant off-centre and at a
scale that depends on how much soil each cloud had. That would explain why P4d is worse than P4.

### 2.3 P1 and P4a are more stable than P0, but not clearly more accurate

Both formally pass the decision rule ("beats P0" in the table): all three seeds of each are above
P0's mean of 0.845. But the margin is small and P0 itself is noisy:

| | Full-run mean target acc | Seed std | Selected-checkpoint acc |
|---|---|---|---|
| P0 | 0.845 | 0.048 | 0.862 ± 0.102 |
| P1 | 0.864 (+0.019) | 0.009 | 0.915 ± 0.018 |
| P4a | 0.863 (+0.018) | 0.015 | 0.884 ± 0.037 |

- The +0.02 accuracy margin is smaller than P0's own seed std (0.048). P0's seeds are 0.791 /
  0.860 / 0.884, so two of P0's three seeds are at or above P1's and P4a's means. P0's mean is
  pulled down by seed 1.
- What is clearly different is **stability across seeds**: std 0.009 (P1) and 0.015 (P4a)
  against P0's 0.048, and selected-checkpoint accuracy std 0.018 / 0.037 against P0's 0.102.
- Selected-checkpoint accuracy is higher for P1 (0.915 against 0.862), but with P0's selected
  std of 0.102 this is not a reliable difference with 3 seeds.

Reading: P1 (DefRec/PointDA-10 style: up-axis align, voxel every cloud, unit cube) and P4a (our P4
without soil removal) give a more reproducible baseline than P0. They do not give a clearly more
accurate one.

### 2.4 Other ablations, for completeness

- **P4b** (old conditional sampling path, voxel only if > 50k points) against P4: 0.678 against
  0.703. Small drop, all three P4b seeds below all three P4 seeds (P4b 0.685 / 0.673 / 0.677;
  P4 0.715 / 0.688 / 0.707). Consistent with the conditional path hurting slightly, but this is
  measured inside the soil-removed group, all of which is far below P0.
- **P4c** (no up-axis alignment) against P4: identical (caveat 1).
- **P2** (random sampling instead of FPS, soil removed): 0.717, close to P3/P4 (0.703).

---

## 3. Caveats

### 3.1 P3, P4 and P4c are bit-identical on this data, so they count as one pipeline

Checked directly, not inferred from the equal table rows:

- **Caches**: all 531 `.npz` files in `data/preprocessed_variants/P3/`, `P4/` and `P4c/` have
  identical arrays for every data key (`points`, `labels`, `norm_scale`, `norm_center_*`, all
  point counts, `voxel_size_mm`, `sampling_method`, `norm_kind`). The only differences are the
  metadata fields `pipeline_id` (all three) and `up_axis_aligned` (P4 only). The file checksums
  differ only because of these metadata fields.
- **Training logs**: after removing timestamps and run names, `results/PP_P3_s<n>/run.log`,
  `PP_P4_s<n>/run.log` and `PP_P4c_s<n>/run.log` are line-for-line identical for each seed n = 1,
  2, 3.

Why: P4 is P3 plus up-axis alignment, and P4c is P4 without it. Up-axis alignment does nothing on
this data. Crops3D maize and Pheno4D are confirmed +z, and Crops3D tomato is trusted as +z
(`TOMATO_UP_AXIS_ALIGN = False` in `scripts/preprocessing_variants.py`; decision recorded in
`step_notes/PP_0_Prerequisite_Checks.md`). So no cloud is rotated. Consequences:

- There are effectively **7 distinct pipelines**, not 9. P3/P4/P4c are one result counted three
  times. The table should not be read as three independent confirmations.
- The P4c ablation **tells us nothing about up-axis alignment**. Whether alignment matters is
  untested on this data, because there was nothing to align.

### 3.2 Maize mIoU is not comparable across pipelines

Soil-removal pipelines (P2, P3/P4/P4c, P4b, P4d) have no Maize class 0 (soil) or class 5 (pot)
points left. The summarizer scores absent classes as 0, so their Maize mIoU is capped at 4/6 of
what it could be. P0, P1 and P4a keep all six classes. So the lower Maize mIoU (about 0.15–0.19
against about 0.32–0.35) is mostly a scoring artefact and must not be read as worse
segmentation. Tomato mIoU (no soil class in Crops3D tomato) is comparable, and it is flat across
pipelines (0.295–0.333, all within about 1–2 seed stds of each other).

### 3.3 The shortcut probe is mixed evidence

The probe (`results/pp_summary/shortcut_probe.md`) fits a logistic regression on non-shape
features, trained on Crops3D species and tested on Pheno4D. `all3` = raw point count + mean
nearest-neighbour spacing + flatness of the lowest 10% of points; `cache_only` = the last two.

| Pipeline | all3 src acc | all3 tgt acc | cache_only src acc | cache_only tgt acc |
|---|---|---|---|---|
| P0 | 0.978 | 0.651 | 0.978 | 0.921 |
| P1 | 0.978 | 0.667 | 0.978 | 0.921 |
| P2 | 0.978 | 0.651 | 0.956 | 0.540 |
| P3 = P4 = P4c | 0.978 | 0.762 | 0.978 | 0.746 |
| P4a | 0.978 | 0.667 | 0.978 | 0.905 |
| P4b | 0.978 | 0.730 | 0.978 | 0.635 |
| P4d | 1.000 | 0.746 | 0.978 | 0.571 |

Why mixed:

- **No pipeline removes the source shortcut.** Non-shape features separate Crops3D species at
  0.956–1.000 accuracy in every pipeline. Preprocessing does not make the source species
  indistinguishable by density/flatness.
- **The two feature sets move in opposite directions.** Soil removal lowers the `cache_only`
  target accuracy (P4a 0.905 → P4 0.746) but raises the `all3` target accuracy (P4a 0.667 → P4
  0.762). So "soil removal weakens the shortcut" is true for one feature set and false for the
  other.
- **Target AUC stays high while target accuracy is low** for `all3` (AUC 0.995–0.999 in every
  pipeline, accuracy 0.65–0.76). The features still rank Pheno4D species almost perfectly; only
  the decision threshold learned on Crops3D does not transfer. A shortcut with a shifted threshold
  is still a shortcut.
- The probe's target accuracy tracks the C1 results only loosely. For example, `cache_only` is
  high where C1 is high (P0/P1/P4a), but P2 has the lowest probe accuracy (0.540) and still
  matches P3/P4 in C1 (0.717 against 0.703).

The probe therefore cannot tell us whether the C1 model uses these non-shape cues. It only shows
that the cues exist in every pipeline.

### 3.4 Open question: soil or pot may itself be a species cue shared by both datasets

The plan's starting hypothesis was that soil/pot is a **source-only** confound (Crops3D maize has
soil and pot, Crops3D tomato has none), so removing it should help transfer. The results go the
other way: removing it costs about 16 points. One explanation that fits, **not tested**: Pheno4D
clouds also contain pot and soil for both species (plan section 2), and the soil/pot region may
differ between species in a way that is consistent across both datasets (for example, how much of
the normalised cloud the soil occupies, or plant height relative to the pot). If so, the model
uses soil/pot as a species cue that happens to transfer, and soil removal takes away a useful
(but non-plant) signal rather than a spurious one.

One supporting observation, also not a test: the `cache_only` probe, which includes flatness of
the lowest 10% of points (where soil sits), transfers to Pheno4D at 0.905–0.921 accuracy in the
pipelines that keep soil (P0, P1, P4a). That is consistent with bottom-of-cloud geometry carrying
species information in both datasets.

Why this matters: if true, the higher accuracy of P0/P1/P4a is partly not from plant shape, and a
"best" preprocessing choice made on this table alone would be choosing a non-plant cue. Possible
tests (not run; decide with the user): per-species statistics of the soil/pot region in Pheno4D
(point fraction, extent relative to plant height); a classifier trained on the soil/pot points
alone; or evaluating soil-kept models on Pheno4D clouds with the soil cut applied only at test
time.

### 3.5 Minor

- `pp_runs.csv` flags P1 seed 3: logged best epoch 98, but the argmin of the logged (4-decimal)
  validation loss is epoch 96. Most likely a rounding tie in the logged values; the summarizer
  uses the logged best epoch, which is what the training script itself selected.
- 3 seeds per pipeline is a small sample. Std values are from 3 numbers.

---

## 4. Not done in this stage

- **Proposed preprocessing strategy**: not written up. Caveat 3.4 should be resolved, or at least
  discussed with the user, before choosing between "keep soil" (P1/P4a) and "remove soil" (P4).
- **Before/after renders** of the variant pipelines (Stage 2 item): still not done.
- **Table + note sent to maam**: not done.
- **P5** (noise matching, plan section 5): not run. The plan says to decide it with the user.

## Reproducing

```bash
python scripts/10_pp_summarize.py   # reads results/PP_*/run.log -> results/pp_summary/pp_*.{csv,md,txt}
python scripts/09_shortcut_probe.py # laptop, from each pipeline's cache -> results/pp_summary/shortcut_probe.*
```
