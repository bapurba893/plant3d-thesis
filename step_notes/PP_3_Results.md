# Block PP, Stage 5: learned soil weight and the Pheno4D soil-removal inspection

Scope: follow-ups to `step_notes/PP_2_Results.md`, whose open question (caveat 3.4) was whether
soil/pot acts as a species cue. Stage 4 found that the soil-removed pipeline P4 scores well below
the soil-kept pipeline P4a on the held-out Pheno4D target (full-run mean target accuracy 0.703 vs
0.863). This note covers three pieces of work done since then:

1. one preprocessing function with a `remove_soil` switch, checked against the existing caches;
2. a learned weight on the soil/pot points' segmentation loss (instead of keeping or cutting soil
   outright), compared against P4a and P4;
3. an inspection of the held-out Pheno4D clouds to find out where P4 loses accuracy.

Every number comes from a file in the repo; the source is named for each section.

Common setup, unchanged from Stage 4: row C1 (DGCNN, DA-0, ALL augmentation, `L_cls + L_seg`),
100 epochs, seeds 1, 2, 3, 263 Crops3D train / 45 Crops3D val / 63 Pheno4D held-out clouds
(40 tomato, 23 maize). "Full-run mean" = mean over the 100 epochs of the per-epoch target
accuracy (the primary metric). "Selected checkpoint" = lowest source validation loss; target
labels are never used for selection.

---

## 1. Single `remove_soil` function verified (commit 6fd7e77)

`scripts/preprocessing_pipeline.py::preprocess_cloud(..., remove_soil=...)` replaces the two
separate pipeline functions: `remove_soil=True` reproduces P4, `remove_soil=False` reproduces P4a.
`scripts/12_check_single_pipeline.py` reran it on every raw cloud (on the laptop, where the raw
files are) and compared with the cached `.npz` files, recomputing outlier removal from raw (the
stricter mode).

Result (`results/pp_checks/single_pipeline_check.csv`): **1062 of 1062 comparisons match**
(531 clouds × 2 settings). Maximum absolute point difference 0.0 on every cloud; labels and
normalisation centre/scale equal on every cloud. The function is a drop-in replacement for both
caches.

---

## 2. Learned soil weight (commit 39e129e)

### 2.1 What was trained (commit 12c8bf5)

`adapters/train_c1_dgcnn_da0_softw.py` (a copy of the C1 script with one added option) on the
**soil-kept P4a cache**. Full definition in `adapters/soil_weight.py`; in short:

- The weight multiplies the weighted cross-entropy term of `L_seg` on **Crops3D maize soil/pot
  points only** (ids 0 and 5). Tomato has no soil; Pheno4D is unlabelled in this baseline; the
  Lovász term is left unweighted. The existing class weights stay.
- Two settings:
  - `learned_scalar`: one trainable value, the same weight on every soil/pot point.
  - `learned_features`: a small MLP (3 → 16 → 1) giving a per-point weight from three label-free
    features (height from the nearer vertical end, radial distance from the vertical axis, local
    density).
- Raw weight bounded to [0.2, 5.0]. The weights are then normalised so the mean over all maize
  points in a batch is 1. The weight can therefore only shift emphasis between soil and plant
  points, never scale the loss down overall (user decisions 2026-10-14).
- Model selection uses the original unweighted `L_seg`, so it compares directly with P4a (which is
  exactly the "no weight" case and was not rerun).

Jobs: `jobs/pp_softweight.sbatch`, array job 351327 (task 0 = scalar, task 1 = features), all
6 runs complete. Logs: `results/PP_SW_learned_{scalar,features}_s{1,2,3}/run.log`.

### 2.2 Results

From `results/pp_summary/sw_summary.md` (`scripts/13_sw_summarize.py`, which reuses the parsing
in `scripts/10_pp_summarize.py`). Mean ± sample std over 3 seeds; per-seed values in the source
file.

| Version | Full-run mean target acc | Full-run mean target bal. acc | Selected-checkpoint target acc |
|---|---|---|---|
| P4a (soil kept) | 0.8632 ± 0.0155 | 0.8462 ± 0.0139 | 0.8836 ± 0.0367 |
| P4 (soil removed) | 0.7033 ± 0.0139 | 0.7655 ± 0.0105 | 0.7302 ± 0.0692 |
| Learned scalar | 0.8504 ± 0.0319 | 0.8348 ± 0.0242 | 0.8783 ± 0.0485 |
| Learned features | 0.8641 ± 0.0064 | 0.8442 ± 0.0138 | 0.9154 ± 0.0183 |

**Decision rule** (a version beats P4a only if all 3 of its seeds exceed the P4a 3-seed mean,
0.8632, on the full-run mean):

- Learned scalar: **no**. 2 of 3 seeds above (s1 0.8138, s2 0.8724, s3 0.8651).
- Learned features: **no**. 2 of 3 seeds above (s1 0.8695, s2 0.8570, s3 0.8657). Its mean is
  essentially tied with P4a and has the smallest seed-to-seed spread; its selected-checkpoint
  accuracy is the highest of the four, but that is not the primary metric.

Both learned versions stay far above P4. Maize mIoU is not compared across versions (not
comparable between soil-kept and soil-removed caches, see PP_2 section 3).

### 2.3 What the weights learned

Last 10 epochs (90–99), from the `SoilW` lines in each run.log. Means are averaged over the
epochs; min/max are over the epochs. "At 0.2" = fraction of soil points within 0.048 (1% of the
bound range) of the lower bound.

| Version | Raw soil weight (mean, range) | Normalised soil weight (mean) | Plant weight (mean) | Fraction at 0.2 | Fraction at 5.0 |
|---|---|---|---|---|---|
| Scalar s1/s2/s3 | 0.862 / 0.860 / 0.863 (range under 0.001) | 0.899 / 0.897 / 0.898 | 1.041 / 1.041 / 1.040 | 0 | 0 |
| Features s1/s2/s3 | 0.257 / 0.255 / 0.237 (0.20–0.39) | 0.329 / 0.326 / 0.304 | 1.269 / 1.270 / 1.275 | 0.45 / 0.43 / 0.76 | 0 |

- The scalar weight barely moves from its starting value of 1 (it settles at about 0.86) and never
  approaches a bound.
- The feature-based weight pushes soil down hard, and 43–76% of soil points end at the 0.2 floor
  on every seed. Nothing reaches 5.0.
- So the feature-based model strongly de-emphasises soil in the segmentation loss, yet target
  classification accuracy stays the same as P4a. Interpretation (not tested): the species
  classifier's use of soil, if any, does not depend on how much the segmentation loss weights soil
  points.

---

## 3. Pheno4D soil-removal inspection (commit a91dc3a)

`scripts/14_pheno4d_inspection.py`, outputs and a ranked findings list in
`results/pp_checks/pheno4d_before_after/README.md`. Nothing trained. The saved P4/P4a checkpoints
were evaluated on their own caches on GPU (job 351955). All six reproduce their run.log FINAL
target line exactly. On CPU, P4a seed 1 is off by one cloud, a known precision effect (PP_2).

### 3.1 Findings

1. **The whole P4 deficit is early-season tomato clouds called maize.** Per-class recall (3-seed
   mean): P4a tomato 0.933 / maize 0.797; P4 tomato 0.575 / maize 1.000. 15 clouds are wrong in P4
   (≥2 of 3 seeds) but right in P4a (≥2 of 3). All 15 are tomato, scanned at dates 305–313 (the
   earliest scans; plants 27–64 mm tall in P4), from both held-out plants, mostly with P4 maize
   probability > 0.99. Early tomato accuracy: P4 0.119, P4a 1.000. Without the 22 early clouds:
   P4 0.886, P4a 0.935.
2. **Early clouds are near species-blind in both versions, with opposite biases.** P4a calls 4
   early maize clouds (dates 313–314) tomato on all 3 seeds; P4 gets every maize cloud right. In
   P4a the early clouds are 90–97% pot surface (soil-labelled) with a small plant. P4a calls them
   tomato and P4 calls them maize. The held-out set has 14 early tomato and 8 early maize clouds,
   so P4a's bias scores higher. P4a's advantage on early clouds is therefore not evidence of
   better plant-shape recognition.
3. **A soil remnant on small plants goes with the maize calls (supported, confounded with plant
   size).** The cut removes 89–99% of soil on every annotated tomato cloud, as validated in PP_0.
   But on tiny early plants the remainder is a large share of the 4096 points. In P4, soil-labelled
   points make up a median of 0.44 of the flipped clouds versus 0.04 in the other tomato clouds
   (Mann–Whitney p = 2.5e-5, 8 vs 14 annotated). Across annotated tomato clouds, P4's maize
   probability rises with that fraction (Spearman ρ = 0.79). The renders show the remnant as a
   dense clump at the stem base. Early maize in P4 looks the same and is called maize.
4. **Plant size / growth stage cannot be separated from finding 3.** Flipped clouds are also
   smaller (P4 height median 42 vs 87 mm; Spearman height vs maize probability ρ = −0.70) and
   earlier. Small plant and large soil share go together.
5. **Normalised point spacing of Pheno4D tomato moves away from Crops3D tomato in P4** (KS 0.33 in
   P4a → 0.80 in P4). This is most likely a side effect of the dense soil clump, not a separate
   cause.

Not supported by the data:

- **Truncated stems.** Plant points lost to the cut have a median of 0.05% and a maximum of 0.7%
  on the flipped clouds (raw-cloud check, `results/pp_checks/pheno4d_soil_removal_annotated.csv`,
  margin 2.0 mm), and the stems look intact in the renders.
- **Leftover pot structure.** No rim or wall is visible; the remnant is soil.
- **A general size/shape mismatch after rescaling.** Removing soil moves Pheno4D tomato *closer*
  to Crops3D tomato on height, width, height/width, bottom position and mean radius in normalised
  space.
- **Absolute size/density differences between the datasets.** These are the largest differences
  (KS 0.9–1.0), but they are equally large in P4 and P4a, and normalisation removes them before
  the network.

### 3.2 Limits

- **Best-checkpoint breakdown only.** Per-epoch predictions were not saved, so the breakdown
  explains the selected-checkpoint gap (P4 0.730 vs P4a 0.884), not the full-run-mean gap
  (0.703 vs 0.863).
- **Early clouds are near species-blind in both versions** (finding 2), so neither version's
  early-cloud accuracy measures plant recognition. P4a's higher total partly reflects which bias
  happens to match the held-out class mix.
- **Soil-label mapping is inferred** (raw id 0 = soil,
  `adapters/dataset.py::pheno4d_collapse_organ_labels`) and exists only for the 36 annotated
  held-out clouds (22 tomato, 14 maize). The 27 unannotated clouds have no soil share or plant-loss
  values.
- **Point counts before FPS** come from the manifests; they were not recomputed from raw clouds
  (not on the cluster).
- **Plant age is not available.** Growth stage uses the scan date (read as March days, an
  inference from the file names), split into three equal groups of distinct dates per species.
- **Small numbers:** 2 tomato and 2 maize held-out plants, 63 clouds; the p-values are
  descriptive.

---

## 4. Where this leaves the open question

- Removing soil outright (P4) still loses clearly. Down-weighting soil in the segmentation loss
  (learned weights) neither helps nor hurts target accuracy relative to keeping it (P4a).
- The P4-vs-P4a gap is concentrated in early-season clouds, where both models largely fail to use
  plant shape. Outside the earliest scans the two are much closer (0.886 vs 0.935, selected
  checkpoints).
- Not yet tested: whether a soil cut that leaves no remnant on small plants (or removing the
  earliest scans from the evaluation) closes the gap. Not decided: which recipe to propose; this
  is still to be discussed with the user before Stage 4's "proposed strategy" item can be ticked.
