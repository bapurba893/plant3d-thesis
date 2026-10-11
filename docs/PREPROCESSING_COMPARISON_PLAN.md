# Preprocessing Comparison Plan (Block PP)

Status file for the preprocessing comparison requested by Prof. Soumyashree Kar on 2026-10-09.
Claude: read this whole file before doing any work in this block. Update the **Status
checklist** (section 9) every time a step is finished. Never tick a box that is not verified.

---

## 1. Goal

Maam's instruction (2026-10-09): compare the different preprocessing pipelines found in the
literature on the **baseline model only**, report a comparative assessment, and from that
comparison derive our own preprocessing strategy. The rest of the framework stays as it is.

Deliverable for maam: one comparison table (section 7) with every pipeline's baseline result,
plus a short note on which preprocessing steps matter and what our proposed strategy is.

## 2. Corrections to project facts (fix CLAUDE.md first)

`CLAUDE.md` currently says Crops3D is "structured-light/RGB-D, controlled greenhouse" and
that "both datasets are greenhouse-acquired". That is wrong. Per the Crops3D paper
(Zhu et al., Scientific Data 2024, https://www.nature.com/articles/s41597-024-04290-0):

- **Crops3D maize**: terrestrial laser scanning (FARO Focus S70), **field**, labelled classes
  soil, pot, stem, leaves, tassel, fruit.
- **Crops3D tomato**: structure-from-motion photogrammetry (Canon 77D), greenhouse,
  **no soil points** (also found in `step_notes/D0_Trait_Extraction_Pipeline.md`).
- **Pheno4D** (Schunck et al., PLOS ONE 2021): Perceptron ScanWorks V5 laser triangulation
  scanner on a measuring arm; coordinates are in the **arm's local frame** (up-axis not
  guaranteed to be Z); clouds **include parts of pot and soil** for both species.

Consequence: in the source domain, species is confounded with sensor, field vs greenhouse,
soil/pot presence and point density. This is the main hypothesis this block tests.

## 3. Measured problems in the current pipeline (v1 = P0)

From `data/preprocessed/preprocessed_manifest.csv`:

- Raw point count alone separates Crops3D species with AUC 0.988 (threshold rule: 96.4%).
- 102 of 225 Crops3D maize clouds have <= 50,000 points after outlier removal, so they skip
  the voxel step and go straight to FPS. No Crops3D tomato and no Pheno4D cloud does. The
  sampling path is therefore tied to the class label in the source only.
- Soil/pot is never removed. Up-axis is assumed to be Z and never checked.
- `norm_scale` / `norm_center_*` are recorded for Pheno4D only, not for Crops3D.

## 4. Fixed experimental setup (identical for every pipeline)

- **Model**: row C1 — DGCNN, DA-0 (no adaptation), ALL augmentation, L_cls + L_seg.
  Script: `adapters/train_c1_dgcnn_da0.py`. Change only what is needed to point it at a
  different cache manifest.
- **Same everything else**: splits (`data/crops3d_train.csv`, `crops3d_val.csv`,
  `pheno4d_heldout_eval.csv`), 100 epochs, batch 32, lr 1e-3, wd 5e-5, dropout 0.5.
- **Points per cloud**: 4,096 for every pipeline. Do not vary point count in this block.
- **Seeds**: 1, 2, 3 for every pipeline.
- **Selection rule**: unchanged — lowest source validation loss. Target labels are used only
  for reporting, never for selection.
- **Do not modify or delete** the existing `data/preprocessed/` cache, existing `results/`
  folders, or any Block A–D script. All new work goes in new files and new folders.

## 5. Pipelines to compare

All pipelines end with exactly 4,096 points. "Soil/pot removal" is defined in section 6.

| ID | Based on | Steps (in order) |
|---|---|---|
| P0 | Current pipeline (v1) | Statistical outlier removal (k=20, 2.0 std) -> voxel (diag/200) only if > 50k points -> FPS 4096 -> centre + unit sphere. **Reuse the existing cache; do not regenerate.** |
| P1 | DefRec / PointDA-10 (Achituve et al., WACV 2021) | Outlier removal -> align up-axis to +Z -> voxel (diag/200) for every cloud -> FPS 4096 -> centre + scale to unit **cube** |
| P2 | Pheno4D baselines / AGS-PlantSeg (Salazar & Nalpantidis 2026) | Outlier removal -> soil/pot removal -> **random** sampling to 4096 (no FPS) -> centre + unit sphere |
| P3 | Eff-3DPSeg (Luo et al. 2022) | Outlier removal -> soil/pot removal -> voxel at a fixed size relative to plant height (section 6) -> FPS 4096 -> centre + unit sphere |
| P4 | Our proposed v2 | Outlier removal -> soil/pot removal -> align up-axis to +Z -> voxel at height-relative size for **every** cloud -> FPS 4096 -> centre + unit sphere computed **after** soil removal; save norm_scale/center for both datasets |

### Ablations of P4 (one step removed at a time)

| ID | Same as P4 except |
|---|---|
| P4a | No soil/pot removal |
| P4b | Old conditional sampling path (voxel only if > 50k points) |
| P4c | No up-axis alignment |
| P4d | Normalise before soil removal (soil included in the scale) |

**P3 = P4 = P4c on this data (verified 2026-10-11, Stage 4).** All 531 cached clouds have
identical data arrays (points, labels, normalisation, point counts) in the three caches; only the
`pipeline_id` / `up_axis_aligned` metadata fields differ, and the three pipelines' C1 `run.log`s
are line-for-line identical per seed once timestamps and run names are removed. Reason: up-axis
alignment is a no-op here (Crops3D maize and Pheno4D are +z; Crops3D tomato is trusted as +z,
`TOMATO_UP_AXIS_ALIGN = False`), so no cloud is rotated. Count them as **one** pipeline (7
distinct pipelines, not 9). The P4c ablation therefore says nothing about up-axis alignment. See
`step_notes/PP_2_Results.md` caveat 3.1.

Optional, only if P4 does not clearly beat P0: **P5** = P4 + noise matching (add the same
small Gaussian jitter to every cloud at preprocessing time). Decide with the user first.

Pipelines not included, with reason (state this in the report): OmniPlantSeg's
full-resolution KD-tree block subsampling suits per-point segmentation of large clouds, not
one label per whole cloud, so it is out of scope for the baseline classification comparison.

## 6. Definitions of the shared steps

- **Soil/pot removal**
  - Crops3D maize: drop the soil id (CONFIRMED id 0) and the pot id (CONFIRMED id 5 — identified
    in Stage 1 by hollow-ring geometry and colour; see step_notes/PP_0_Prerequisite_Checks.md
    check 1). Uses Crops3D's own source labels — fine, since Crops3D is the labeled source
    dataset, not the unlabeled target.
  - Crops3D tomato: nothing to remove (CONFIRMED no soil id and no unlabeled ground-like layer
    either; check 3).
  - **Pheno4D, ALL scans, annotated or not: a single LABEL-FREE geometric cut, never using soil
    labels** — corrected 2026-10 from the original "use labels on `_a` scans, RANSAC on
    unannotated scans" split. Rationale: Pheno4D is the **target** dataset; using its own soil
    labels inside preprocessing (even just on the `_a` subset) would put target labels into a
    step that the rest of the pipeline treats as unsupervised, silently leaking supervision the
    adaptation protocol is supposed to do without. Labels are used **only** to validate the cut
    (score soil-removed%/plant-points-lost% on the `_a` scans) — never to compute it, on `_a`
    scans or otherwise. Method (replaces the tilted-plane-RANSAC-on-lowest-15%-of-points
    approach, which failed on late-growing-season scans where drooping leaves/stem dominate the
    lowest 15% of *points*): find the height z0 of the densest thin horizontal slab that also
    has a wide lateral footprint — restricting the search to the bottom 30% of the scan's
    *physical z-range* (a geometric fact about the scene: soil sits below the plant it supports;
    not a label), not the bottom 30% of *points* (which is what let drooping plant material
    dominate the old approach) — then cut at z0 plus a margin. Validated on all 126 annotated
    scans; see step_notes/PP_0_Prerequisite_Checks.md Part C for the numbers, the worst scans,
    and whether the > 95% soil removed / < 1% plant points lost starting target is reached.
- **Up-axis alignment**: determine per dataset (not per cloud) which axis is up by rendering
  about 5 clouds per species per dataset. Store the rotation in a config/constant with a
  comment saying how it was verified.
- **Height-relative voxel size**: voxel = plant height / 200 measured after soil removal and
  alignment. Rationale: Crops3D tomato is photogrammetry and may have arbitrary scale, so a
  voxel size in mm is not safe across datasets. Check in Stage 1 whether Crops3D tomato is in
  metric units; if it is, a fixed mm voxel is an option to discuss with the user.

### 6a. Manifest portability bug, found and fixed (2026-10-11)

`scripts/07_preprocess_variants.py` originally wrote `source_filepath` and `cache_path` into
each variant manifest as **absolute, laptop-specific paths** (from `data/dataset_manifest.csv`'s
own absolute raw-data root, and from `--out_dir.resolve()`). This was not just a cluster-transfer
problem — it broke `PlantClsSegDataset`/`PlantSpeciesDataset` on **any** machine, including this
laptop: `adapters/dataset.py` matches a split CSV's `filepath` (e.g. `../data/crops3d/Maize/
3-54.ply`) against the manifest's `source_filepath` by exact string (after stripping a leading
`../`), and an absolute path never matches that relative form — confirmed directly, 0 of 263
`crops3d_train.csv` rows matched before the fix. Separately, `pathlib`'s `/` operator *discards*
its left operand whenever the right operand is already absolute, so `_REPO_ROOT / cache_path`
silently resolved to the original laptop path even when `_REPO_ROOT` was a different machine's
checkout — never an error, just a missing file, which `PlantClsSegDataset` treats as "skip this
row," i.e. a silently empty dataset rather than a crash.

Fixed in `scripts/07_preprocess_variants.py` (`to_portable_source_filepath`/
`to_portable_cache_path`, used for all future runs) and applied as a manifest-only rewrite to all
8 already-generated manifests (no `.npz` file touched). Re-verified after the fix: 100% of
`crops3d_train.csv`/`crops3d_val.csv`/`pheno4d_heldout_eval.csv` rows resolve to an existing
cache file through every one of the 8 variant manifests.

## 7. What to report for each pipeline (the table for maam)

For each pipeline: mean ± std across 3 seeds of

1. Full-run mean target accuracy (headline number)
2. Full-run mean target balanced accuracy
3. Selected-checkpoint target accuracy (protocol-selected)
4. Source segmentation mIoU, tomato and maize
5. Shortcut score: logistic regression on non-shape features (raw point count, mean
   nearest-neighbour spacing of the 4096 points, flatness of the lowest 10% of points),
   trained on Crops3D species, tested on Pheno4D. Report source accuracy and target accuracy.

Reference lines in the table: trivial majority baseline on held-out = 40/63 = 0.635;
current P0 single-seed C1 = 0.791 full-run mean (std 0.104 across epochs).

Decision rule: a pipeline beats P0 only if all 3 of its seeds have a higher full-run mean
target accuracy than P0's 3-seed mean. Report the rest as "no clear difference".

## 8. Files to create (names fixed to avoid collisions)

- `scripts/preprocessing_variants.py` — the pipeline steps as functions + a `PIPELINES` dict.
- `scripts/07_preprocess_variants.py` — CLI: `--pipeline P1|P2|P3|P4|P4a|P4b|P4c|P4d`,
  writes `data/preprocessed_variants/<ID>/` with `.npz` files and its own
  `preprocessed_manifest.csv` (same columns as the existing one).
- `scripts/08_soil_removal_check.py` — validates the RANSAC soil cut on annotated Pheno4D.
- `scripts/09_shortcut_probe.py` — the shortcut score (section 7, item 5). Runs on the laptop
  from each pipeline's cache/manifest; it is NOT a training metric and is not in `run.log`.
- `scripts/10_pp_summarize.py` — parses `results/PP_*/run.log`, writes to `results/pp_summary/`:
  `pp_runs.csv` (one row per run), `pp_by_pipeline.csv` (mean/std per pipeline + decision),
  `pp_summary.txt` (plain-text tables) and `pp_summary.md` (the table for maam). The shortcut
  score is not included; it comes from `scripts/09_shortcut_probe.py`.
- `adapters/train_c1_dgcnn_da0.py` — add a `--manifest` argument (default = current path,
  so old behaviour is unchanged). Nothing else changes.
- `jobs/pp_compare.sbatch` — Slurm job array over pipeline × seed, `--time=02:00:00`,
  `--exp_name PP_<ID>_s<seed>`, same partition/qos/conda lines as `jobs/c1_dgcnn_da0_clsseg.sbatch`.
- `step_notes/PP_0_Prerequisite_Checks.md`, `step_notes/PP_1_Pipelines.md`,
  `step_notes/PP_2_Results.md` — findings for each stage.

Where things run: raw Crops3D/Pheno4D are only on the laptop (`/home/apurba/plant3d-data`),
so all preprocessing runs on the laptop. Training runs on Prajna. Transfer the new caches the
same way the existing cache was transferred (via git, as for `data/preprocessed/`).

## 9. Status checklist (Claude: keep this updated)

### Stage 0 — Setup
- [x] CLAUDE.md dataset facts corrected (section 2) and pointer to this plan added
### Stage 1 — Prerequisite checks (laptop, CPU) -> `step_notes/PP_0_Prerequisite_Checks.md`
- [x] Crops3D maize pot label id identified
- [x] Pheno4D soil label id confirmed (tomato and maize label columns)
- [x] Crops3D tomato confirmed to have no soil
- [x] Up-axis per dataset determined from renders; rotation recorded. Maize +z and Pheno4D +z
      CONFIRMED via renders. **Crops3D tomato: DECIDED 2026-10-11 — Option A (trust the raw +z
      convention, no realignment; `TOMATO_UP_AXIS_ALIGN = False` in
      `scripts/preprocessing_variants.py`).** Visually verified on 36 of 83 clouds via
      `results/pp_checks/tomato_axis_montage.png` — the 24 clouds the shape heuristics disagreed
      on most, plus 12 random controls — all 36 render upright. See step_notes/
      PP_0_Prerequisite_Checks.md Part C for the full decision record and the acknowledged
      residual uncertainty (36 of 83 checked, not all).
- [x] Crops3D tomato units/scale checked (checked; result is INFERRED not CONFIRMED — see step note)
- [x] Label-free horizontal-slab soil cut tuned and validated on ALL 126 annotated Pheno4D scans
      (numbers recorded; **target REACHED** at margin=2.0mm: 95.32% soil removed, 0.377% plant
      lost, pooled — see step note Part C. Supersedes the original tilted-plane-RANSAC approach,
      which never reached the target; replaced because it also used soil labels on annotated
      scans during preprocessing, which Pheno4D being the target dataset ruled out — see section 6)
### Stage 2 — Implement pipelines -> `step_notes/PP_1_Pipelines.md`
- [x] `scripts/preprocessing_variants.py` + `scripts/07_preprocess_variants.py` written
- [x] Smoke test `--limit 5` per pipeline passes (all 8, zero failures) — since superseded by the
      full generation itself finishing cleanly (stronger evidence than the smoke test alone).
      **Not done: before/after renders** — flagging this explicitly rather than ticking a
      two-part box half-true; no visual before/after comparison was made for the variant
      pipelines (unlike the shape/NaN/manifest-consistency checks below, which were done).
- [x] All 8 caches generated (P1–P4, P4a–P4d); point counts all = 4096 (verified: 531/531 rows
      per pipeline manifest, zero `n_final != 4096`, zero nulls in `n_final`/`norm_scale`, 120
      randomly-sampled `.npz` files checked directly for shape/NaN/Inf — zero problems). Known,
      confirmed limitation carried through correctly: 9 of 225 Crops3D maize clouds have neither
      the soil id (0) nor the pot id (5) at all (step_notes/PP_0 check 1) — those 9 show
      `n_soil_pot_removed == 0` in every pipeline that removes soil/pot (216/225 show > 0,
      matching exactly), which is the correct behaviour for clouds with nothing to remove, not a
      bug.
- [x] `--manifest` argument added to C1 script; old default verified unchanged (2026-10-11, by
      diff: omitted -> same hardcoded path; only side effect is `manifest=None` in the logged args)
- [x] `jobs/pp_compare.sbatch` written (2026-10-11; 27-task array, mapping checked with
      `PP_DRY_RUN=1`). **Variant manifest file name CONFIRMED 2026-10-11: `preprocessed_manifest.csv`**
      (matches the sbatch script's own `VARIANT_MANIFEST_NAME` guess) — verified by a full
      round-trip check against `adapters/dataset.py`'s actual lookup logic, not just by filename:
      100% of `data/crops3d_train.csv`/`crops3d_val.csv`/`pheno4d_heldout_eval.csv` rows resolve
      to an existing `.npz` through every one of the 8 variant manifests. This check caught and
      fixed a real bug first — see section 6a below.
- [ ] Caches committed/transferred to Prajna (explicitly deferred — not done this pass)
### Stage 3 — Run (Prajna)
- [x] P0 run with seeds 2 and 3 (seed 1 = existing C1 result if identical settings; else rerun)
      — seed 1 was rerun as `results/PP_P0_s1` through the new `--manifest` path and reproduces
      the existing C1 result exactly (full-run mean 0.7913, selected epoch 95, acc 0.746, Tomato
      mIoU 0.3248, Maize mIoU 0.3427)
- [x] P1–P4 × 3 seeds finished (all `complete=True`, 100 epochs, `results/pp_summary/pp_runs.csv`)
- [x] P4a–P4d × 3 seeds finished (same check)
### Stage 4 — Analyse -> `step_notes/PP_2_Results.md`
- [x] Shortcut probe run for every pipeline (`results/pp_summary/shortcut_probe.md`, all 9)
- [x] `results/PP_summary.md` table produced — actual path `results/pp_summary/pp_summary.md`
      (per section 8's `scripts/10_pp_summarize.py` output layout); results and caveats in
      `step_notes/PP_2_Results.md`
- [ ] Proposed strategy written up (which steps matter, from the ablations) — findings on which
      steps matter are in `step_notes/PP_2_Results.md`, but no strategy chosen yet: the open
      question there (soil/pot may be a species cue in both datasets, caveat 3.4) should be
      discussed with the user first
- [x] PROGRESS_LOG.md milestone added (milestone 41)
- [ ] Table + note sent to maam

## 10. Rules for this block

- One stage at a time; stop and show results to the user at the end of each stage.
- Every number in a step note must come from a file or command output, not memory.
- Flag inferences as inferences (e.g. "consistent with", "not yet tested").
- Do not change Block A–D scripts other than the `--manifest` argument in C1.
- Do not use Pheno4D labels for anything except soil-cut validation and final reporting.
