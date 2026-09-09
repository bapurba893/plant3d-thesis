# Project: Domain-Invariant, Growth-Aware Trait & Growth Prediction from Plant 3D Point Clouds

M.Tech thesis project, IIT Bombay. Read `docs/Domain_Invariance_Strategy_Table.docx` and
`docs/plant_3d_pipeline_v6_corrected.drawio` for full detail — this file is the summary.

## Problem
Deep learning models on plant 3D point clouds fail to generalize across sensors/datasets
(domain shift). Goal: learn domain-invariant representations that transfer from a labeled
source dataset to an unlabeled target dataset, AND predict plant traits/growth trends using
growth-curve-informed features — not just species classification.

## Datasets
- **Source: Crops3D** — Tomato + Maize, structured-light/RGB-D sensors, controlled greenhouse.
  Fully labeled. Format: PLY files, organized `Crops3D/<Species>/*.ply`.
- **Target: Pheno4D** — Tomato + Maize, laser triangulation scanner, multi-temporal (multiple
  scan days per plant), different greenhouse setup. Unlabeled during adaptation training;
  point-wise soil/stem/leaf labels available for evaluation and the Oracle upper-bound only.
  Format: `.xyz` files, `<Species><NN>/<code>_<date>[_a].xyz`. `_a` suffix = annotated
  (4 cols for tomato: x,y,z,label; 5 cols for maize: x,y,z,label1,label2). Unannotated = 3 cols.
- IMPORTANT: both datasets are greenhouse-acquired. The domain gap is SENSOR + ACQUISITION
  SETUP, not indoor/outdoor — do not describe it as "outdoor" anywhere.

## Backbones (3, each representing a different architecture family)
- **PointNet++** — pointwise-MLP, hierarchical set abstraction. Known weakness: fixed-radius
  ball query is density-sensitive (this is why its own authors proposed multi-scale grouping).
- **KPConv** — convolutional, kernel point convolution. Claimed strength: density-robust due
  to grid-subsampled neighborhoods (not fixed point count). Needs a compiled CUDA extension —
  expect real setup friction here, budget time for it.
- **DGCNN** — graph-based, EdgeConv on dynamic k-NN. Known weakness: unrestricted k-NN lets
  noisy points become spurious graph edges. Also the field-standard encoder in point-cloud UDA
  benchmark literature (PointDA-10) — use it when comparability to published numbers matters.

## Domain-invariance strategies (5 codes, used throughout the strategy table)
- **DA-0** Source-only — no adaptation. Lower bound.
- **DA-A** Adversarial — DANN: shared feature extractor + domain discriminator + Gradient
  Reversal Layer (GRL), plus entropy minimization on unlabeled target predictions. This is the
  ANCHOR method — appears identically across all 3 backbones so cross-backbone differences are
  attributable to architecture, not method.
- **DA-D** Discrepancy — Deep CORAL (aligns feature covariances) or MMD (kernel mean embedding
  distance). No discriminator, no adversarial training — cheaper and more stable.
- **DA-S** Self-supervised — deformation reconstruction (DefRec-style): point cloud is
  deformed, model reconstructs it via Chamfer distance loss, run on BOTH domains so it forces
  domain-general geometric features.
- **DA-O** Oracle — trains directly on labeled target data. NEVER a real candidate strategy —
  exists only as the upper-bound reference point. Do not treat DA-O results as "the best model."

## Augmentation categories (Global/Local taxonomy)
- **G-R** Global rigid: rotation (Z-axis), flip, cubic symmetry
- **G-S** Global similarity: isotropic scaling
- **L-N** Local stochastic: Gaussian jitter
- **L-D** Local structural: RandomCrop3D, CoarseDropout3D
- **ALL** = all four applied jointly
- Pad3D/PadIfNeeded3D is a fixed-size compatibility step, NOT an invariance-inducing
  augmentation — never categorize it alongside the above.

## The 24-row strategy table (full detail in docs/Domain_Invariance_Strategy_Table.docx)
- **Block A (PointNet++)**: A1 DA-0/ALL, A2 DA-A/ALL (anchor), A3 DA-S/ALL, A4 DA-A/L-D
  (isolates density weakness), A5 DA-O/ALL (upper bound)
- **Block B (KPConv)**: B1 DA-0/ALL, B2 DA-A/ALL (anchor), B3 DA-D/ALL, B4 DA-0/L-D
  (deliberately unadapted — confirms claimed density robustness), B5 DA-O/ALL
- **Block C (DGCNN)**: C1 DA-0/ALL (also reproduces published baseline), C2 DA-A/ALL (anchor),
  C3 DA-S/ALL (comparable to PointDA-10), C4 DA-A/L-N (isolates noise weakness), C5 DA-O/ALL
- **Block D (fusion, on best backbone+DA+aug from A-C)**: D1 deep features only, D2 +growth
  curve params, D3 +temporal info, D4 +previous growth stage (full model), D5 D4+Logistic ODE
  physics loss, D6 D4+Gompertz ODE physics loss. Backbone/DA/augmentation MUST stay identical
  across D1-D6 — only the fusion input changes. This isolates each component's contribution.
- **Block E (deployment, optional/lowest priority)**: E1 teacher reference, E2 student+Logit KD,
  E3 student+Feature KD (MobilePointNet)

## Loss architecture (full formulas in the strategy table docx)
Two-regime weighting — do not conflate these:
- **Learned (Kendall uncertainty weighting)**: applies ONLY to cooperative task losses —
  L_cls, L_seg, L_trait, L_growth, and cooperative DA losses (L_CORAL/L_MMD/L_defrec).
- **Fixed/scheduled, NEVER learned**: L_dom (adversarial — a learnable weight would let the
  network cheat by inflating uncertainty to silently disable adaptation), L_ent, L_phys,
  L_mono, L_KD-logit, L_KD-feat.
- L_seg = L_wCE + λ_lov·L_Lovász (class-imbalance-aware + direct mIoU optimization)
- L_trait = L_Huber + λ_corr·(1 − ρ_Pearson) (magnitude + cross-plant ranking)
- Logistic ODE (raw trait space): ε(t) = dŷ/dt − ρ·ŷ·(1 − ŷ/K)
- Gompertz ODE (LOG space, y = ln V — not raw space): ε(t) = dŷ/dt − (α − β·ŷ)
- L_mono applied ONLY to height and stem diameter (leaf area/volume/count can legitimately
  decrease via senescence — do not apply monotonicity there)

## Reference codebase
Build on Achituve et al. (WACV 2021), "Self-Supervised Learning for Domain Adaptation on
Point Clouds" (DefRec) — already implements source-only/DANN-style/self-supervised training
on PointNet/DGCNN with the source-target benchmark protocol this project follows. Do not
reimplement DANN or the training loop from scratch; adapt this repo.

## Current status
Stage 0 — complete. Nothing trained yet, but the full data pipeline is real, verified, and
ready:
- Real Crops3D (308 PLY files: 225 Maize + 83 Tomato) and real Pheno4D (223 files: 83 Maize +
  140 Tomato) are downloaded, integrity-verified (size + MD5 against source checksums), and in
  place under `data/crops3d/<Species>/*.ply` and `data/pheno4d/<Plant>/*.txt`. Note: Pheno4D's
  real archive ships `.txt` files, not `.xyz` as some secondary sources describe — same column
  layout (3/4/5 cols), `data_io.py` accepts both extensions.
- Scripts `01_dataset_statistics.py` through `04_augmentation_sanity_check.py` all run clean
  against the real data (previously only tested against synthetic placeholder data).
- `05_preprocess_pointclouds.py` (new) is built and run end-to-end: statistical outlier removal
  (k=20, std_ratio=2.0) → voxel-grid pre-decimation (deterministic, not random — only for FPS
  speed on million-point raw scans) → true farthest-point sampling → normalize/center, cached
  as `.npz`. Full run: 531/531 files processed, 0 failures, target_n=4096, ~3.8% mean outlier
  removal across both datasets. Cache + manifest at `data/preprocessed/`.
- `03_train_test_split.py` produces the leakage-safe split: Crops3D stratified train/val, and
  Pheno4D split BY WHOLE PLANT (2 plants/species held out entirely) so no plant's time-series
  crosses the adaptation-pool/held-out-eval boundary.
- `02_visualize_pointclouds.py` produced a verified source-vs-target comparison image
  (`data/comparison_tomato.png`) using maturity-matched samples from both datasets (an earlier
  version accidentally paired a young Pheno4D seedling scan with a mature Crops3D sample,
  conflating growth stage with sensor domain — regenerated to isolate the actual domain gap).

**Hardware constraint (important):** this development machine has no NVIDIA GPU — Intel Iris Xe
integrated graphics only, no CUDA. Fine for the CPU-bound data/preprocessing work above, but
actual training (all 3 backbones, and especially KPConv's compiled CUDA extension) needs a
different machine — university cluster/HPC or a cloud GPU. Decide this before DefRec setup.

Immediate next steps: get the DefRec reference repo running on its own example (on whatever
GPU machine is chosen), then adapt it to this project's cached `data/preprocessed/` arrays and
train row A1 (PointNet++, DA-0, ALL augmentation) first as the simplest possible end-to-end
validation before scaling to the remaining 23 rows.

## Style notes
- Documents/reports: black and white only, no color.
- Code: individual files preferred over one large monolith; complete replacements over
  partial patches when editing.
- Flag inferences vs. confirmed facts explicitly — don't silently assume.
