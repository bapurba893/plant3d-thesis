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

DefRec_and_PCM is cloned **outside** this repo, as a sibling directory
(`../DefRec_and_PCM`, i.e. next to `plant3d-thesis/`, both under `/home/pearl/25m0301/` on
the cluster) — pinned at commit `5cb7b797099b0d44abcd00860669a5f95812ae22`. It is used purely
as an unmodified upstream library (imported via `PYTHONPATH`/`sys.path`, e.g. `PointDA.Models`
for the DGCNN/PointNet backbones), the same way you'd depend on a pip package — it is not
version-controlled as part of this repo and should never contain project-specific code. All
project-specific code (adapters, training scripts, job configs, results) lives inside this
repo (`plant3d-thesis/`) so the repo is self-contained and presentable on its own.

**DefRec repo caveats found when integrating (2026-09-09):** requirements.txt pins
torch==1.4.0/numpy 1.x — do NOT `pip install -e .` against it (would downgrade our
torch 2.6.0+cu124); install missing deps (`h5py`, `scikit-learn`, `gdown`) individually
instead. `PointDA/data/dataloader.py` used `np.int` (removed in numpy>=1.24) — patched to
plain `int`. It only implements **PointNet and DGCNN** natively — no PointNet++, so Block A
rows need an external PointNet++ implementation integrated before they can run (tracked as a
task, not yet done). `trainer.py` always trains the DefRec self-supervised loss on target data
unconditionally (that's the paper's whole point) — a true DA-0 (no adaptation) baseline can't
use it as-is, so `adapters/train_c1_dgcnn_da0.py` is a from-scratch-but-thin training loop
(source-only classification, target used for held-out eval only) that imports DGCNN from the
reference repo rather than its trainer.

## Repository layout
- `data/` — CSVs/manifests + `preprocessed/` `.npz` cache (531 files), tracked in git (~25MB).
  Raw Crops3D PLY / Pheno4D txt are NOT in this repo (too large) — see `docs/00_README_download.md`
  to regenerate via `scripts/05_preprocess_pointclouds.py` if needed.
- `scripts/` — Stage 0 data pipeline (numbered `01_`–`05_`, plus `data_io.py`/`augmentations.py`/
  `preprocessing.py` shared modules).
- `adapters/` — project-specific PyTorch `Dataset`/training code that bridges our cached data to
  the DefRec_and_PCM reference backbones (e.g. `dataset.py`, `train_c1_dgcnn_da0.py`).
- `jobs/` — `sbatch` scripts for the Prajna HPC cluster (`--partition=dgx --qos=dgx --gres=gpu:1`;
  interactive `srun` is not permitted for this account).
- `results/` — training logs/checkpoints per experiment (checkpoints gitignored, logs kept).
- `docs/` — strategy table docx, pipeline diagram, dataset download README.

## Current status
Stage 0 — complete (see above). Stage 1 — underway on the Prajna HPC cluster
(GPU: A100 80GB, PyTorch 2.6.0+cu124, conda env `plant3d`; jobs submitted via
`sbatch --partition=dgx --qos=dgx --gres=gpu:1`, interactive `srun` not permitted for this
account). Real Crops3D/Pheno4D were downloaded and preprocessed on a separate laptop (no GPU
there); only the resulting `.npz` cache + CSVs were transferred to the cluster (raw PLY/txt
were not — see Repository layout above).

**Done:**
- Cluster environment confirmed working end-to-end: GPU access verified via `sbatch`
  (A100 80GB), conda env `plant3d` has PyTorch 2.6.0+cu124.
- DefRec_and_PCM cloned as a sibling repo and integrated (see Reference codebase above for the
  compatibility fixes this took). Its own bundled PointDA-10 example (ModelNet→ShapeNet,
  DGCNN+DefRec+PCM) runs clean end-to-end via `sbatch` on the `dgx` partition, confirming the
  environment/repo integration is sound before adapting it to our data.
- **Row C1 (DGCNN, DA-0, ALL) trained and committed** (commit `7b5a139`) — classification-only
  interim (species Tomato-vs-Maize as the `L_cls` proxy; see known gap below), using
  `adapters/train_c1_dgcnn_da0.py`. Finding worth remembering: plain cross-entropy on Crops3D's
  2.7:1 Maize:Tomato imbalance let the model settle into a majority-class shortcut for the first
  several epochs (frozen at exactly the majority-baseline accuracy, `avg_acc=0.5000` — the
  signature of a constant-class predictor); switched `L_cls` to weighted CE
  (`w_c = (f_c+eps)^-1`, matching the `L_seg` weighting pattern already defined in the strategy
  table). Weighted CE recovers faster (100% source val by epoch 6 vs epoch 9 unweighted) though
  final target accuracy is comparable between the two (~89–90%, within noise for n=1 runs on a
  63-sample eval set) — the real benefit is convergence robustness, not a final-accuracy win.
  Both runs' logs are in `results/`.
- **Per-point Crops3D organ segmentation labels backfilled into the `.npz` cache** (2026-09-11).
  Correction to the original plan: `Crops3D_IS` (the variant the old task list pointed at) turned
  out to be the wrong source — per the paper (Zhu et al. 2024) and the `clawCa/Crops3D` /
  `harpreetsahota204/crops3d_to_fiftyone` GitHub repos, it's plot-scale *instance* segmentation
  (which individual plant a point belongs to), covering only Maize/Potato/Rapeseed (no Tomato),
  with no organ granularity — useless for this project's `L_seg`. What `L_seg` actually needs was
  already sitting unused in the base `Crops3D`/`Crops3D_10k` PLYs (already downloaded, already
  used for C1): a `scalar_sf` scalar-field property per point giving the organ category. No
  source anywhere (paper, both GitHub repos, the Voxel51 HF dataset card) publishes a numeric
  value→organ-name table, so the raw integer id is used directly as the class label rather than
  a guessed name — `L_seg` only needs consistent per-species integer ids to train. Extracted it in
  `data_io.py::load_crops3d_ply` (now returns `(pts, labels)`; switched `plyfile` to the primary
  parser since `open3d.io.read_point_cloud` can't see custom scalar fields) and threaded it through
  `05_preprocess_pointclouds.py`; re-ran on the laptop (raw PLYs live there, not on the cluster)
  and transferred the regenerated cache back. Confirmed via a full scan of all 308 cached files:
  **Tomato num_classes=3** (ids `{0,1,2}`; `2` is rare, present in only 12% of files, consistent
  with fruit), **Maize num_classes=6** (ids `{0,...,5}`). Two ids confirmed by RGB fingerprinting
  (id `0` = soil for Maize — clearly brown; each species' dominant-point-count id = leaf); the
  rest are unconfirmed by name and should not be assumed without re-deriving (e.g. the one-off
  `scripts/inspect_crops3d_sf.py` diagnostic, or 3D visualization colored by label).

**Next (in order):**
1. Integrate a PointNet++ backbone (DefRec_and_PCM has none natively — only PointNet/DGCNN) into
   the training loop, adapted to DefRec's `{"cls": ..., "DefRec": ...}` output interface, then
   train row A1 (PointNet++, DA-0, ALL) — deferred behind C1 since DGCNN needed no extra backbone
   work and gave a faster real-data validation.
2. Wire `L_seg` (weighted CE + λ_lov·Lovász, per the strategy table) into the training
   loop/dataset adapter now that per-point Crops3D organ labels are available (see Done below,
   "Per-point Crops3D organ segmentation labels backfilled") — needed before training C2 or any
   row beyond C1, since every strategy-table row specifies `L_cls + L_seg` and C1's
   classification-only result was explicitly an interim scaffold, not the full spec.

## Style notes
- Documents/reports: black and white only, no color.
- Code: individual files preferred over one large monolith; complete replacements over
  partial patches when editing.
- Flag inferences vs. confirmed facts explicitly — don't silently assume.
