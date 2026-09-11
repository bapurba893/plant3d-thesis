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

DefRec_and_PCM has no PointNet++ (see caveat above), so Block A's backbone is vendored
separately: **yanx27/Pointnet_Pointnet2_pytorch**, cloned the same way as DefRec_and_PCM — a
sibling directory (`../Pointnet_Pointnet2_pytorch`, next to `plant3d-thesis/` and
`DefRec_and_PCM/`, all under `/home/pearl/25m0301/` on the cluster), pinned at commit
`eb64fe0b4c24055559cea26299cb485dcb43d8dd`, unmodified upstream, imported via
`POINTNET2_ROOT`/`sys.path` (`adapters/models_pointnet2.py` inserts its `models/` subdirectory
onto `sys.path` and imports `pointnet2_utils` directly — not version-controlled as part of this
repo, never to contain project-specific code, same rules as DefRec_and_PCM.

**PointNet++ repo compatibility check (2026-09-11):** unlike KPConv, this implementation is
**pure PyTorch — no compiled CUDA extension needed**; its farthest-point-sampling/ball-query/
feature-propagation ops in `models/pointnet2_utils.py` are plain torch tensor code, confirmed
by running them directly against our data shape (`(B, 3, 4096)`) under torch 2.6.0+cu124 +
numpy 2.2.6 with no errors. Its own driver scripts (`train_partseg.py`/`train_semseg.py`/
`test_*.py`) use the removed `np.float` alias, but nothing here imports those — only
`pointnet2_utils.py`'s two building-block classes are used directly, which have no numpy
dependency. One integration snag: `models/` has no `__init__.py` upstream, and their own
`from models.pointnet2_utils import ...` style collides with this repo's own flat
`adapters/models.py` module (also named `models`) once both are on `sys.path` — worked around
by adding the vendored repo's `models/` subdirectory itself to `sys.path` and importing
`pointnet2_utils` by its unique name instead (see `adapters/models_pointnet2.py` docstring).

`adapters/models_pointnet2.py::PointNet2_ClsSeg` composes `PointNetSetAbstraction`/
`PointNetFeaturePropagation` from the vendored repo (3-level SSG set-abstraction encoder for
the pooled global feature, mirroring `pointnet2_cls_ssg.py`; 3-level feature-propagation
decoder for full-resolution per-point features, mirroring `pointnet2_sem_seg.py` — not
`pointnet2_part_seg_ssg.py`, which conditions on a ShapeNet category label we have no
equivalent for) with the same `PointSegDA.Models.segmentation` per-species head reused for
`DGCNN_ClsSeg`, exposing the identical `{"cls": ..., "seg_feat": ...}` forward interface (raw
logits, not log-softmax) so `adapters/train_a1_pointnet2_da0.py` mirrors
`adapters/train_c1_dgcnn_da0.py` almost line-for-line — only the backbone differs. DefRec/DANN
are not wired for this model yet (row A1 is DA-0, same scope C1 shipped with); needed when
rows A2/A3/A4 reach the PointNet++ track. Full CPU smoke test (1 epoch, real Crops3D/Pheno4D
data) passed end-to-end before this was considered done.

## Repository layout
- `data/` — CSVs/manifests + `preprocessed/` `.npz` cache (531 files), tracked in git (~25MB).
  Raw Crops3D PLY / Pheno4D txt are NOT in this repo (too large) — see `docs/00_README_download.md`
  to regenerate via `scripts/05_preprocess_pointclouds.py` if needed.
- `scripts/` — Stage 0 data pipeline (numbered `01_`–`05_`, plus `data_io.py`/`augmentations.py`/
  `preprocessing.py` shared modules).
- `adapters/` — project-specific PyTorch `Dataset`/training code that bridges our cached data to
  the DefRec_and_PCM/Pointnet_Pointnet2_pytorch reference backbones (e.g. `dataset.py`,
  `train_c1_dgcnn_da0.py`, `train_a1_pointnet2_da0.py`).
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
- **Row C1 (DGCNN, DA-0, ALL), full spec (`L_cls + L_seg`), trained and committed** — supersedes
  the earlier classification-only interim result. Uses `adapters/train_c1_dgcnn_da0.py` +
  `adapters/models.py::DGCNN_ClsSeg` (backbone from `PointDA.Models.DGCNN`, per-species seg
  heads shaped after `PointSegDA.Models.segmentation`, both composed rather than reimplemented
  — see that file's docstring) + `adapters/losses.py` (Lovász-softmax, `λ_lov=1.0` per the
  strategy table docx, and Kendall uncertainty weighting combining `L_cls`/`L_seg` — both
  classification-type per the docx's Level-2 formula, `exp(-s_j)·L_j + s_j/2`). Segmentation is
  per-species (Tomato num_classes=3, Maize num_classes=6 — see the label-backfill entry above);
  `L_cls` keeps the weighted-CE fix from the original interim run (plain CE let the model settle
  into a majority-class shortcut on Crops3D's 2.7:1 imbalance).

  **Final numbers** (100 epochs, best model selected by lowest combined source val loss —
  correctly never touches target labels, matching DA-0's protocol — landed at epoch 95):
  source val cls acc 1.0000, target (Pheno4D) cls acc 0.7460 (avg acc 0.7446), Tomato seg
  mIoU 0.3248 (acc 0.6645), Maize seg mIoU 0.3427 (acc 0.7698).

  **Target cls accuracy dropped vs. the old interim result (0.8889 → 0.7460) — investigated,
  and it is NOT evidence that adding L_seg hurts target transfer.** Epoch 6 of this same run
  (the epoch the old interim script happened to select) shows target acc 0.9206 — matching or
  beating the old result. What actually happens: target accuracy drifts steadily downward over
  the full 100 epochs (settling frozen at exactly 0.7460 for the last ~10 epochs) while source
  val cls stays saturated at 1.0 and seg loss keeps improving the whole time. So under DA-0,
  continued training keeps helping the only signals model-selection is allowed to see (source
  loss), while target generalization quietly erodes in the background with nothing to detect or
  prevent it — a real, expected DA-0 characteristic (motivates why C2's DANN anchor exists), not
  a training bug and not a reason to change the selection criterion (selecting on target labels
  would defeat the point of DA-0 as a lower bound).
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
- **PointNet++ backbone integrated** (2026-09-11) — see Reference codebase above for the
  vendored repo, compatibility check, and how `adapters/models_pointnet2.py::PointNet2_ClsSeg`
  composes it into the same `{"cls": ..., "seg_feat": ...}` interface as `DGCNN_ClsSeg`.
  `adapters/train_a1_pointnet2_da0.py` + `jobs/a1_pointnet2_da0.sbatch` are ready; a 1-epoch
  CPU smoke test on real Crops3D/Pheno4D data passed end-to-end (sane non-zero cls/seg
  metrics, no shape/gradient errors) — since trained for real on the GPU, see below.
- **Row A1 (PointNet++, DA-0, ALL) trained on the cluster GPU** (2026-09-11, job 308286) —
  same 100-epoch setup, data split, and DA-0 protocol as C1, giving the first real
  cross-backbone comparison. Best model at epoch 97 (selected by lowest source val total loss,
  same protocol as C1).

  **Final numbers:** source val cls acc 1.0000 (avg acc 1.0000), target (Pheno4D) cls acc
  0.4603 (avg acc 0.5750), Tomato seg mIoU 0.3207 (acc 0.6729), Maize seg mIoU 0.4741
  (acc 0.9055).

  **Comparison to C1 (DGCNN, DA-0, ALL):**

  | Metric | C1 (DGCNN) | A1 (PointNet++) |
  |---|---|---|
  | Source val cls acc | 1.0000 | 1.0000 |
  | Target cls acc | 0.7460 (avg 0.7446) | 0.4603 (avg 0.5750) |
  | Tomato seg mIoU | 0.3248 (acc 0.6645) | 0.3207 (acc 0.6729) |
  | Maize seg mIoU | 0.3427 (acc 0.7698) | 0.4741 (acc 0.9055) |

  PointNet++ transfers markedly worse to target classification (confusion matrix shows 34/40
  target Tomato plants misclassified as Maize — a systematic bias, not just noise), while
  segmenting Maize distinctly better than DGCNN; Tomato seg mIoU is essentially tied between
  the two backbones. A1 shows the same DA-0 drift pattern documented for C1 above (target acc
  peaked ~0.92-0.94 around epoch 8-9, then eroded over the full 100 epochs while source metrics
  kept improving) — confirms the drift is a general DA-0 characteristic, not DGCNN-specific,
  reinforcing the case for the A2/C2 adversarial anchor rows.

**Next (in order):**
1. Train row C2 (DGCNN, DA-A, ALL) — the adversarial anchor. Adds a domain discriminator +
   Gradient Reversal Layer + entropy minimization on top of the now-working joint `L_cls+L_seg`
   DGCNN adapter; `L_dom`/`L_ent` are fixed/scheduled weights, never learned (see Loss
   architecture above) — do not route them through the Kendall uncertainty module used for
   `L_cls+L_seg`. Natural next step on the DGCNN track: isolated, one-variable-at-a-time
   increment on top of C1 rather than a rewrite.

## Style notes
- Documents/reports: black and white only, no color.
- Code: individual files preferred over one large monolith; complete replacements over
  partial patches when editing.
- Flag inferences vs. confirmed facts explicitly — don't silently assume.
