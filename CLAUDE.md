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
  to grid-subsampled neighborhoods (not fixed point count). Needs compiled C++ extensions for
  grid subsampling/radius neighbor search — CPU-only, not CUDA, confirmed while integrating
  (2026-09-11): they built and ran fine on the cluster login node with no GPU present; the
  KPConv convolution itself is plain PyTorch. Still expect real setup friction (numpy/setuptools
  version incompatibilities in the vendored repo's ~2020-era C-extension code), just not a CUDA
  one — see `step_notes/B1_KPConv_DA0.md`.
  **IMPORTANT — wall-clock budget: every Block B (KPConv) `sbatch` job needs `--time=60:00:00`
  (not the ~4h that suffices for DGCNN/PointNet++ rows), from the start, not discovered per row.**
  KPConv's collate step (grid subsampling + radius neighbor search) is CPU-bound and
  single-threaded (no OpenMP in the vendored code; a `DataLoader(num_workers>0)` parallelization
  attempt made things worse, not better — reverted, see B1's incident below), and B1 additionally
  hit real GPU/node contention on the shared `dgx` partition (per-batch fwd+bwd swung 0.2s-86s;
  a first attempt at 24h was itself too small and had to be killed and resubmitted). This is a
  backbone-specific cost, not a bug — budget for it on B2-B5 up front rather than re-deriving it.
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
  (deliberately unadapted — confirms claimed density robustness), B5 DA-O/ALL. **B3b DA-S/ALL
  — ADDED 2026-09-13, outside the original 25-row count, per explicit user instruction.** Block
  B was originally specced without a DA-S row (unlike Block A/C, where the third row IS DA-S) —
  B4 was meant to be Block B's own density-robustness check instead. B3b exists specifically to
  test cross-backbone comparability with A3/C3: A3 showed DA-S's effect on target-classification
  transfer flips direction entirely between PointNet++ (helps) and DGCNN (hurts) — see
  `step_notes/A3_PointNet2_DA_S.md` — and B2 then showed DA-A's effect *also* doesn't generalize
  cleanly to KPConv (DANN hurts DGCNN/PointNet++ but not KPConv overall — see
  `step_notes/B2_KPConv_DA_A.md`). B3b asks whether DA-S flips for KPConv the same way DA-A did,
  or behaves like one of the other two backbones. Does NOT replace B3 (DA-D), which stays in the
  plan as originally specified and runs separately. See `step_notes/B3b_KPConv_DA_S.md` for full
  design decisions, including the real KPConv-specific engineering problem this row required
  solving (rebuilding a fresh multi-layer neighbor structure for DefRec's deformed points, since
  unlike DGCNN/PointNet2, KPConv precomputes that structure outside the model rather than
  recomputing it automatically inside `forward`).
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

**Done (continued):**
- **Row C2 (DGCNN, DA-A, ALL) trained on the cluster** (2026-09-11, job 308389, ~13.5 min).
  Adds a domain discriminator + Gradient Reversal Layer (`adapters/dann.py`, new — confirmed
  nothing DANN-like exists in DefRec_and_PCM to adapt, it only implements DefRec self-supervised
  + PCM mixup) + entropy minimization on top of C1's working joint `L_cls+L_seg` DGCNN adapter
  (`adapters/train_c2_dgcnn_da_a.py`). `L_dom`/`L_ent` fixed/scheduled per the loss architecture
  above (GRL alpha = `λ_p = 2/(1+e^(−10p))−1` per the strategy table docx, `L_dom` coefficient
  left at 1 in the total loss to avoid double-applying `λ_p`; `L_ent` uses a fixed, non-ramped
  constant — both flagged as explicit inferences, not literal spec text, in `adapters/dann.py`'s
  docstring), never routed through Kendall. `DGCNN_ClsSeg.forward` now also exposes
  `logits["feat"]` (pooled global feature) as the discriminator's input. CPU smoke test (1 epoch,
  real data) passed before submitting.

  **First run (job 308389, unramped `λ_ent`) — SUPERSEDED, see fix below.** Protocol-selected
  checkpoint (epoch 49) showed target cls acc 0.9365, but full-trajectory analysis (every
  epoch's target-acc diagnostic, never used for training/selection) revealed this was misleading:
  C2 was far more volatile than C1 (stdev 0.214 vs 0.104) and, once the GRL's `λ_p` schedule
  saturated (~epoch 30-50), back-half target accuracy averaged only 0.456 — worse than C1's
  0.77, not better. Root-caused (not just described as "adversarial instability" in the
  abstract): **Crops3D (source, 73% Maize) and Pheno4D (target, 62-63% Tomato) have opposite
  class majorities**, and 23 of C2's 100 epochs showed target predictions collapsed to source's
  majority class (avg_acc≈0.5, acc≈23/63) while entropy loss fell from 0.32→0.05 — entropy
  minimization was successfully forcing confident predictions, just often confident in the
  source-biased direction, because `λ_ent` was applied at full fixed strength from epoch 0
  despite `dann.py`'s own docstring already warning this was risky.

  **Fix (2026-09-11) and rerun (job 308608):** `adapters/dann.py` gained
  `lambda_ent_schedule(p, lambda_ent_max, gamma) = lambda_ent_max * lambda_p_schedule(p, gamma)`
  — `λ_ent` now ramps alongside the GRL alpha instead of being held fixed. Since this changes
  the anchor method itself, C2 was rerun (required before C4, which reuses the same `dann.py`,
  per CLAUDE.md's "identical across all 3 backbones" rule) rather than left to diverge from C4.

  **Corrected protocol-selected checkpoint** (epoch 66, source val total loss 0.9099):

  | Metric | C1 (DA-0) | C2 v1 (unramped, buggy) | C2 v2 (ramped, fixed) |
  |---|---|---|---|
  | Target cls acc | 0.7460 (avg 0.7446) | 0.9365 (avg 0.9130) | 0.6667 (avg 0.7005) |
  | Full-run mean target acc | 0.791 | 0.593 | **0.633** |
  | Full-run stdev | 0.104 | 0.214 | **0.117** |
  | Epochs collapsed to one class | not re-checked | 23/100 | **5/100** |

  **The fix worked exactly as diagnosed — stdev roughly halved, collapse-epochs cut 4.6x — but
  did NOT make DA-A beat DA-0's full-trajectory mean** (0.633 vs. 0.791; quartile means still
  drift down 0.691→0.678→0.608→0.554, a gentler version of the same decline, not a reversal).
  **Conclusion: the entropy-ramp fix resolved a real, specific, now-confirmed bug (unramped
  entropy minimization interacting with a source/target label-prior mismatch), and makes DA-A
  rows behave predictably with trustworthy selected checkpoints — it does not, on its own, make
  vanilla adversarial adaptation outperform no-adaptation on this dataset.** That broader finding
  stands, and is plausibly explained by a combination of factors verified while investigating
  (not just asserted): the label-prior mismatch above, a small-N training regime (263
  source-train / 160 target-adapt samples, ~800 total optimizer steps over 100 epochs — DANN's
  schedule was validated at far larger scale), and Pheno4D being genuinely multi-temporal
  (`scan_date` populated) while Crops3D is single-snapshot (`scan_date` empty in every row) — a
  structurally different, arguably harder domain gap than "same shapes, different sensor."
  Original (buggy) run kept in full at `results/C2_dgcnn_da_a/run_v1_unramped_entropy.log`
  (not deleted) — the diagnosis process is itself a documented, correct piece of work, only the
  numbers it explains are superseded. Full derivation, quartile breakdown, and the fix's exact
  before/after numbers in `step_notes/C2_DGCNN_DA_A.md`.

**Done (continued):**
- **Row C3 (DGCNN, DA-S, ALL) trained on the cluster** (2026-09-11, job 308459, ~1h04m; the
  first submission, job 308449, hit a CUDA OOM in `DefRec_and_PCM`'s (unmodified upstream)
  Chamfer-distance code — it builds dense `(B,N,N,3)` tensors, and at our `N=4096` (16x
  PointDA-10's 1024) a single call needs ~6 GiB per intermediate tensor; fixed by chunking
  `L_defrec`'s computation into `DEFREC_CHUNK=8`-sized slices with incremental backward, via a
  new `KendallUncertaintyWeighting.weighted_term()` helper — no changes to the sibling repo, see
  `step_notes/C3_DGCNN_DA_S.md` for the full root-cause derivation). Reuses
  `DefRec.deform_input`/`.calc_loss` and `DGCNN`'s already-inherited `self.DefRec` head unmodified
  — a genuine "adapt this repo" row, unlike C2's from-scratch adversarial machinery. DefRec runs
  on BOTH domains per this project's DA-S definition; `L_defrec` routed through Kendall as a
  third regression-type cooperative task; `--DefRec_weight 1.0` neutralizes the upstream repo's
  own fixed between-task weight since Kendall now owns that tradeoff.

  **Protocol-selected checkpoint** (epoch 95, lowest source val total loss):

  | Metric | C1 (DA-0) | C2 (DA-A) | C3 (DA-S) |
  |---|---|---|---|
  | Target cls acc | 0.7460 (avg 0.7446) | 0.9365 (avg 0.9130) | 0.5714 (avg 0.6625) |
  | Tomato seg mIoU | 0.3248 (acc 0.6645) | 0.2416 (acc 0.5210) | 0.3306 (acc 0.6838) |
  | Maize seg mIoU | 0.3427 (acc 0.7698) | 0.1996 (acc 0.5305) | 0.3774 (acc 0.7971) |

  **DA-S underperforms the DA-0 baseline on both the selected checkpoint and the full
  100-epoch trajectory** (full-run mean target acc: C1 0.791 vs. C3 0.703) — but, unlike C2,
  **C3's trajectory is stable, not unstable** (stdev 0.104, matching C1's 0.104, vs. C2's 0.214;
  C3 never crashes to C1/C2's ~0.37 floor). `corr(source val loss, target acc)` is *positive*
  for C3 (+0.22, vs. C1's −0.29) — the selection signal is actively unhelpful, not just weak,
  plausibly diluted by folding a self-supervised reconstruction term (`L_defrec`) that doesn't
  obviously track target-domain transfer into the same total loss used for selection (hypothesis,
  not verified by ablation). Compared against published PointDA-10 numbers (DGCNN, same paper):
  their DefRec-target-only improves over source-only by +3.6 pts avg (62.2→65.8); here DefRec
  (both-domain, jointly trained with `L_seg`) *degrades* vs. source-only by 8.8–17.5 pts — not a
  numerically fair comparison (10-class vs. 2-class, cls-only vs. cls+seg, 1024 vs. 4096
  points, synthetic-vs-real gap vs. sensor-vs-sensor gap), but the flipped direction is notable.
  One concrete, paper-documented contributing factor: the same paper's own ablation found
  running DefRec on both domains (this project's mandated DA-S protocol) underperforms
  target-only DefRec by ~0.9 pts in their benchmark — real, but far smaller than the gap
  observed here, so not the whole story. **Conclusion: DA-S doesn't help transfer here, but
  fails safely (stable training) rather than DA-A's failure mode (unstable, destabilizing).**
  Full trajectory/quartile/correlation breakdown and PointDA-10 comparison in
  `step_notes/C3_DGCNN_DA_S.md`.

**Done (continued):**
- **Row C4 (DGCNN, DA-A, L-N) trained on the cluster** (2026-09-12, job 308634, ~13 min).
  Byte-for-byte reuses C2's fixed `adapters/dann.py` (ramped `λ_ent`); only the augmentation
  pipeline differs (`adapters/dataset.py` gained an `augment_mode` parameter, "all"/"ln_only",
  additive — existing callers unaffected; `scripts/augmentations.py::compose_pipeline_ln_only`
  applies Gaussian jitter only, skipping G-R/G-S/L-D). Given the same full-trajectory analysis as
  C2/C3 per explicit user instruction, comparing against corrected C2 (v2, job 308608) — not the
  original buggy run.

  **Full-trajectory comparison** (all 100 epochs, target held-out cls accuracy):

  | Metric | C1 (DA-0) | C2 v2 (DA-A, ALL, fixed) | C4 (DA-A, L-N only) |
  |---|---|---|---|
  | Full-run mean target acc | 0.791 | 0.633 | 0.613 |
  | Full-run stdev | 0.104 | 0.117 | 0.138 |
  | corr(source val loss, target acc) | −0.286 | −0.061 | +0.041 |
  | Quartile means (Q1→Q4) | 0.798/0.834/0.773/0.760 | 0.691/0.678/0.608/0.554 | 0.646/0.664/0.582/0.559 |

  **Conclusion: narrowing augmentation to noise-only makes no meaningful difference to
  classification transfer relative to ALL** — both DA-A variants underperform DA-0 by a similar
  large margin (~16-18 pts on full-run mean) and show the identical qualitative failure shape
  (early rise, declining quartiles, near-zero correlation between the source-loss selection
  signal and target accuracy). This is evidence C2's post-fix underperformance vs. DA-0 reflects
  the adversarial method interacting with this dataset (label-prior mismatch, small-N, multi-
  temporal target — see C2 above), not an artifact of the ALL augmentation mix. **One clear
  augmentation effect found: C4's source-val seg mIoU is substantially higher than C1/C2**
  (Tomato 0.4514 vs. 0.3248/0.2384, Maize 0.4378 vs. 0.3427/0.3037) — most plausibly because
  ALL's L-D component (RandomCrop3D/CoarseDropout3D) removes points and makes segmentation
  strictly harder, while L-N (jitter) never removes points; flagged as a plausible mechanism, not
  a verified ablation. Full trajectory/quartile tables and reasoning in
  `step_notes/C4_DGCNN_DA_A_LN.md`.

**Done (continued):**
- **Row C5 (DGCNN, DA-O, Oracle) trained on the cluster** (2026-09-12, job 308716, ~24 min).
  **Block C is now complete (C1-C5).** Trains directly on a small labeled Pheno4D subset
  (plant-level train/val split: 72/18 scans across 10 plants, disjoint from the 4 held-out test
  plants) instead of adapting from Crops3D, then evaluates on the SAME `pheno4d_heldout_eval.csv`
  file/metric C1-C4 all report against, for a directly comparable number. Required a new,
  explicitly-flagged-as-inferred label scheme for Pheno4D's own per-point annotations (raw ids
  grow over the growing season, consistent with per-leaf instance ids folded into the label
  column rather than a flat 3-class scheme; collapsed to `{0: soil, 1: stem, 2: leaf}` — see
  `adapters/dataset.py::PHENO4D_SEG_NUM_CLASSES`/`pheno4d_collapse_organ_labels` docstrings for
  the full histogram-based reasoning). `PlantClsSegDataset` gained optional
  `label_transform`/`num_classes` args (additive, C1-C4 unaffected).

  **Full-trajectory comparison across all of Block C** (target held-out cls acc):

  | Metric | C1 (DA-0) | C2 v2 (DA-A ALL) | C3 (DA-S) | C4 (DA-A L-N) | C5 (DA-O, Oracle) |
  |---|---|---|---|---|---|
  | Full-run mean | 0.791 | 0.633 | 0.703 | 0.613 | **0.925** |
  | corr(selection-loss, target acc) | −0.286 | −0.061 | +0.222 | +0.041 | **−0.823** |
  | Quartile means (Q1→Q4) | rising→falling | falling | falling | falling | **rising** (.827→.999) |
  | Selected-checkpoint acc | 0.7460 | 0.6667 | 0.5714 | 0.5556 | **1.0000** |

  **Conclusion: the Oracle ceiling (0.925 full-run mean / 1.0000 selected) sits well above the
  DA-0 baseline (0.791 / 0.7460) — a real ~13-25 point gap, so C1 was NOT already near the
  ceiling.** This means C2/C3/C4's failure to beat C1 reflects genuine unclaimed headroom, not an
  already-saturated task — all three landed below C1, not just short of C5. C5 is also
  qualitatively different in kind: its quartile means rise monotonically over training (the
  opposite of every DA-0/DA-A/DA-S row) and its selection-loss-vs-target-acc correlation is
  strongly negative (informative signal), confirming DGCNN itself is fully capable of
  near-perfect target-domain classification (and reasonably good target segmentation, Tomato
  mIoU 0.83/Maize mIoU 0.52 on Pheno4D's own real labels — an Oracle-only metric, not comparable
  to C1-C4's source-val seg mIoU, different label space) given real target supervision — the
  sensor/domain gap, not model capacity or task difficulty, is what's defeating C1-C4. Full
  trajectory tables and caveats (small-N on both the Oracle's 90-scan labeled pool and the
  63-scan target test set) in `step_notes/C5_DGCNN_DA_O.md`.

**Done (continued):**
- **Row A2 (PointNet++, DA-A, ALL) trained on the cluster** (2026-09-12, job 308754, ~13.5 min).
  Ports C2's adversarial DANN machinery (`adapters/dann.py`, reused byte-for-byte — already
  backbone-agnostic by design) to PointNet++, to directly test whether C2/C3/C4's
  underperformance-vs-DA-0 pattern is DGCNN-specific or a general property of this dataset's
  source/target label-prior mismatch and small-N regime. `PointNet2_ClsSeg.forward` gained
  `logits["feat"]` (pooled global feature, domain discriminator input — additive, A1 unaffected).

  **Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

  | Metric | A1 (DA-0) | A2 (DA-A) | C1 (DA-0, for reference) | C2 v2 (DA-A) |
  |---|---|---|---|---|
  | Full-run mean | 0.599 | 0.554 | 0.791 | 0.633 |
  | Full-run stdev | 0.196 | 0.193 | 0.104 | 0.117 |
  | corr(selection-loss, target acc) | −0.165 | +0.086 | −0.286 | −0.061 |
  | Epochs collapsed to one class | 17/100 | 15/100 | 3/100 | 3/100 |
  | Selected-checkpoint acc | 0.4603 | 0.4444 | 0.7460 | 0.6667 |

  **Conclusion: the underperformance pattern is real on both backbones but much smaller/less
  decisive on PointNet++.** A1→A2's full-run mean drops only 4.5 points (vs. C1→C2's 15.8) and
  the selected-checkpoint gap is under 2 points (essentially a tie, vs. C1→C2's 7.9-point gap).
  The direction still replicates (DA-A doesn't beat DA-0; the selection-loss-vs-target-acc
  correlation gets less informative under DA-A on both backbones), supporting the dataset-level
  explanation (label-prior mismatch, small-N — see C2 above) as a real, shared contributor. But
  **A1's own DA-0 baseline is already highly unstable independent of any DA method** (stdev
  0.196, 17/100 collapse epochs — vs. C1's 0.104/3-of-100), matching/sharpening the earlier
  systematic-Maize-bias finding logged for A1 above — this pre-existing noise is why DGCNN's
  C1→C2 comparison isolates the adversarial-training effect more cleanly than PointNet++'s does,
  not because the effect is absent on PointNet++. One consistent cross-backbone side finding:
  segmentation mIoU is worse under DA-A than DA-0 on both backbones (plausibly shared backbone
  capacity competing between `L_seg` and the adversarial/entropy terms). Full trajectory tables
  and reasoning in `step_notes/A2_PointNet2_DA_A.md`.

**Done (continued):**
- **Row A3 (PointNet++, DA-S, ALL) trained on the cluster** (2026-09-12, job 308789, ~33 min).
  Ports C3's DefRec self-supervised reconstruction to PointNet++, to test whether "DA-S was
  DGCNN's least-bad adaptation result" (C1 0.791 → C3 0.703, an 8.8-point full-run-mean decline)
  is a backbone-general property. Required adding a DefRec reconstruction head to
  `PointNet2_ClsSeg` (`PointDA.Models.RegionReconstruction` — the same generic per-point Conv1d
  stack `DGCNN_ClsSeg` already uses via inheritance — composed unmodified, fed the
  already-computed `seg_feat` tensor). DefRec's core machinery (`deform_input`/`calc_loss`,
  chunked-Chamfer-distance OOM workaround) is fully backbone-agnostic and reused unchanged.

  **Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

  | Metric | A1 (DA-0) | A3 (DA-S) | C1 (DA-0, ref) | C3 (DA-S, ref) |
  |---|---|---|---|---|
  | Full-run mean | 0.599 | **0.800** | 0.791 | 0.703 |
  | Full-run stdev | 0.196 | 0.158 | 0.104 | 0.104 |
  | corr(selection-loss, target acc) | −0.165 | −0.009 | −0.286 | +0.222 |
  | Collapse epochs | 17/100 | **2/100** | 3/100 | not re-checked |
  | Selected-checkpoint acc | 0.4603 | **0.8889** | 0.7460 | 0.5714 |

  **This is the first row where an adaptation method clearly beats its own backbone's DA-0
  baseline — it does not just fail to replicate C1→C3's decline, it reverses direction
  entirely** (A1→A3 gains ~20 points full-run mean / ~43 points selected checkpoint, vs.
  C1→C3's ~9/17.5-point loss). Checked for target-label leakage/selection-signal contamination
  before trusting this (none found — same vetted code path as C3) and confirmed the gain holds
  across nearly the whole trajectory (all quartiles but Q2 improve), not one lucky checkpoint.
  **But it comes with a real, sustained cost**: per-epoch Tomato seg mIoU (source val) collapses
  from ~0.27 to a ~0.02-0.09 floor within the first few epochs and stays there (FINAL 0.0594 vs.
  A1's 0.3207), while Maize seg mIoU steadily improves to ~0.34-0.39 (FINAL 0.3404, close to
  A1's 0.4741) — plausibly because Tomato is already the minority/harder-imbalanced species and
  the added `cls`+`defrec` training pressure (Kendall's learned weights end up favoring `cls`
  heavily by epoch 99) crowds out its already-fragile segmentation, an inference not verified by
  ablation. **Conclusion: unlike DA-A (same underperformance direction on both backbones, just
  different magnitude — A2 above), DA-S's effect is genuinely architecture-dependent in
  direction, not just degree — it actively helps PointNet++ classification transfer while
  actively hurting DGCNN's, and its PointNet++ benefit is not free (a real segmentation
  tradeoff).** Full trajectory tables and the segmentation-collapse detail in
  `step_notes/A3_PointNet2_DA_S.md`.
- **KPConv backbone integrated, Block B started** (2026-09-12) — third and last backbone
  architecture family. Vendored `HuguesTHOMAS/KPConv-PyTorch` as a sibling repo (pinned at
  commit `d19c575d3fa9fcfd5a74845b5b27aac7e50472c7`), same convention as DefRec_and_PCM/
  Pointnet_Pointnet2_pytorch. Required patching two compiled C++ extensions
  (`cpp_wrappers/cpp_subsampling`, `cpp_wrappers/cpp_neighbors` — grid subsampling + radius
  neighbor search) for numpy>=2/modern-setuptools incompatibilities in the vendored repo's own
  ~2020-era build files (`numpy.distutils` removal, `PyArray_DATA`/`_NDIM`/`_DIM` signature
  tightening) — minimal compatibility patches to the vendored repo, same category as the
  `np.int` fix already documented for DefRec_and_PCM, not project-specific logic. Confirmed
  along the way that these extensions are CPU-only (not CUDA, correcting the Backbones section
  above) — they compiled and ran on the cluster login node with no GPU present.
  `adapters/kpconv_collate.py` builds KPConv's "stacked batch" format (all samples'
  points/neighbors/pools concatenated per layer, not a fixed `(B,3,N)` tensor) by reusing the
  reference repo's own `PointCloudDataset.segmentation_inputs` unmodified.
  `adapters/models_kpconv.py::KPConv_ClsSeg` mirrors `KPFCNN`'s encoder/decoder block
  construction line-for-line (3 layers, 2 stridings, no deformable blocks — matching
  `PointNet2_ClsSeg`'s depth for a comparable cross-backbone architecture) plus a classification
  head off the pooled bottleneck feature, exposing the same `{"cls", "feat", "seg_feat"}`
  interface as the other two backbones' models. `adapters/train_b1_kpconv_da0.py` mirrors
  `train_c1_dgcnn_da0.py`'s DA-0 training loop. CPU smoke test (1 epoch, real data) passed with
  sane, non-degenerate numbers before submitting. Full integration detail, the exact compiler
  errors/fixes, and design decisions in `step_notes/B1_KPConv_DA0.md`.
- **Row B1 (KPConv, DA-0, ALL) hit a real CUDA OOM bug, root-caused and fixed, resubmitted**
  (2026-09-12). First submission (job 308845) appeared to hang for over an hour with zero
  per-batch visibility; a follow-up debug run with added per-batch timing crashed with a CUDA
  OOM (`Tried to allocate 31.86 GiB`) inside KPConv's neighbor-distance computation on only the
  3rd training batch. Root cause: `neighborhood_limits` (caps each layer's neighbor-matrix
  width) was left uncalibrated/empty, a design choice originally reasoned as safe for this
  project's small N=4096 clouds — wrong in practice, since a small fraction of augmented batches
  produce pathologically dense local point clusters (back-calculated to ~2900 neighbors/point for
  the batch that OOM'd). Fixed by implementing real calibration
  (`adapters/kpconv_collate.py::calibrate_neighborhood_limits`, run once at training start,
  mirroring the reference repo's own `datasets/*.py::calibration` routines) — caught and fixed a
  second bug while wiring it in (`neighborhood_limits` lives on the dataset/batch-builder
  instance, not `config`, confirmed by reading the reference repo's source rather than assuming).
  Tried parallelizing KPConv's CPU-bound, single-threaded collate via `DataLoader(num_workers=4)`
  as a speed lever — made things worse (no output at all vs. `num_workers=0`'s clean run),
  consistent with this row's original, now-vindicated caution flagging multi-worker safety as
  unverified; reverted, kept at the safe default. **Real, accepted cost, revised upward twice**:
  first estimated 10-15h (CPU-only extrapolation) and raised the time budget from 4h to 24h
  (confirmed this isn't a cluster-imposed limit first — `dgx` partition/QOS has no MaxWall,
  `scontrol show partition dgx` → `MaxTime=6-00:00:00`); job 308845's resubmission (308956) then
  showed real GPU/node contention on the shared A100 (4 epochs in ~1h55m, per-epoch time ranging
  14-51 min, per-batch fwd+bwd swinging 0.2s-86s) putting it on pace for ~48h — job killed at
  4/100 epochs (before more compute was sunk into a run that would've been killed by SLURM
  mid-trajectory anyway) and resubmitted a second time at `--time=60:00:00`, job **309015**.
  `--verbose_batches` left on for observability given the original silent-stall failure mode.
  **This contention risk applies to every Block B row, not just B1** — see the Backbones section
  above, now flagged so B2-B5 start with the 60h budget rather than rediscovering this. Full
  incident writeup in `step_notes/B1_KPConv_DA0.md`.
- **Row B1 (KPConv, DA-0, ALL) finished — Block B's own baseline established** (2026-09-13, job
  309015, 2h15m — landed on `cn17-dgx` with none of 308956's contention, hard evidence the
  earlier slowdown was transient/node-specific, not structural; the 60h budget stays the standing
  Block B convention regardless, since it's cheap insurance against a risk that already
  materialized once). Best epoch **99** (the literal last epoch — source val loss was still
  monotonically improving through the end of training).

  **Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

  | Metric | A1 (DA-0) | C1 (DA-0) | B1 (DA-0) |
  |---|---|---|---|
  | Full-run mean | 0.599 | 0.791 | 0.629 |
  | Full-run stdev | 0.196 | 0.104 | **0.203** |
  | corr(selection-loss, target acc) | −0.165 | −0.286 | **+0.316** |
  | Collapse epochs (all-one-class) | 17/100 | 3/100 | 3/100 |
  | Selected-checkpoint acc | 0.4603 | 0.7460 | 0.4444 |

  **Answering the motivating question (does KPConv's claimed density-robustness show up as a
  stable baseline like DGCNN?): mixed, does not cleanly confirm the hypothesis.** By
  collapse-epoch count B1 looks DGCNN-like (3/100, matching C1, far better than A1's 17/100). But
  by full-run stdev B1 is the *least* stable of the three backbones (0.203, even above A1's
  0.196), and its selection-signal correlation is *positive* (+0.316) — the only DA-0 row across
  three backbones where a better source-domain loss actively predicts *worse* target accuracy,
  the wrong direction for model selection to be useful (only C3's DGCNN/DA-S row showed this
  pattern before, never a DA-0 baseline). So KPConv's baseline has its own distinct instability
  signature — resistant to total class-collapse, but with the highest continuous variance and the
  most misleading selection signal of the three. This complicates, rather than confirms, the
  density-sensitivity hypothesis from A3's investigation. Two more findings: B1 shares A1's exact
  systematic Maize-bias failure mode (Tomato recall 0.125, Maize recall 1.00 — DGCNN doesn't show
  this), and B1's Q4 settles into the same "frozen at a low floor" DA-0 drift pattern as C1/A1,
  just at a lower floor (~0.46-0.48 vs. C1's 0.746). One genuinely positive, separate finding:
  **B1's source-val segmentation is the best of the three DA-0 baselines** — Tomato mIoU 0.3808
  (vs. A1 0.3207, C1 0.3248), Maize mIoU 0.4414 (close to A1's best-of-three 0.4741, clearly above
  C1's 0.3427). Full trajectory tables and reasoning in `step_notes/B1_KPConv_DA0.md`.
- **Row B2 (KPConv, DA-A, ALL) submitted to the cluster** (2026-09-13, job 309232 on `cn19-dgx`,
  `--time=60:00:00`). Ports the DANN anchor method (`adapters/dann.py`, reused byte-for-byte) to
  KPConv, per explicit user instruction, to test whether B1's already-worst-of-three
  selection-signal reliability (`corr = +0.316`, the only DA-0 row with a positive correlation)
  gets worse under adversarial training the way it did for DGCNN (C1→C2) and PointNet++ (A1→A2),
  or behaves differently. Two real KPConv-specific details, not copy-paste from A2/C2: (1)
  `DomainDiscriminator(in_dim=256)`, not 1024 — traced KPConv_ClsSeg's actual pooled-feature
  width directly rather than assuming DGCNN/PointNet2's value carries over; (2) `neighborhood_
  limits` now calibrated over BOTH domains (new `collate_fn_factory` param on `calibrate_
  neighborhood_limits` + new `combine_neighborhood_limits` helper in `adapters/kpconv_collate.py`,
  both additive/backward-compatible with B1's existing call), since DA-A pushes target batches
  through the same encoder every step and source-only calibration risks under-covering
  target-domain neighbor counts — exactly B1's original OOM failure mode. Verified this wasn't
  theoretical: the CPU smoke test's actual calibration found target-only differed from
  source-only at layer 2 (29 vs 28), a real measured instance, not hypothetical. CPU smoke test
  (1 epoch, real data) passed cleanly before submitting. Full design decisions in
  `step_notes/B2_KPConv_DA_A.md`.
- **Row B2 (KPConv, DA-A, ALL) finished — the first backbone where DANN does NOT show a clear
  DA-0→DA-A underperformance pattern** (2026-09-13, job 309232, 2h44m, no contention). Best
  epoch 65.

  **Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

  | Metric | B1 (DA-0) | B2 (DA-A) | A1→A2 (DA-0→DA-A) | C1→C2 v2 (DA-0→DA-A) |
  |---|---|---|---|---|
  | Full-run mean | 0.629 | **0.632** | 0.599→0.554 (−0.045) | 0.791→0.633 (**−0.158**) |
  | Full-run stdev | 0.203 | **0.141** | 0.196→0.193 (flat) | 0.104→0.117 (+0.013) |
  | corr(selection-loss, target acc) | +0.316 | +0.238 | −0.165→+0.086 (worse) | −0.286→−0.061 (worse) |
  | Collapse epochs | 3/100 | 2/100 | 17/100→15/100 | 3/100→5/100 |
  | Selected-checkpoint acc | 0.4444 | **0.5714 (+0.127)** | 0.4603→0.4444 (−0.016) | 0.7460→0.6667 (−0.079) |

  **Answering the motivating question directly: B1's poor selection-signal reliability does NOT
  get worse under DANN — it improves marginally (+0.316→+0.238) but stays on the wrong
  (positive) side, the same general unreliability rather than a deepened one.** More broadly,
  KPConv responds to DANN on almost every axis differently from both other backbones: full-run
  mean is essentially unchanged (not the clear decline C1→C2 shows or the smaller decline A1→A2
  shows), the selected checkpoint actually *improved* (+12.7 points, the opposite direction from
  both other backbones), and full-run stdev *decreased* (DANN stabilized KPConv's baseline,
  unlike DGCNN where it rose slightly). The quartile shape also differs qualitatively — B2 dips
  in Q2 then partially recovers through Q3/Q4 (ending on a low-variance, moderately-good
  plateau), instead of the monotonic "erode to a low floor" pattern every other DA-0/DA-A row
  shows. B1's systematic Maize-bias also eased under DANN (Tomato recall 0.125→0.375), not
  worsened. **The one place DA-A costs KPConv something real, matching both other backbones: source-val
  segmentation mIoU drops** (Tomato 0.3808→0.2730, Maize 0.4414→0.3889) — plausibly the same
  shared-capacity competition between `L_seg` and the adversarial/entropy terms documented for
  A2/C2. **This is now the second row (after A3's DA-S reversal) where an adaptation method's
  effect is genuinely architecture-dependent in a way that isn't just "same direction, different
  magnitude"** — DA-A's underperformance, previously read (after A2) as a shared, dataset-driven
  property appearing on both backbones to different degrees, does not extend to KPConv at all.
  Full trajectory tables and reasoning in `step_notes/B2_KPConv_DA_A.md`.
- **Row B3b (KPConv, DA-S, ALL) submitted to the cluster** (2026-09-13, job 309416 on `cn19-dgx`,
  `--time=60:00:00`). See the strategy-table Backbones-section entry above for why this row is
  labeled B3b (an addition to the documented plan, not the originally-planned B3=DA-D) and its
  purpose: testing whether KPConv also flips DA-S's effect the way it flipped DA-A's (B2), given
  A3 already showed DA-S flips between PointNet++ (helps) and DGCNN (hurts). Required adding a
  DefRec reconstruction head to `KPConv_ClsSeg` (`PointDA.Models.RegionReconstruction`, same
  class DGCNN/PointNet2 already use, composed unmodified) plus two new helpers in
  `adapters/kpconv_collate.py` (`batch_points_bcn`, `build_batch_from_points`) to solve a real
  KPConv-specific problem: unlike DGCNN/PointNet2 (which recompute their geometric structure
  automatically inside `forward` from whatever coordinates they're given), KPConv precomputes
  its multi-layer neighbor structure OUTSIDE the model via collate — so DefRec's deformed points
  need a genuine batch rebuild, not just a value swap, to be faithfully processed through the
  encoder. Caught a real bug before it could crash a run: `DefRec.calc_loss` expects the *full*
  logits dict (reads `logits['DefRec']` internally), not a pre-extracted tensor — found by
  reading the reference repo's source directly and comparing against A3's exact call pattern,
  fixed before resubmitting the smoke test. CPU smoke test then passed cleanly, but measured a
  real, substantial cost: per-batch times (collate 12-120s, fwd+bwd 53-236s at batch_size=4) are
  far higher than B1/B2's, from needing ~5 collate calls per training step (1 clean + 4 deformed-
  chunk rebuilds) instead of 1-2 — expected to use much more of the 60h budget than B1/B2 did,
  though still comfortably inside it barring contention. Full design decisions and the bug
  writeup in `step_notes/B3b_KPConv_DA_S.md`.
- **Row B3b (KPConv, DA-S, ALL) finished — the strongest non-Oracle result in the whole
  project** (2026-09-13, job 309416, 4h58m, no contention). Best epoch 84.

  **Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

  | Metric | B1 (DA-0) | B3b (DA-S) | A1→A3 (DA-0→DA-S) | C1→C3 (DA-0→DA-S) |
  |---|---|---|---|---|
  | Full-run mean | 0.629 | **0.860** | 0.599→0.800 (+0.201) | 0.791→0.703 (−0.088) |
  | Full-run stdev | 0.203 | **0.086** | 0.196→0.158 | 0.104→0.104 |
  | corr(selection-loss, target acc) | +0.316 | +0.291 | −0.165→−0.009 | −0.286→+0.222 |
  | Collapse epochs | 3/100 | **0/100** | 17/100→2/100 | 3/100→n/a |
  | Selected-checkpoint acc | 0.4444 | 0.7143 (+0.270) | 0.4603→0.8889 (+0.429) | 0.7460→0.5714 (−0.175) |

  **Answering the motivating three-way question (flip DA-A's pattern like B2, align with C3's
  null result, or something distinct?): distinct from both, and stronger than either.** It
  clearly does NOT align with C3 (DGCNN: DA-S hurts by 8.8 points; KPConv: DA-S helps by 23.1
  points — about as far apart as two results on the same method can be). It is NOT simply "A3's
  reversal ported over" either — B1→B3b's full-run-mean gain (+23.1 points) actually *exceeds*
  A1→A3's (+20.1 points), and B3b's stdev (0.086) and collapse count (0/100) both beat every
  other row trained in this project so far, DA-0/DA-A/DA-S/DA-O included — the single most
  stable trajectory seen yet. (A3 still wins on selected-checkpoint magnitude: +0.429 vs. B3b's
  +0.270 — A3's best single snapshot lands higher even though B3b's overall trajectory is
  stronger and steadier.) It does not mirror B2 (DA-A) either — B2 was roughly a tie with B1
  (+0.003 full-run mean), while B3b is a decisive win — so within Block B, DA-A and DA-S land in
  genuinely different places on the same backbone. **B3b's full-run mean (0.860) sits only 0.065
  below C5's Oracle ceiling (0.925)** — remarkable for a method that never touches a target
  label. Unlike A3, this win is NOT bought with a segmentation collapse: reading the full
  per-epoch trajectory, both Tomato and Maize seg mIoU *rise* steadily over training (no
  collapse-and-floor pattern at all), ending only modestly below B1's baseline (Tomato
  0.2697 vs. 0.3808, Maize 0.3910 vs. 0.4414) — the cleanest win of any adaptation-method result
  in the project. One thing that does NOT change from B1: the selection-signal correlation stays
  positive and essentially flat (+0.316→+0.291, still wrong-signed) — this specific KPConv
  quirk appears structural to the backbone+dataset pairing, independent of which adaptation
  method sits on top of it. Full trajectory tables and reasoning in
  `step_notes/B3b_KPConv_DA_S.md`.
- **Row B3 (KPConv, DA-D, ALL) submitted to the cluster** (2026-09-14, job 310284 on
  `cn11-dgx`, `--time=60:00:00`). The row CLAUDE.md's strategy table actually specifies as B3
  (discrepancy-based, not self-supervised) — see the Backbones/strategy-table entries above for
  the B3-vs-B3b naming distinction this project already caught and corrected once. Per explicit
  user instruction: tests whether B3b's dramatic KPConv+DA-S win (+23.1 points full-run mean,
  the project's best non-Oracle result) is specific to the self-supervised reconstruction
  mechanism, or whether KPConv responds well to domain-adaptation pressure generally, using a
  mechanistically distinct method (Deep CORAL feature-covariance alignment — no discriminator,
  no reconstruction pretext) as the test. No reference implementation existed to adapt
  (DefRec_and_PCM implements DefRec+PCM only, confirmed by grep, same as `dann.py`'s situation)
  — new `adapters/coral.py` implements standard Deep CORAL from scratch, chosen over MMD for
  having no kernel-bandwidth hyperparameter to select (an explicit, flagged choice, not a spec
  mandate). Per CLAUDE.md's loss architecture, `L_CORAL` is Kendall-weighted (a cooperative DA
  loss, unlike DA-A's fixed/scheduled `L_dom`/`L_ent`) — a real, substantive difference from B2's
  loss combination, not just a different loss function. Structurally much simpler than both B2
  (no discriminator/GRL/entropy schedule) and B3b (no deformed-batch rebuilds at all) — expected
  per-epoch cost close to B1/B2's real runtimes (a few hours), not B3b's ~5h. CPU smoke test was
  interrupted by a session/environment issue (not a code bug) after 60/65 batches had already
  run cleanly with sane, bounded per-batch costs — judged sufficient to proceed without a third
  attempt; see `step_notes/B3_KPConv_DA_D.md` for the full reasoning.
- **Row B3 (KPConv, DA-D, ALL) finished — KPConv's SECOND clear adaptation win, landing next to
  B3b (DA-S) not B2 (DA-A)** (2026-09-14/15, job 310284, 10h12m — a confirmed real GPU-contention
  episode mid-run, ~15x slowdown for several hours around epochs 69-79, checked directly via
  per-epoch timestamps, not guessed; absorbed fine by the 60h budget, second independent
  confirmation of this cluster-level risk after B1's original incident). Best epoch 94.

  **Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

  | Metric | B1 (DA-0) | B2 (DA-A) | B3 (DA-D) | B3b (DA-S) |
  |---|---|---|---|---|
  | Full-run mean | 0.629 | 0.632 (+0.003) | **0.808 (+0.179)** | 0.860 (+0.231) |
  | Full-run stdev | 0.203 | 0.141 | **0.094** | 0.086 |
  | corr(selection-loss, target acc) | +0.316 | +0.238 | +0.268 | +0.291 |
  | Collapse epochs | 3/100 | 2/100 | **0/100** | 0/100 |
  | Selected-checkpoint acc | 0.4444 | 0.5714 (+0.127) | **0.6825 (+0.238)** | 0.7143 (+0.270) |

  **Answering the motivating question directly: Deep CORAL lands clearly and unambiguously on
  the "large win" side, much closer to B3b than to B2.** B3's full-run-mean gain (+17.9 points)
  is nearly 60x B2's (+0.3 points) and reaches 77% of B3b's own gain — the second-most-stable
  trajectory of any row in the entire project (stdev 0.094, beaten only by B3b's 0.086, below
  even C1's previous best of 0.104), with zero collapse epochs matching B3b exactly. **This
  clarifies the broader question in a specific, informative way**: DA-D (CORAL) and DA-S
  (DefRec) sit on OPPOSITE ends of the "exposes the model to real target-domain geometry"
  spectrum — CORAL only ever touches abstract pooled feature-covariance statistics, never
  reconstructs or aligns actual target point coordinates, while DefRec's whole mechanism IS
  reconstructing deformed target geometry — yet both produce large wins, while DA-A (also
  mechanistically distinct from both) is the one outlier. If "needs real target-domain geometry
  exposure" were the operative explanation, CORAL should have looked more like DA-A's tepid
  result; it didn't. **The best-supported read: what predicts success for KPConv here is
  specifically whether the DA loss is cooperative/directly-minimized (DA-D and DA-S are both
  Kendall-weighted per CLAUDE.md's loss architecture) rather than adversarial (DA-A's `L_dom`/
  `L_ent` are fixed/scheduled precisely because of their min-max structure) — not "any
  adaptation helps" and not "needs geometry exposure specifically."** Flagged as the
  best-supported reading, not a proven causal mechanism — a competing, equally-consistent
  explanation (KPConv is just generally adaptation-receptive, with DA-A being its weakest form
  of help rather than categorically different) isn't ruled out by this row alone; see
  `step_notes/B3_KPConv_DA_D.md` for the full reasoning on why neither B4 nor B5 will resolve
  this either. One place B3 differs from B3b: **Tomato seg mIoU (0.2482) is the lowest of any
  DA method** (vs. B2's 0.2730, B3b's 0.2697), while **Maize mIoU (0.4150) is the highest of the
  three** (vs. B2's 0.3889, B3b's 0.3910) — no collapse in either case, both rise steadily over
  training. Full trajectory tables and reasoning in `step_notes/B3_KPConv_DA_D.md`.
- **Row B4 (KPConv, DA-0, L-D only, ALL replaced with L-D) submitted to the cluster**
  (2026-09-15, job 311004 on `cn17-dgx`, `--time=60:00:00`). Per explicit user instruction:
  Block B's own deliberately-unadapted density-robustness check (per CLAUDE.md's strategy table
  — Block A/C use A3/C3 as their third row, Block B never did; B4 is Block B's differently-shaped
  substitute), pairing L-D (RandomCrop3D + CoarseDropout3D, the two augmentations that directly
  perturb local point density/coverage) with NO adaptation method at all, unlike A4/C4 (which
  pair L-D/L-N with an adversarial method). Given B1's already-mixed stability story and B2/B3/
  B3b's finding that KPConv responds well to cooperative but not adversarial adaptation losses,
  B4 tests a cleanly different question: does the raw architecture, with no adaptation signal at
  all, hold up against a targeted density/coverage perturbation on its own. Identical to B1 in
  every respect except one line (`augment_mode="ld_only"` on `src_train_set`) — mirrors how C4
  differed from C2 by one augmentation swap. Required adding
  `compose_pipeline_ld_only`/`_with_labels` to `scripts/augmentations.py` (same pattern as C4's
  `compose_pipeline_ln_only`) and registering `"ld_only"` in `adapters/dataset.py`'s
  `_AUGMENT_PIPELINES` — both purely additive. No dataset padding-logic changes needed (checked
  directly: `pad_if_needed3d` already runs unconditionally regardless of `augment_mode`). CPU
  smoke test passed cleanly (26 batches, fast/consistent per-batch costs matching B1's own
  original smoke-test profile) — launched detached (`nohup`/`disown`) from the start this time
  given B3's smoke test was lost to a session-teardown issue, though not needed here (no
  interruption). Full design decisions in `step_notes/B4_KPConv_DA0_LD.md`.
- **Row B4 (KPConv, DA-0, L-D only) finished — a mixed result that SHARPENS rather than
  resolves KPConv's selection-signal problem** (2026-09-15, job 311004, 2h10m, no contention).
  Best epoch 73.

  **Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

  | Metric | B1 (DA-0, ALL) | B4 (DA-0, L-D only) |
  |---|---|---|
  | Full-run mean | 0.629 | 0.649 (+0.020) |
  | Full-run stdev | 0.203 | 0.168 |
  | corr(selection-loss, target acc) | +0.316 | **+0.470 (worse)** |
  | Collapse epochs | 3/100 | 1/100 |
  | Selected-checkpoint acc | 0.4444 | 0.5556 (+0.111) |

  **Answering the motivating question (does isolating L-D with zero adaptation cleanly confirm
  KPConv's claimed density-robustness?): no.** Raw trajectory numbers look like a mild
  improvement (mean +2 points, stdev down, collapse epochs down, selected checkpoint +11.1
  points), but `corr(selection-loss, target acc)` gets meaningfully WORSE, not better — +0.316→
  +0.470, **the single most positive (most actively misleading) correlation of any row trained
  in this project so far**, surpassing B1's own previous project-worst. The quartile trend still
  shows the same "early peak, then erode" DA-0 shape every prior DA-0/DA-A row has shown — L-D
  alone doesn't change the qualitative drift pattern, only its magnitude. This adds to, rather
  than resolves, B1's own finding that KPConv's baseline has a distinct instability signature
  that complicates the density-sensitivity hypothesis — even the cleanest possible test (density-
  focused augmentation, zero adaptation-method interference) makes the backbone's
  selection-signal reliability problem worse, not better. **One real, KPConv-specific
  augmentation effect found**: segmentation mIoU improves on BOTH organ classes (Tomato 0.3808→
  0.4286, Maize 0.4414→0.4624) despite B4 still including the same point-removing L-D
  augmentation B1's ALL mix has — since point-removal can't explain this (unlike C4's DGCNN-
  specific mechanism), the more likely driver is that the REMOVED augmentations (G-R/G-S/L-N)
  were themselves making KPConv's segmentation harder, flagged as an inference, not verified by
  ablation. The systematic Maize-bias eases somewhat (Tomato recall 0.125→0.300) but doesn't
  disappear. Full trajectory tables and reasoning in `step_notes/B4_KPConv_DA0_LD.md`.
- **Row B5 (KPConv, DA-O, Oracle, ALL) submitted to the cluster** (2026-09-15, job 311262 on
  `cn12-dgx`, `--time=60:00:00`). The KPConv-track counterpart of `train_c5_dgcnn_da_o.py`,
  completing Block B (B1-B5 all done after this row). Per explicit user instruction: directly
  tests whether KPConv's wrong-signed selection-signal correlation — positive in EVERY
  configuration tried so far (B1 +0.316, B2 +0.238, B3 +0.268, B3b +0.291, B4 +0.470) — is
  specifically a domain-adaptation artifact (should flip/resolve once trained and selected
  directly on labeled target data, no domain gap to cross) or a more fundamental
  backbone+dataset property that persists even under Oracle conditions. C5 already answered this
  for DGCNN: `corr` flips to strongly negative (−0.823) under Oracle training, the most
  informative selection signal of any Block C row — B5 checks whether KPConv matches that
  resolution or becomes the first row where even direct target supervision doesn't fix it.
  Mirrors C5's Oracle protocol exactly (domain roles swapped: trains on Pheno4D's own annotated
  subset, `data/pheno4d_oracle_train.csv`/`pheno4d_oracle_val.csv` — C5's exact plant-disjoint
  split, reused unchanged, not re-derived — evaluated on the SAME held-out target test file
  B1-B4 all report against) with KPConv's batch/collate mechanics substituted for DGCNN's
  raw-tensor forward pass. `batch_size=8` (same data-driven necessity as C5 — 72 training
  samples). `neighborhood_limits` calibrated on the Oracle train set only, not both domains —
  correct here since this row never touches Crops3D at all, unlike B2/B3/B3b. CPU smoke test
  passed cleanly (full single-epoch pass, 18/18 batches, no errors; dataset sizes matched C5's
  exactly). Full design decisions in `step_notes/B5_KPConv_DA_O.md`.
- **Row B5 (KPConv, DA-O, Oracle) finished — Block B is now complete, and the wrong-signed
  selection-correlation question is resolved** (2026-09-15/16, job 311262, 45 min, no
  contention). Best epoch 97.

  **Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

  | Metric | B1 (DA-0) | B2 (DA-A) | B3 (DA-D) | B3b (DA-S) | B4 (DA-0/L-D) | B5 (DA-O) | C5 (DGCNN DA-O) |
  |---|---|---|---|---|---|---|---|
  | Full-run mean | 0.629 | 0.632 | 0.808 | 0.860 | 0.649 | **0.9425** | 0.925 |
  | Full-run stdev | 0.203 | 0.141 | 0.094 | 0.086 | 0.168 | **0.0815** | n/a |
  | corr(selection-loss, target acc) | +0.316 | +0.238 | +0.268 | +0.291 | +0.470 | **−0.858** | −0.823 |
  | Selected-checkpoint acc | 0.4444 | 0.5714 | 0.6825 | 0.7143 | 0.5556 | **0.9841** | 1.0000 |

  **Answering the motivating question directly: the wrong-signed selection correlation does NOT
  persist under Oracle conditions — it flips to strongly negative (−0.858), resolving exactly
  the way DGCNN's did in C5 (−0.823), and slightly exceeds it in magnitude.** Every KPConv
  configuration with a domain gap to cross (B1/B2/B3/B3b/B4) showed a positive, wrong-signed
  correlation (a better-looking validation loss never reliably predicted better target accuracy);
  this is confirmed to be a domain-adaptation-specific artifact, not a fundamental property of the
  KPConv backbone or this dataset — it disappears entirely once the model is trained and selected
  directly on labeled target data. **B5's full-run mean (0.9425) also edges out C5's own Oracle
  ceiling (0.925)**, and its target confusion matrix shows the systematic Maize-bias seen in
  every other Block B row (B1 Tomato recall 0.125) essentially gone (Tomato recall 0.975) — real
  target supervision resolves both the selection-signal problem and the class-bias problem at
  once. Full derivation, quartile tables, and caveats (small-N Oracle pool, same as C5) in
  `step_notes/B5_KPConv_DA_O.md`.

**Next (in order):**
1. **Block B is complete (B1-B5, all five rows done).** Next: remaining Block A rows (A4 DA-A/
   L-N, A5 Oracle) — the `--time=60:00:00` convention is Block B (KPConv)-specific and does not
   apply to Block A (PointNet++'s own runtimes are the ~4h every non-KPConv row so far has used).
   Self-supervised deformation reconstruction is 2-for-3 across backbones (helps PointNet++,
   helps KPConv even more, hurts DGCNN) — a real, cross-backbone pattern worth keeping in mind
   for Block D, now joined by CORAL's similarly strong KPConv-specific result.

## Style notes
- Documents/reports: black and white only, no color.
- Code: individual files preferred over one large monolith; complete replacements over
  partial patches when editing.
- Flag inferences vs. confirmed facts explicitly — don't silently assume.
