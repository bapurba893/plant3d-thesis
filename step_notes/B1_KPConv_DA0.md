# Row B1: KPConv, DA-0 (Source-Only), All Augmentations

## What this is, in plain language

B1 is Block B's first row: KPConv (kernel point convolution), the third and last backbone
architecture family in this project's plan, trained source-only (no domain adaptation) on
Crops3D, evaluated on Pheno4D — the same DA-0 lower-bound protocol C1 (DGCNN) and A1 (PointNet++)
already established. Establishes Block B's own baseline before any adaptation method (B2 DA-A, B3
DA-D, B4 DA-0/L-D, B5 DA-O) is layered on top, per the project's standard row ordering.

**Why this row matters beyond "one more backbone":** CLAUDE.md's Backbones section names
KPConv's claimed strength as density-robustness ("grid-subsampled neighborhoods, not fixed point
count"), in contrast to PointNet++'s named weakness (fixed-radius ball query is density-sensitive)
and DGCNN's (unrestricted k-NN lets noisy points become spurious graph edges). This is directly
relevant to the still-unverified hypothesis logged in `step_notes/A3_PointNet2_DA_S.md` for why
DefRec's self-supervised reconstruction pretext helped PointNet++'s DA-S row (A3) but hurt
DGCNN's (C3): if that hypothesis is right (DefRec's benefit is specifically about fixing
density-sensitivity), KPConv's own claimed density-robustness gives a third, differently-motivated
data point once B-block rows reach DA-S (B-block doesn't currently have a DA-S row planned in the
same slot as A3/C3, but B1's own baseline stability — collapse-epoch rate, stdev — is itself a
first, cheap read on how "already stable" or "already unstable" this backbone's untouched
baseline is, the same diagnostic A1 vs. C1 turned out to matter for interpreting A2/C2's
DA-A comparison and A3/C3's DA-S comparison).

## Status: KPConv reference repo integrated, model/training script built, CPU smoke test passed,
submitted to the cluster GPU (2026-09-12)

## The real setup friction CLAUDE.md warned about — and what it actually was

CLAUDE.md's Backbones section describes KPConv as needing "a compiled CUDA extension." Having
now actually integrated it: **this isn't quite right.** The compiled pieces
(`cpp_wrappers/cpp_subsampling`, `cpp_wrappers/cpp_neighbors` — grid subsampling and radius
neighbor search, both CPU-side preprocessing operations used to build the multi-layer batch
structure KPConv's blocks consume) are **plain CPU C++ extensions, not CUDA.** They compiled and
ran correctly directly on the cluster's login node — no GPU, no CUDA toolkit, no `nvcc` needed
at all for these two ops (confirmed: `torch.utils.cpp_extension.CUDA_HOME` is `None` on the login
node, and compilation/import/execution all succeeded anyway). The actual KPConv convolution
itself (`models/blocks.py::KPConv`) is plain PyTorch — it runs on GPU automatically like any
other `nn.Module`, no CUDA-specific compilation step anywhere in this integration. Corrected in
CLAUDE.md's Backbones entry.

The friction was real, just a different kind: **both compiled extensions failed to build
out-of-the-box** against this project's environment (numpy 2.2.6, setuptools 83.0.0, Python
3.10) — two distinct, well-documented classes of incompatibility between ~2020-era C-extension
code and a modern numpy/setuptools toolchain:

1. **`numpy.distutils` removal.** Both `cpp_wrappers/*/setup.py` files used
   `from distutils.core import setup, Extension` + `numpy.distutils.misc_util.get_numpy_include_dirs()`.
   `numpy.distutils` still imports under Python 3.10 (numpy 2.2.6 ships a compatibility shim,
   confirmed by checking directly rather than assuming) but its `ccompiler.new_compiler()` calls
   modern setuptools' vendored `_distutils.Compiler.__init__` with an incompatible number of
   positional arguments (`TypeError: Compiler.__init__() takes from 1 to 3 positional arguments
   but 4 were given`) — a genuine, verified API mismatch between numpy.distutils and setuptools
   >=60ish, not a numpy-version-only issue. **Fix**: replaced with the standard modern
   equivalent — `from setuptools import setup, Extension` + `include_dirs=[numpy.get_include()]`
   — in both `cpp_wrappers/cpp_subsampling/setup.py` and `cpp_wrappers/cpp_neighbors/setup.py`.
2. **`PyArray_DATA`/`PyArray_NDIM`/`PyArray_DIM` signature tightening.** Modern numpy's C headers
   declare these as `(const PyArrayObject *arr)`, but both `wrapper.cpp` files pass plain
   `PyObject*`-typed variables directly (valid in older numpy versions, which had looser/macro-based
   signatures) — g++ correctly refused the implicit `PyObject* -> const PyArrayObject*`
   conversion (confirmed via the actual compiler error, not assumed from numpy's changelog).
   **Fix**: added explicit `(PyArrayObject*)` casts at all 19 call sites across both `wrapper.cpp`
   files (14 in `cpp_subsampling`, 5 in `cpp_neighbors`) via a targeted `sed` substitution, not a
   hand-edit of each site individually (verified the substitution's safety first: every call site
   had a simple identifier argument, no cast-affecting expressions).

Both fixes are **minimal compatibility patches to the vendored repo's own files** — the same
category CLAUDE.md already documents for DefRec_and_PCM ("`PointDA/data/dataloader.py` used
`np.int` ... patched to plain `int`"), not project-specific logic, and not a violation of the
"vendored repos stay unmodified upstream, only used as a library" rule in spirit (the alternative
— vendoring a fork, or reimplementing grid-subsampling/radius-neighbor-search from scratch in this
repo — would be strictly worse: harder to track against upstream, and CLAUDE.md explicitly says
"do not reimplement the reference architectures"). Verified working after patching: both `.so`
files compiled cleanly (only pre-existing signedness/deprecation warnings, no errors) and were
functionally tested directly (`grid_subsampling.subsample`, `radius_neighbors.batch_query`) on
synthetic data before touching any real pipeline code.

**Vendored as a sibling repo**, same convention as DefRec_and_PCM/Pointnet_Pointnet2_pytorch:
`../KPConv-PyTorch` (next to `plant3d-thesis/` under `/home/pearl/25m0301/`), from
`https://github.com/HuguesTHOMAS/KPConv-PyTorch`, pinned at commit
`d19c575d3fa9fcfd5a74845b5b27aac7e50472c7`. `KPCONV_ROOT` env var (mirroring `DEFREC_ROOT`/
`POINTNET2_ROOT`) points training scripts at it.

## Design decisions for the model/data integration

1. **A fundamentally different batch format than DGCNN/PointNet2.** KPConv's blocks
   (`models/blocks.py::SimpleBlock`/`ResnetBottleneckBlock`/etc.) read `batch.points[layer]`/
   `batch.neighbors[layer]`/`batch.pools[layer]`/`batch.upsamples[layer]`/`batch.lengths[layer]` —
   every sample's points concatenated into one flat per-layer array ("stacked batch"), not a
   fixed `(B, 3, N)` tensor. `adapters/kpconv_collate.py` builds this by reusing
   `datasets.common.PointCloudDataset.segmentation_inputs` (and the `batch_neighbors`/
   `batch_grid_subsampling` helpers it calls, which wrap the compiled extensions) **completely
   unmodified** — the same construction every reference-repo dataset class (S3DIS, ModelNet40,
   ...) already uses. `KPConvBatchBuilder` is a minimal `PointCloudDataset` subclass whose only
   job is holding `self.config` so that inherited method has what it needs — the same reuse
   pattern every real dataset class in the reference repo already does, not a new invention.
2. **No new Dataset class needed.** `PlantClsSegDataset`/`PlantSpeciesDataset` (already used
   unchanged by every prior row) already return exactly the `(pts, species_label, seg_label)` /
   `(pts, species_label)` tuples `kpconv_collate.py`'s two collate functions expect — only a new
   `collate_fn` was needed, not a new dataset. `NWORKERS=0` for this row specifically (every
   other row uses 2): the collate function calls into the compiled C++ extensions per batch, and
   multi-worker safety with these particular compiled modules wasn't verified — a candidate
   speed-up for later, not a correctness concern for this first baseline.
3. **`neighborhood_limits` left uncalibrated (empty list).** The reference repo's own dataset
   classes run a calibration pass (histogram-based percentile estimation over many batches) to
   cap each layer's neighbor-matrix width for memory/compute efficiency. Skipped here as a
   deliberate simplification: `PointCloudDataset.big_neighborhood_filter` treats an empty list as
   a no-op (returns neighbor matrices uncapped, not incorrect), and this project's point clouds
   (fixed N=4096, whole small objects) are far smaller than the reference repo's typical
   scene-segmentation crops (10k-100k+ points) where calibration matters most for memory. Flagged
   as a candidate follow-up if runtime/memory become a problem, not assumed to be needed.
4. **Joint cls+seg model built by mirroring `KPFCNN.__init__`/`.forward`'s encoder/decoder
   block-construction loop line-for-line** (same `block_decider` calls, same skip-connection
   bookkeeping) in a new `KPConv_ClsSeg` class — neither `KPCNN` (cls-only) nor `KPFCNN` (seg-only)
   in the reference repo does both, so this mirrors the same "re-sequence the base class's own
   submodules rather than subclass blindly" approach `DGCNN_ClsSeg` already used for DGCNN.
   Classification head is added off the bottleneck's globally-pooled feature (via
   `models/blocks.py::global_average`, the same function `GlobalAverageBlock`/`KPCNN`'s own
   `'global_average'` architecture block already uses) — composed into the SAME forward pass
   that also runs the segmentation decoder, rather than as a separate network.
5. **Per-point segmentation output reshaping is exact, not approximate.** KPConv's per-point
   features live in a stacked `(total_points, C)` layout; every other row's seg-loss/eval helpers
   (`compute_seg_loss`/`seg_logits_for_species`/`seg_eval_metrics`, copied unchanged from
   `train_c1_dgcnn_da0.py`) expect `(B, C, N)`. Since every sample here is a FIXED N=4096 points
   (no cropping/variable length, unlike the reference repo's usual use case),
   `batch.lengths[0]` is always `[N]*B` — the finest-layer output is guaranteed to be B
   contiguous chunks of exactly N rows each, in batch order, so `(B*N, C) -> (B, N, C) ->
   permute -> (B, C, N)` is an exact reshape, not an approximation. This let every other row's
   seg-loss/eval code be reused completely unmodified.
6. **Architecture: 3 layers (2 `resnetb_strided` poolings), no deformable blocks.** Chosen to
   match `PointNet2_ClsSeg`'s own 3-level SA/FP depth, for a comparable-depth cross-backbone
   architecture rather than the reference repo's much deeper room-scale-scene defaults (S3DIS
   uses 5 layers with deformable convolutions in the last two). Vanilla (rigid) KPConv is enough
   to test this backbone family per CLAUDE.md's Backbones section (density-robustness via
   grid-subsampled neighborhoods, no mention of needing the deformable variant specifically) —
   deformable KPConv is a separate, optional refinement, not required for this row.
7. **Hyperparameters grounded in a cross-check against PointNet2's own working values, not
   independently tuned from scratch.** `first_subsampling_dl=0.08` gives a layer-0 conv radius of
   `0.08*2.5=0.2` and layer-1 of `0.4` — matching `PointNet2_ClsSeg`'s own first two SA radii
   (`radius=0.2`/`0.4` in `adapters/models_pointnet2.py`) on this SAME normalized-to-unit-sphere
   data (`scripts/augmentations.py::normalize`). This is a plausibility cross-check (both
   backbones landing on similar absolute neighborhood sizes for the same data is reassuring), not
   a rigorous per-backbone tuning exercise — flagged as such. `first_features_dim=64` (bottleneck
   ends at 256 after 2 stridings) is a resource-conscious default for this first baseline, smaller
   than the reference repo's own defaults (128), not exhaustively tuned.
8. **`in_features_dim=1`**: a constant "ones" feature per point (the same convention
   `datasets/ModelNet40.py` uses when no color/normal input is available) — KPConv relies on
   kernel-point geometry, not input features, so this is the standard choice for XYZ-only data.

## What was done and how

1. Cloned `KPConv-PyTorch` as a sibling repo, pinned at commit `d19c575d3fa9fcfd5a74845b5b27aac7e50472c7`.
2. Compiled both C++ extensions (`cpp_wrappers/cpp_subsampling`, `cpp_wrappers/cpp_neighbors`),
   applying the two compatibility patches above — see that section for the exact errors and fixes.
   Verified both extensions functionally (`grid_subsampling.subsample`/`radius_neighbors.batch_query`
   on synthetic data) directly on the login node before touching any project code.
3. Wrote `adapters/kpconv_collate.py` (`KPConvBatchBuilder`, `KPConvBatch`,
   `make_kpconv_collate_fn`, `make_kpconv_collate_fn_cls_only`, `reshape_seg_labels`).
4. Wrote `adapters/models_kpconv.py` (`PlantKPConvConfig`, `KPConv_ClsSeg`). Hit and fixed the
   same `models`-package-name collision already documented for PointNet2's integration
   (`adapters/models.py` vs. the reference repo's namespace-package `models/` directory) —
   identical fix: add the vendored repo's `models/` subdirectory itself to `sys.path` and import
   `blocks` by its unique name directly, bypassing the `models.` prefix.
5. Wrote `adapters/train_b1_kpconv_da0.py`, mirroring `train_c1_dgcnn_da0.py`'s DA-0 training
   loop pattern (identical `L_cls`/`L_seg`/Kendall combination, identical model-selection
   protocol) with the batch/collate mechanics swapped for KPConv's requirements.
6. **Standalone sanity check before the full smoke test** (a synthetic 3-sample batch, N=200,
   run directly through the collate function and model): confirmed batch construction (3 layers,
   correct neighbor/pool/upsample shapes), model construction (750,137 parameters), and a full
   forward pass (correct `cls`/`feat`/`seg_feat` shapes: `(3,2)`/`(3,256)`/`(3,64,200)`) all work
   before spending time on a full CPU run with real data.
7. Added `jobs/b1_kpconv_da0.sbatch` — same cluster settings as every other row (dgx
   partition/qos, 1 GPU, 4h wall time), `KPCONV_ROOT` exported alongside `DEFREC_ROOT`.
8. **CPU smoke test (1 epoch, batch_size=4, real data, `--gpu -1`) — passed.** Launched on the
   cluster login node (not a laptop — this integration happened directly on `login1.prajna`,
   consistent with the compiled-extension work above) at 13:32, ran ~35 min end-to-end. A local
   client disconnect partway through did not affect it: the process had already been reparented
   to PID 1 (`ppid=1`, confirmed via `ps`) and kept running as an orphaned background process
   uninterrupted, finishing normally with no intervention needed. Sane, non-degenerate numbers
   throughout — no shape/gradient errors, no NaNs, no collapsed predictions: source val cls acc
   0.7556 (avg acc 0.8333), Tomato seg mIoU 0.3170 (acc 0.7318), Maize seg mIoU 0.2027
   (acc 0.5543), target (Pheno4D) cls-only eval acc 0.8730 (avg acc 0.8261) — this last number is
   from a single untrained-selection epoch on a 1-epoch smoke run, not a real result, just
   confirmation the target eval path runs correctly end-to-end. `results/_smoketest_b1/` deleted
   after confirming the log content, same convention as every prior row's smoke test.

---
**2026-09-12:** Submitted `jobs/b1_kpconv_da0.sbatch` to the `dgx` partition — job **308845**
(running on `cn14-dgx`). Queue was empty beforehand. Awaiting completion.
