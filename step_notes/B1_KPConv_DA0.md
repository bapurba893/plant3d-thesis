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

## Status: DONE. KPConv reference repo integrated; hit and fixed a real CUDA OOM bug (uncapped
neighbor matrices), then a real GPU/node-contention timing problem (24h budget too small) — see
the two "Incident" sections below. Job 309015 completed in 2h15m (2026-09-13) — see "Results"
below for the full-trajectory comparison against A1/C1.

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
   `collate_fn` was needed, not a new dataset. `--num_workers` defaults to 0 for this row
   specifically (every other row uses 2): the collate function calls into the compiled C++
   extensions per batch, and multi-worker safety with these particular compiled modules wasn't
   verified. **This flag was later tried at 4 and reverted** — see the "Incident" section below;
   the caution here turned out to be justified, not just conservative.
3. **`neighborhood_limits` left uncalibrated (empty list) — ORIGINALLY, since corrected.** The
   original reasoning here (this project's point clouds, fixed N=4096 whole small objects, are
   far smaller than the reference repo's typical scene-segmentation crops of 10k-100k+ points,
   so skipping calibration seemed like a safe simplification, flagged only as "a candidate
   follow-up if runtime/memory become a problem, not assumed to be needed") **turned out to be
   wrong in practice** — see the "Incident" section below for the full story: leaving this
   uncapped caused a job to stall for over an hour and a follow-up debug run to hit a 31.86 GiB
   CUDA OOM. Real calibration (`adapters/kpconv_collate.py::calibrate_neighborhood_limits`) is
   now run once at the start of every training run.
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

---
## Incident (2026-09-12): job 308845 stalled, root-caused to uncapped neighbor matrices, fixed

**Symptom.** Job 308845 showed no training progress for over an hour (`squeue` still `RUNNING`,
but `run.log` had nothing past the one-time setup lines — data loading, class weights — and
`sacct`'s live CPU-time accounting stayed near zero across repeated samples a few minutes apart).
The training loop only logs once per epoch (`Trn - Source {epoch}, ...`), so there was no
per-batch visibility into whether it was working slowly or actually stuck. Given every other
row in this project finished its full 100-epoch run in 13 min–1h04m, and this hadn't logged even
one epoch after 70+ minutes, the job was cancelled (`scancel 308845`) rather than left to
potentially run for the full 4h limit for nothing.

**Diagnosis.** Added a `--verbose_batches` flag (per-batch collate/step timing, gated so normal
runs stay at the established one-line-per-epoch log convention) and ran a short, time-bounded
debug job (job 308884, `--epochs 1`, 25 min cap) to get direct visibility. It crashed on the
**3rd training batch** with:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 31.86 GiB. ...
File "KPConv-PyTorch/models/blocks.py", line 301, in forward
    sq_distances = torch.sum(differences ** 2, dim=3)
```
Back-calculating from the allocation size (31.86 GiB / (n_points≈65536 × 15 kernel points × 3
dims × 4 bytes)) gives an average of **~2900 neighbors per point** for whatever batch triggered
this — confirming the exact risk flagged (but left unaddressed) in this file's original design
decision #3: leaving `neighborhood_limits` as an empty list makes
`PointCloudDataset.big_neighborhood_filter` a no-op, so neighbor index matrices are never capped.
A small fraction of augmented batches produce local point clusters dense enough to blow this up.
This also retroactively explains job 308845's "hang": with no per-batch logging, a batch whose
CPU-side neighbor search (or eventual GPU tensor construction) is this pathologically expensive
is indistinguishable from a genuine deadlock — it likely wasn't stuck, just extremely slow on
whichever specific batch it drew, then probably would have crashed the same way eventually.

**Fix: real neighbor-count calibration**, matching how the reference repo's own dataset classes
(`datasets/ModelNet40.py::calibration`, etc.) are meant to be used, minus their point-budget/
dynamic-batch-size machinery (irrelevant here — every sample is already a fixed N=4096, so a
plain fixed-`batch_size` DataLoader is used for calibration too). Added
`calibrate_neighborhood_limits()` to `adapters/kpconv_collate.py`: runs the SAME uncapped collate
over ~10 real training batches, builds a per-layer neighbor-count histogram (same method as the
reference repo's own calibration code), and sets each layer's limit to the 90th-percentile count
(`untouched_ratio=0.9`, the reference repo's own default — at most ~10% of points get their
neighbor list truncated). Calibration itself never touches the GPU (pure CPU/numpy), so it's safe
from the same OOM even if it happens to sample a pathological batch.

**A real bug caught while wiring this in**: `neighborhood_limits` lives on the
`PointCloudDataset`/`KPConvBatchBuilder` *instance* (`self.neighborhood_limits`), not on
`self.config` — confirmed by reading `datasets/common.py::PointCloudDataset.__init__` (always
resets `self.neighborhood_limits = []`) and `big_neighborhood_filter` (reads `self.
neighborhood_limits` directly) rather than assuming config was the right place to store it. The
first version of this fix set `config.neighborhood_limits` and would have silently done nothing.
Fixed in `KPConvBatchBuilder.__init__`, applied after `super().__init__()` runs (which is what
resets it).

**Verified on the CPU (login node), not GPU** — three separate GPU debug/validation attempts
(jobs 308884's follow-up 308889, and 308943) landed on a node with highly variable timing
(data loading alone ranged from 9 seconds to 10 minutes across otherwise-identical runs; one
timed out at its 30/40-min bound mid-calibration with no error, just node/NFS-load variability
unrelated to this code) — GPU debug runs stopped being a reliable way to validate quickly.
Switched to the same low-risk method used for every prior row's initial validation: a bounded
CPU run on the login node (`timeout 550 python ... --gpu -1 --verbose_batches`). This completed
calibration cleanly in ~4.6 min, producing `neighborhood_limits = [499, 45, 31]`, then ran 4
batches with bounded, non-crashing per-batch costs (collate 24-39s, fwd+bwd+step 10-16s on CPU)
before the timeout cut it off deliberately — confirms the fix prevents the OOM.

**A second attempted fix, tried and reverted**: since collate (CPU-bound, single-threaded — no
OpenMP in the vendored `cpp_wrappers/` source, checked directly) dominates per-batch cost, tried
parallelizing it via `DataLoader(num_workers=4)` (added a `--num_workers` CLI flag). This made
things *worse*, not better: a CPU debug run with `--num_workers 4` produced no output at all
(not even the calibration line) within a 550s bound, versus `num_workers=0`'s clean 4.6 min
calibration — consistent with this file's original, now-vindicated caution that "multi-worker
safety with these compiled modules wasn't verified." Left the `--num_workers` flag in place
(defaults to 0, the safe/verified setting) for future investigation, but the production job does
NOT use it.

**Real, accepted cost: this row is much slower than every other row in the project, not a bug.**
At ~30-50s/batch (collate-dominated, CPU-bound, unaffected by GPU) × 17 batches/epoch × 100
epochs, a full run is expected to take on the order of **10-15+ hours** — vs. every other row's
13 min–1h04m. Checked whether the project's own 4h-per-job convention was a hard cluster limit
before accepting this: it is not (`scontrol show partition dgx` → `MaxTime=6-00:00:00`; `sacctmgr
show qos dgx` → no `MaxWall` set) — the 4h budget in every prior `jobs/*.sbatch` was this
project's own convention, sized to match the other backbones' actual runtimes, not a cluster
policy. `jobs/b1_kpconv_da0.sbatch` raised to `--time=24:00:00` accordingly (`--cpus-per-task=4`
also added, though collate is single-threaded so this mainly guards against node-level CPU
oversubscription, not a parallelism win). `--verbose_batches` is kept ON for the real production
run (unlike a normal row) specifically so this run stays observable rather than repeating
308845's silent-stall failure mode if node variability recurs.

**2026-09-12:** Cleared the stale partial `results/B1_kpconv_da0/run.log` left by cancelled job
308845 (would otherwise have been appended to, not overwritten — `IOStream` opens in append
mode) and submitted the fixed job — **308956**, running on `cn14-dgx`. This is expected to run
for many hours; results to follow once it completes.

---
## Second incident (2026-09-12): job 308956's 24h budget also too small — real GPU/node contention

Checked on 308956 after ~2 hours: 4/100 epochs done, with striking per-epoch variance (14-51
min/epoch) and per-batch `fwd+bwd+step` timing swinging from 0.2s to 86.4s within the same
epoch. Averaging the 4 observed epochs (~28.75 min/epoch) and extrapolating to 100 epochs gives
**~48 hours** — roughly double the 24h budget set after the first incident, which was sized from
a CPU-only extrapolation that couldn't see this GPU-side variance. `scontrol update
JobId=308956 TimeLimit=...` was tried and refused ("Access/permission denied") — this account
can't extend a running job's time limit, only set it at submission.

**Why this matters more than it might look:** `train_b1_kpconv_da0.py` only writes its final
summary (`Best model at epoch...`, `FINAL best-model source val...`, `FINAL target test
accuracy...`, confusion matrix) *after* the full epoch loop completes — checked directly
(`grep -n "torch.save\|checkpoint\|best_model"` shows the periodic `model.pt` save happens
inside the loop, but the final-summary `io.cprint` calls are all after it). A 24h timeout at the
observed pace would have killed the job somewhere around epoch 50 with only per-epoch
`Trn`/`Val`/`Eval` lines and a stale `model.pt`, not the full 100-epoch trajectory this analysis
needs to be comparable to A1/C1.

**Decision (confirmed with the user rather than assumed):** kill 308956 now, while only ~2h/4
epochs are sunk, rather than let it run the full 24h for a guaranteed-incomplete result. Archived
the partial run for reference (`results/B1_kpconv_da0/run_v1_killed_at_epoch4_slow_node.log`,
`model_v1_killed.pt` — same "keep, don't delete" convention as C2's buggy v1 run) and resubmitted
with `--time=60:00:00` (25% margin over the ~48h extrapolation) — job **309015**, on `cn17-dgx`
(different node than 308845/308956's `cn14-dgx`, for whatever that's worth against a
contention-driven explanation).

**Per the user's explicit instruction, this 60h budget now applies to every future Block B row
(B2-B5) from the start, not just B1** — the underlying cause (KPConv's CPU-bound collate,
compounded by shared-A100 contention on the `dgx` partition) is not B1-specific, so there is no
reason to expect B2-B5 to be faster. Flagged in `CLAUDE.md`'s Backbones section so this doesn't
need rediscovering when B2 starts.

---
## Results (2026-09-12/13, job 309015, 2h15m — far faster than the 48-60h feared)

**Job 309015 landed on `cn17-dgx` instead of `cn14-dgx` and saw none of 308956's contention**:
epoch 0 finished in under 4 minutes (collate ~2.6-3.5s/batch, fwd+bwd 0.2-0.9s/batch, vs.
308956's 14-51 min/epoch). All 100 epochs completed in 2h15m total — this is now hard evidence
the earlier slowdown really was transient shared-GPU contention on that specific node, not a
structural property of every KPConv run. The 60h budget stays the standing convention for B2-B5
regardless (a real risk that materialized once already, cheap to keep budgeting for, expensive
to get caught by again), but it may turn out to be conservative rather than exactly-needed most
of the time.

**Protocol-selected checkpoint** (best epoch by lowest source val total loss) — best epoch was
**99, the literal last epoch** (source val total loss 0.3317, still monotonically improving
through the end of training, unlike C1's epoch 95 or A1's epoch 97):

| Metric | A1 (PointNet++, DA-0) | C1 (DGCNN, DA-0) | B1 (KPConv, DA-0) |
|---|---|---|---|
| Source val cls acc | 1.0000 | 1.0000 | 1.0000 |
| Target cls acc | 0.4603 (avg 0.5750) | 0.7460 (avg 0.7446) | **0.4444 (avg 0.5625)** |
| Tomato seg mIoU (source val) | 0.3207 (acc 0.6729) | 0.3248 (acc 0.6645) | **0.3808** (acc 0.7551) |
| Maize seg mIoU (source val) | 0.4741 (acc 0.9055) | 0.3427 (acc 0.7698) | 0.4414 (acc 0.8801) |

**Full-trajectory analysis** (all 100 epochs' target-eval numbers, same methodology as
A1-A3/C1-C5 — `avg_acc == 0.5` exactly identifies a total-collapse-to-one-class epoch on this
63-sample, Tomato=40/Maize=23 target set):

| Metric | A1 (DA-0) | C1 (DA-0) | B1 (DA-0) |
|---|---|---|---|
| Full-run mean target acc | 0.599 | 0.791 | **0.629** |
| Full-run stdev | **0.196** | 0.104 | **0.203** |
| corr(selection-loss, target acc) | −0.165 | −0.286 | **+0.316** |
| Collapse epochs (all-one-class) | 17/100 | 3/100 | **3/100** |
| Quartile means (Q1→Q4) | 0.676/0.717/0.507/0.495 | 0.798/0.834/0.773/0.760 | 0.750/0.617/0.667/0.484 |
| Selected-checkpoint acc | 0.4603 | 0.7460 | 0.4444 |

### Answering the user's question: does KPConv's claimed density-robustness show up as a stable baseline like DGCNN, or something else?

**Mixed — it does not cleanly confirm the density-robustness-implies-stability hypothesis.**
Two different stability measures point in different directions:

- **By collapse-epoch count, B1 looks DGCNN-like**: 3/100 epochs collapsed to predicting one
  class for the whole target set, matching C1 exactly and far better than A1's 17/100. On this
  specific, coarse measure, KPConv's baseline is as stable as DGCNN's, not as unstable as
  PointNet++'s.
- **By full-run stdev, B1 is the LEAST stable of the three backbones**: 0.203, slightly *higher*
  than even A1's 0.196 (which is itself far more volatile than C1's 0.104). So on the
  continuous, finer-grained measure of epoch-to-epoch swing, KPConv's claimed density-robustness
  does not translate into a calmer trajectory — if anything the opposite.
- **By selection-signal correlation, B1 is worst of the three, and in the "actively
  counterproductive" category** (only C3's DGCNN/DA-S row showed this before): `corr(source val
  loss, target acc) = +0.316` — positive, meaning epochs with a *better* (lower) source-domain
  validation loss tend to have *worse* target accuracy, the wrong direction for the
  model-selection protocol to be useful. A1 and C1 are both negative (informative, if weakly for
  A1); B1 is the only DA-0 row across three backbones where the selection signal actively points
  the wrong way. Directly visible in the outcome: the protocol selected epoch 99 (last epoch,
  lowest source loss) and got target acc 0.4444 — near the *bottom* of B1's own trajectory
  (full-run mean 0.629), not a representative or lucky-high pick.

**So KPConv's baseline is not simply "the stable one" or "the unstable one" — it has its own,
distinct instability signature**: resistant to total class-collapse (like DGCNN) but with high
continuous variance and an actively misleading selection signal (worse than either other
backbone in that specific respect). This complicates rather than confirms the working hypothesis
from `step_notes/A3_PointNet2_DA_S.md` (that DefRec helped PointNet++ specifically because of a
named density-sensitivity weakness KPConv shouldn't share) — B1 doesn't show the kind of clean
DGCNN-like stability that hypothesis would most naturally predict for a density-robust backbone,
though it also doesn't show A1's specific *systematic* failure mode (see below) in quite the same
form. Since B1 is DA-0 (no adaptation method at all), this is a baseline-only data point; whether
KPConv's DA-S row (not currently planned in the strategy table's Block B slot) would behave more
like A3 or C3 remains untested and out of scope here.

### Same DA-0 drift pattern as every prior backbone, but settling at a lower floor

The last ~20 epochs oscillate tightly in a narrow band (0.43-0.52), a clear "freeze" — the same
qualitative pattern CLAUDE.md documents for C1 ("settling frozen... for the last ~10 epochs") and
the erosion pattern noted for A1, except **B1 freezes at a much lower floor** (~0.46-0.48 vs.
C1's 0.746). Q4's stdev (0.037) is the tightest of any quartile in this run, confirming the
freeze is real and not noise. This is now the third backbone showing the identical qualitative
shape (source metrics keep improving; target accuracy rises early, then erodes, then plateaus low
while nothing in the training signal can detect or prevent it) — strengthening the case that this
is a general DA-0 property of this dataset, independent of architecture, motivating every DA-A/
DA-S/DA-O row already built.

### A shared systematic bias with A1, worth flagging

B1's final confusion matrix: Tomato 5/40 correct (35 misclassified as Maize), Maize 23/23
correct — precision/recall highly lopsided (Tomato precision 1.00/recall 0.125; Maize precision
0.40/recall 1.00). This is the same *systematic* Maize-leaning bias CLAUDE.md documents for A1
("34/40 target Tomato plants misclassified as Maize"), not the milder, more balanced error
pattern C1 shows. Two of the three backbones (A1, B1) share this specific failure mode; DGCNN
does not — a plausible, unverified hypothesis is that this tracks Crops3D's 73%-Maize source
majority combined with something DGCNN's dynamic-graph features do differently from both
ball-query (PointNet++) and kernel-point (KPConv) approaches, but this is not tested here.

### One genuinely positive finding: segmentation quality is competitive-to-best

Despite the weak classification-transfer story, B1's source-val segmentation is strong relative
to the other two DA-0 baselines: **Tomato mIoU 0.3808 is the highest of the three** (vs. A1's
0.3207, C1's 0.3248), and Maize mIoU 0.4414 is close to A1's best-of-three 0.4741 and clearly
above C1's 0.3427. So KPConv's per-point segmentation head, despite sharing this row's
classification-transfer weaknesses, produces the best/near-best organ-segmentation quality of the
three backbones under DA-0 — a real, separate finding worth carrying forward independent of the
stability story above.

### Conclusion

Block B's own DA-0 baseline (B1) is now established. It does **not** cleanly validate "KPConv is
density-robust → stable baseline" — collapse-epoch count says yes, continuous variance and
selection-signal correlation say no, and it shares A1's systematic Maize-bias failure mode rather
than showing something new. It does, however, produce the best segmentation quality of the three
backbones under DA-0, and (once run on an uncontended node) trains just as fast as DGCNN/
PointNet++ in practice, not meaningfully slower despite the CPU-bound collate concern. Full
trajectory data in `results/B1_kpconv_da0/run.log`; the killed first attempt's partial data is in
`results/B1_kpconv_da0/run_v1_killed_at_epoch4_slow_node.log` for reference only, not used in
any of the numbers above.
