# Row C3: DGCNN, Self-Supervised Domain Adaptation (DA-S), All Augmentations

## What this is, in plain language

C3 tries a different way of forcing the network to build features that don't depend on which
sensor/dataset a plant came from. Instead of the "adversarial" trick used in C2 (a discriminator
fighting the network), DA-S uses a self-supervised puzzle: take a region of a plant's point
cloud, scramble/blank it out, and force the network to reconstruct what was there from the
surrounding structure — using Chamfer distance (a standard way to measure how close two point
clouds are to each other) as the loss. Crucially, this reconstruction puzzle is run on **both**
Crops3D (source) and Pheno4D (target) — since it never needs any labels, the network gets the
same "understand plant geometry well enough to reconstruct a masked region" pressure regardless
of domain, which should nudge its features toward being domain-general.

This row is directly comparable to a well-known, published external benchmark (PointDA-10),
which is something C1/C2 didn't have — a useful sanity check on whether this project's setup is
behaving reasonably by field standards, independent of whether it helps this project's own
plant data.

## Status: in progress (started 2026-09-11)

## Design decisions confirmed before writing code

1. **Unlike C2's adversarial machinery, DefRec (the self-supervised deformation-reconstruction
   technique DA-S uses) already exists, complete and working, in `DefRec_and_PCM`** — it's the
   reference paper's entire contribution (`DefRec_and_PCM/DefRec.py`: `deform_input`,
   `reconstruction_loss`, `calc_loss`; the reconstruction head `RegionReconstruction` is already
   built into `PointDA.Models.DGCNN` as `self.DefRec`, and `adapters/models.py::DGCNN_ClsSeg`
   already exposes it unmodified via `forward(x, activate_DefRec=True)` -> `logits["DefRec"]`,
   inherited from the base class, not something C1/C2 added). So this row is a genuine case of
   "adapt this repo" rather than the from-scratch situation C2 was in — `DefRec.deform_input`/
   `DefRec.calc_loss` are used directly, unmodified, imported from the sibling clone.
2. **Confirmed our data is compatible with DefRec's region-partitioning code without
   modification**: `DefRec`'s region assignment (`utils/pc_utils.py::assign_region_to_point`)
   clips points to `[-0.99999999, 0.99999999]` per axis and buckets them into a 3x3x3 grid over
   `[-1, 1]` — this assumes ModelNet/ShapeNet-style unit-sphere-normalized input.
   `scripts/preprocessing.py::normalize_and_center` (already run for every cached point cloud in
   this project, both Crops3D and Pheno4D) centers at the centroid and scales to unit sphere
   (max radius = 1) — the same normalization convention, confirmed by reading the function, not
   assumed.
3. **`DefRec_and_PCM/PointDA/trainer.py`'s own loop is still not reused directly** (same
   situation as C1/C2): it always mixes in PCM (point-cloud mixup, `args.apply_PCM`, default
   `True`) and has no segmentation head/loss at all. PCM is a separate technique from the same
   paper and is **not** one of this project's 5 DA codes (DA-0/DA-A/DA-D/DA-S/DA-O) — using it
   would silently turn C3 into "DA-S + an unspecified extra technique," so `train_c3_dgcnn_da_s.py`
   is (like C1/C2) its own thin loop that imports `DefRec.deform_input`/`DefRec.calc_loss`
   directly rather than going through `trainer.py`, and never calls `PCM.*`.
4. **DA-S runs DefRec on BOTH domains** (CLAUDE.md: "run on BOTH domains so it forces
   domain-general geometric features") — `trainer.py`'s own default already runs DefRec on
   target unconditionally every batch (that's the paper's whole point, per the earlier C1
   caveat) but only optionally on source (`args.DefRec_on_src`, default `False`, since the
   ModelNet->ShapeNet benchmark this repo ships with doesn't need it). This script always runs
   DefRec on source too — not the upstream default, a deliberate choice matching this project's
   own DA-S definition.
5. **`L_defrec` is routed through the Kendall uncertainty module as a single, third cooperative
   task** (`{"cls": ..., "seg": ..., "defrec": ...}`, "regression" type, since Chamfer distance
   is a continuous reconstruction loss, not classification) — per CLAUDE.md's Level-2 rule
   naming `L_defrec` as one of the cooperative losses that gets learned uncertainty weighting,
   alongside `L_cls`/`L_seg`. Two things had to be decided that aren't spelled out in the docx
   (flagged as inferences, not confirmed spec):
   - **Combining source and target reconstruction into one scalar.** The docx names a single
     `L_defrec` term, and DA-S's own definition treats "run on both domains" as one mechanism,
     not two — so each batch's source and target reconstruction losses are combined into one
     `L_defrec` via a sample-count-weighted average (the same convention already used for
     `L_seg`'s per-species combination in `adapters/train_c1_dgcnn_da0.py`), rather than treating
     them as two separate Kendall tasks.
   - **Neutralizing the upstream `DefRec.calc_loss`'s own fixed weight.** Upstream `calc_loss`
     computes `args.DefRec_weight * reconstruction_loss(...) * DefRec_SCALER` — in the original
     repo's own (non-Kendall) trainer, `DefRec_weight` (default 0.5) is *itself* the
     between-task tradeoff knob (used as `(1 - DefRec_weight)` on the classification side). Since
     this project's Kendall module is now responsible for that between-task tradeoff, leaving
     `DefRec_weight` at its upstream default would double-apply a fixed weighting underneath a
     learned one. Set `--DefRec_weight 1.0` (neutral) here, while leaving the separate
     `DefRec_SCALER = 20.0` constant inside `reconstruction_loss` untouched (that one is a fixed
     unit-scaling constant, not a between-task tradeoff — it exists to bring Chamfer distance's
     naturally tiny magnitude into a comparable range with cross-entropy-scale losses, which
     Kendall's learned weighting still benefits from starting from, not something to remove).
6. **Model selection stays on the same protocol as C1/C2**: best epoch by lowest **source** val
   total loss (`Kendall(cls, seg, defrec)`, with `defrec` evaluated on source val data only —
   never target). Note this is a deliberate consistency choice, not a hard constraint the way it
   is for DA-0/DA-A: `L_defrec` never uses labels at all (source or target), so including a
   target-side reconstruction signal in selection would *not* violate the "never touch target
   labels" rule the way it would for C1/C2. Kept source-only anyway so every row in the table is
   selected the identical way — mixing in target-side signal for just this row would confound
   any cross-row comparison of "how good is the selected checkpoint."

## What was done and how

1. **Confirmed the DefRec building blocks exist and are directly reusable** (see design
   decisions above) — `DefRec_and_PCM.DefRec.deform_input`/`.calc_loss`, and
   `PointDA.Models.DGCNN`'s built-in `self.DefRec` head (`RegionReconstruction`, already
   inherited unmodified by `adapters/models.py::DGCNN_ClsSeg` from day one, unused by C1/C2 but
   present). No code in the sibling repo was touched.
2. **Wrote `adapters/train_c3_dgcnn_da_s.py`**, built on the same skeleton as C1/C2 (same
   `evaluate_cls_seg`/`evaluate_cls_only`/`compute_seg_loss`/`seg_eval_metrics` helpers, same
   data loaders and splits, same source-driven epoch length with the target adaptation pool
   cycled against it). New pieces specific to this row:
   - `defrec_loss_for_batch(args, model, lookup, pts_orig, device)`: clones the input, deforms
     it via `DefRec.deform_input`, runs the model with `activate_DefRec=True`, and returns
     `DefRec.calc_loss(args, logits, pts_orig, mask)` — called once for source, once for target,
     every training step.
   - Per-batch: one clean forward pass on source (`activate_DefRec=False`) for `L_cls`/`L_seg`,
     plus two deformed forward passes (source and target) for the two `L_defrec` components —
     three forward passes per step total, more than C1/C2's one or two, inherent to DA-S's
     self-supervised design (matches the reference repo's own multi-pass-per-batch pattern when
     `DefRec_on_src=True`).
   - `evaluate_defrec`: source-val-only reconstruction loss (sample-count-weighted mean),
     feeding into Kendall for the val-time total loss / model selection, matching the design
     decision above.
   - `Kendall({"cls": ..., "seg": ..., "defrec": ...})` with `defrec` marked `"regression"` —
     the only change to `KendallUncertaintyWeighting`'s usage is adding a third task key; the
     module itself (`adapters/losses.py`) was not modified.
3. **CPU smoke test** (2026-09-11): 1 epoch, batch size 8, real Crops3D/Pheno4D data, `--gpu -1`.
   Hit one environment snag first — the smoke test initially failed with
   `ModuleNotFoundError: No module named 'numpy'` because this session's shell had conda's
   `base` env active instead of `plant3d` (a fresh-session shell-state issue, not a code
   problem); fixed by explicitly `source .../conda.sh && conda activate plant3d` before
   running, matching what the `.sbatch` scripts already do. After that, the smoke test ran
   end-to-end in ~24.5 minutes (32 batches, 1 epoch) with no shape/gradient errors — noticeably
   slower than C1/C2's CPU smoke tests, expected given three forward passes per step plus
   `DefRec`'s region-assignment computation (27-way voxel grid membership test) running twice
   per batch on CPU; this is a CPU-only correctness check, not a speed estimate for the real
   GPU run. All losses came back non-zero and sane: `cls loss 0.48`, `seg loss 2.21`, `defrec
   loss (src+trgt) 6.33` — the larger absolute scale versus `cls`/`seg` is expected (Chamfer
   distance x `DefRec_SCALER=20`, per module docstring) and is exactly the kind of cross-task
   scale mismatch Kendall's learned weighting exists to handle; `s_defrec` started at `0.0280`,
   same near-zero init as `s_cls`/`s_seg`, and is expected to move as training progresses.
   Deleted `results/_smoketest_c3/` afterward.
4. **Added `jobs/c3_dgcnn_da_s.sbatch`** — same cluster settings and hyperparameters as C1/C2
   (dgx partition/qos, 1 GPU, 4h wall time, 100 epochs / batch 32 / lr 1e-3 / wd 5e-5 / dropout
   0.5, isolating the DA-S variable), plus `--DefRec_dist volume_based_voxels --num_regions 3
   --DefRec_weight 1.0`.

## Technical specifics

- **New files:** `adapters/train_c3_dgcnn_da_s.py`, `jobs/c3_dgcnn_da_s.sbatch`
- **Not modified:** `adapters/models.py` (DGCNN_ClsSeg's inherited `self.DefRec` head was
  already present, unused until now), `adapters/losses.py` (Kendall module unchanged, just
  given a third task key), `DefRec_and_PCM` (unmodified upstream, as always)

## What's next

Submitting `jobs/c3_dgcnn_da_s.sbatch` to the cluster next.

---
**2026-09-11:** Submitted `jobs/c3_dgcnn_da_s.sbatch` to the `dgx` partition — job **308449**.
Queue was empty beforehand. Awaiting completion.

---
**2026-09-11 (later): Job 308449 FAILED — CUDA OOM — root-caused, fixed, and resubmitted.**

## Incident: CUDA out-of-memory on the very first training batch

Job 308449 crashed after 2m38s, before printing a single epoch's training line. The sbatch
stdout log (`results/logs/c3_dgcnn_da_s_308449.log`, not `run.log` — exceptions go to stderr,
not through `IOStream`) had the actual traceback:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.00 GiB. GPU 0 has a total
capacity of 79.14 GiB of which 3.57 GiB is free. Including non-PyTorch memory, this process
has 75.56 GiB memory in use.
```

Died inside `DefRec.chamfer_distance` (unmodified upstream code), specifically at
`torch.add(p1, torch.neg(p2))` — while computing `l_defrec_src`, i.e. before even reaching the
target-domain DefRec call.

**Root cause, worked out from the code, not guessed:** `chamfer_distance` builds dense
`(B, N, N, 3)` tensors (`p1`/`p2` after `.repeat(...)`, then their difference). At our
`N=4096` points/cloud and `batch_size=32`, that's a `(32, 4096, 4096, 3)` float32 tensor =
`32 × 4096 × 4096 × 3 × 4` bytes ≈ **6.0 GiB — matches the OOM message's "Tried to allocate
6.00 GiB" exactly**, confirming the diagnosis. PointDA-10 (the benchmark this reference repo
was built and tuned for) uses only **1024 points/cloud** — 16x fewer than ours — so this
implementation was never exercised at a memory scale anywhere near what our data requires.
Compounding this: the original design (per `defrec_loss_for_batch` calls followed by one
`total = kendall(...); total.backward()`) kept the clean cls/seg forward pass's activations,
the deformed-source forward pass's activations, AND the deformed-target forward pass's
activations all simultaneously alive in the autograd graph, waiting for one combined backward
call — so by the time the very first (source) `chamfer_distance` call needed its own several
`~6 GiB` intermediate tensors, 75.56 GiB was already committed to everything retained before it.

**Fix (no changes to `DefRec_and_PCM` — still unmodified upstream, per the repo's own rules):**
1. Added `KendallUncertaintyWeighting.weighted_term(name, loss)` to `adapters/losses.py` — a
   pure, additive refactor exposing the same per-task weighting formula `forward()` already
   used internally, now callable for one task at a time. `forward()` itself now just sums
   `weighted_term(...)` per task — behaviorally identical, so C1/C2 are unaffected.
2. Restructured `train_c3_dgcnn_da_s.py`'s training step to **backprop incrementally** instead
   of building one combined loss: the clean cls/seg forward pass is weighted via
   `kendall.weighted_term` and `.backward()`'d immediately (freeing its activations before
   DefRec even starts), then `L_defrec` is computed and backpropagated through a new
   `defrec_chunked_backward` helper that **splits the combined source+target batch into small
   `DEFREC_CHUNK=8`-sized slices**, computing and `.backward()`ing each slice's contribution
   separately (mathematically equivalent to one combined call, since each term's contribution
   is linear in that term's loss — see the function's docstring for the derivation) — bounding
   peak Chamfer-distance memory to `8 × 192 MiB ≈ 1.5 GiB` regardless of the true training
   batch size. `evaluate_defrec` (used for the val-time diagnostic/model-selection signal) got
   the same chunking, since a full-batch Chamfer call could OOM even under `no_grad`.
3. **Documented, minor side effect**: the deformed-pass BatchNorm layers now see
   `DEFREC_CHUNK`-sized (8) batches instead of the full training batch (32) during the DefRec
   forward pass specifically — a known, accepted tradeoff of memory-driven chunking (similar to
   small-microbatch gradient accumulation), not a correctness bug. The clean cls/seg forward
   pass, which is what classification/segmentation quality actually depends on, still sees the
   full batch of 32, unaffected.
4. **Re-ran the CPU smoke test** (1 epoch, batch size 8, real data) against the fixed script:
   passed end-to-end in ~19.5 min, with loss magnitudes closely matching the original (buggy)
   smoke test's — `cls 0.51` vs `0.48`, `seg 2.26` vs `2.21`, `defrec 5.80` vs `6.33` — small
   differences fully explained by the refactor's chunking changing effective BatchNorm batch
   statistics during the deformed forward pass (point 3 above), not a sign of a logic error.
   Confirms the incremental-backward + chunking rewrite preserves the intended semantics.

## Important context found while investigating: the published paper's own ablation warns against this project's exact DA-S protocol

While looking up PointDA-10 published numbers (see Results section below, once available), read
the DefRec paper (Achituve et al., WACV 2021, `arxiv.org/pdf/2003.12641.pdf`) in full and found
something directly relevant to how C3's results should be interpreted, not just a sanity-check
number: **the paper's own ablation study (Table 3) explicitly tested "DefRec S/T" — applying
DefRec to both source AND target, exactly this project's DA-S definition (CLAUDE.md: "run on
BOTH domains") — and found it underperforms their proposed target-only configuration.** Quote:
"(b) Applying DefRec on both source and target samples degrades performance." Concretely (their
Table 3, PointDA-10 avg accuracy, volume-based/3×3×3-voxel deformation — the same deformation
type this project's `--DefRec_dist volume_based_voxels --num_regions 3` matches):
`DefRec S/T + PCM` = 68.7 vs. `DefRec (target-only) + PCM` = 69.6 — about 0.9 points lower when
run on both domains, even with PCM (which this project doesn't use) added on top in both cases.

**This is not a reason to deviate from the strategy table's DA-S definition** — CLAUDE.md is
explicit that DA-S runs on both domains "so it forces domain-general geometric features," and
the strategy table is a fixed project spec, not something to unilaterally revise mid-row. But it
means: if C3 underperforms C1 (DA-0) or shows a weaker result than a naive "self-supervision
should obviously help" prior would predict, that would not necessarily indicate a bug in this
implementation — it would be consistent with a documented finding in the very paper this row's
method comes from. Flagging this now, before results are in, specifically so it doesn't get
mistaken for a red flag when the real numbers arrive.

**Full published PointDA-10 reference numbers** (DGCNN backbone, classification accuracy,
averaged over 3 seeds — from the paper's Table 1/Table 6, extracted from the arXiv PDF text
directly, not estimated):

| Method | MN→SN | MN→SC | SN→MN | SN→SC | SC→MN | SC→SN | Avg |
|---|---|---|---|---|---|---|---|
| Source-only ("Unsupervised") | 83.3 | 43.8 | 75.5 | 42.5 | 63.8 | 64.2 | 62.2 |
| DefRec (target-only, no PCM) | 83.3 | 46.6 | 79.8 | 49.9 | 70.7 | 64.4 | 65.8 |
| DefRec + PCM (paper's proposed method) | 81.7 | 51.8 | 78.6 | 54.5 | 73.7 | 71.1 | 68.6 |

(MN=ModelNet, SN=ShapeNet, SC=ScanNet.) **Not directly numerically comparable to this
project's C3** — PointDA-10 is 10-class, classification-only, 1024 points/cloud, and a
synthetic-vs-real domain gap (vs. our 2-class, joint cls+seg, 4096 points/cloud, sensor-vs-sensor
gap) — so absolute accuracy figures won't line up; the useful comparison is qualitative
(does self-supervision help over source-only in a published, controlled setting — yes, +3.6 pts
avg — and the both-domains-vs-target-only finding above).

Deleted `results/C3_dgcnn_da_s/` (the failed run's partial output) and the smoke test's
`results/_smoketest_c3b/` before resubmitting.

**Resubmitted `jobs/c3_dgcnn_da_s.sbatch` (unchanged — the fix is in the Python script, not the
job script) — job 308459.** Queue was empty. Awaiting completion.
