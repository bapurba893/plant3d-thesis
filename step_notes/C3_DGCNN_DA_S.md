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
