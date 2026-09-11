# Row C4: DGCNN, Adversarial Domain Adaptation (DA-A), L-N Augmentation Only

## What this is, in plain language

C4 asks a narrower question than C2 did: DGCNN's known architectural weakness is that its
graph-building step (dynamic k-nearest-neighbors) has no protection against noisy points —
a single noisy point can end up wired into the feature graph as if it were a real neighbor,
polluting the features around it. C4 tests whether the adversarial domain-adaptation method
(the same one C2 used) still works when the *only* augmentation applied during training is
Gaussian jitter (simulated sensor noise) — no rotation, scaling, flipping, cropping, or
dropout. Everything else (backbone, adaptation method, data, hyperparameters) is identical to
C2; only the augmentation mix changes. That isolation is the whole point of this row.

**Why it matters right now**: C2's first real run showed a serious instability problem (see
`step_notes/C2_DGCNN_DA_A.md`) that was root-caused to a source/target label-prior mismatch
interacting badly with an unramped entropy-minimization term, and fixed (entropy weight now
ramps alongside the GRL alpha, same as `λ_p`). C4 reuses that corrected `adapters/dann.py`
unchanged — so C4 becomes a direct test of whether the adversarial method's remaining behavior
(even after the fix) is a property of the method itself, or specific to the ALL augmentation mix
C2 happened to use. Per explicit user instruction, C4 gets the same full-trajectory analysis
(not just the selected checkpoint) as soon as results are in, not as an afterthought — exactly
because this comparison is only meaningful if both rows are read the same, careful way.

## Status: in progress (started 2026-09-11)

## Design decisions

1. **Byte-for-byte identical adversarial machinery to (corrected) C2.** `adapters/dann.py` is
   imported unchanged — same `GradientReversalLayer`, `DomainDiscriminator`,
   `lambda_p_schedule`, `lambda_ent_schedule` (the ramped-entropy fix). `train_c4_dgcnn_da_a_ln.py`
   is a near-line-for-line copy of `train_c2_dgcnn_da_a.py`; the only functional difference is
   the augmentation pipeline passed to the dataset classes.
2. **New augmentation plumbing, since nothing before this row needed anything but "ALL".**
   `scripts/augmentations.py` gained `compose_pipeline_ln_only`/`_with_labels` — `normalize()`
   (a fixed preprocessing step, not an augmentation category — same treatment CLAUDE.md gives
   Pad3D) followed by `gaussian_noise()` alone, skipping G-R (rotate/flip/cubic symmetry), G-S
   (scale), and L-D (crop/dropout) entirely. `adapters/dataset.py`'s `PlantClsSegDataset`/
   `PlantSpeciesDataset` gained an `augment_mode` parameter ("all" default, unchanged behavior
   for every existing caller; "ln_only" selects the new pipeline) rather than hardcoding
   `compose_pipeline` — an additive change, C1/A1/C2/C3 don't pass this argument and are
   unaffected.
3. **L-N applied to BOTH domains**, same as C2 applied ALL to both domains — consistent with
   how every other row in this project treats "the augmentation column" as something applied
   uniformly to source and target during training, not just source.
4. **Model selection**: identical protocol to C1/A1/C2 — best epoch by lowest source val total
   loss, never touching target labels.
5. **Hyperparameters identical to (corrected) C2** (epochs, batch size, lr, wd, dropout, disc
   dropout, gamma, lambda_ent) — isolating the augmentation variable means everything else must
   stay fixed.

## What was done and how

*(To be filled in as work proceeds.)*
