# C6: DGCNN, DA-D (Deep CORAL) — added row

Not in CLAUDE.md's original strategy table (Block C is C1-C5: DA-0/DA-A/DA-S/DA-A-LN/DA-O — no
DA-D slot). Added per explicit user instruction, alongside its PointNet++ counterpart (A6),
same category of addition as B3b was to Block B. Numbered C6 (next available slot).

## Motivating question

DA-D (B3, KPConv) produced the second-best non-Oracle result in the whole project (+17.9 pts
full-run mean over B1, second only to B3b's DA-S +23.1). CLAUDE.md's cross-backbone FINAL
SUMMARY reads this, together with B3b, as evidence "what predicts success on KPConv is whether
the DA loss is cooperative (Kendall-weighted) rather than adversarial" — but DA-D has only ever
been tested on KPConv. C6 (and A6) test whether that pattern generalizes across backbones or was
KPConv-specific, the same kind of question DA-S's cross-backbone test (A3 vs. C3) already
answered in the "backbone-dependent, not universal" direction for that method.

DGCNN's own story so far: DA-A (C2) hurts badly (-15.8 pts full-run mean vs. C1). DA-S (C3) also
hurts (-8.8 pts). So DGCNN has NOT been helped by any adaptation method tried yet — C6 is the
third and most mechanistically distinct method tried on this backbone (feature-covariance
alignment, no discriminator, no reconstruction pretext).

## Implementation

`adapters/train_c6_dgcnn_da_d.py`: C2's exact plain-tensor DGCNN plumbing (`PlantClsSegDataset`/
`PlantSpeciesDataset`, `DGCNN_ClsSeg`, same source-train/source-val/target-adaptation-pool/
target-held-out-eval split) with B3's DA-D loss machinery substituted for C2's DANN machinery —
`adapters/coral.py::coral_loss` (already backbone-agnostic by design, no changes needed) between
`logits["feat"]` from a source batch and a target batch, `L_CORAL` Kendall-weighted alongside
`L_cls`/`L_seg` (cooperative DA loss per CLAUDE.md's loss architecture, unlike DA-A's fixed/
scheduled `L_dom`/`L_ent`). Model selection: same protocol as every row — best epoch by lowest
source val total loss.

CPU smoke test (1 epoch, real data) passed cleanly before submitting — sane non-degenerate
numbers (cls loss 0.51→0.90 train→val, seg loss ~2.3-2.4, coral loss ~0.0003-0.16, matching B3's
own epoch-0 scale for the coral loss given the `4d^2` normalizer), no shape/gradient errors.

## Status

Submitted to the cluster via `jobs/c6_dgcnn_da_d.sbatch` (2026-09-30), `--time=04:00:00` matching
every other DGCNN/PointNet++ row's budget (not KPConv's 60h — DGCNN has no compiled-extension
collate cost). Full-trajectory analysis (all 100 epochs, not just the selected checkpoint)
against C1's own DA-0 baseline required before drawing conclusions, per explicit user
instruction — same discipline as C2/C3/C4's own trajectory analyses. Results pending.
