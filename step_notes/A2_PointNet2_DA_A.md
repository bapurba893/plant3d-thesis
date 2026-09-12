# Row A2: PointNet++, Adversarial Domain Adaptation (DA-A), All Augmentations

## What this is, in plain language

A2 is the DGCNN-track's C2 row, ported to the PointNet++ backbone: same adversarial
domain-adaptation method (DANN + Gradient Reversal Layer + entropy minimization on unlabeled
target predictions), same augmentation mix, same source/target data and protocol — only the
backbone architecture changes. Per CLAUDE.md, DA-A "appears identically across all 3 backbones
so cross-backbone differences are attributable to architecture, not method" — this row is the
first real test of that claim.

**Why now, specifically:** the DGCNN track (C2/C3/C4) found that neither adversarial (C2) nor
self-supervised (C3) adaptation beats the plain no-adaptation baseline (C1) here, and that
narrowing the augmentation mix to noise-only (C4) doesn't change that picture either (see
`step_notes/C4_DGCNN_DA_A_LN.md`'s conclusion). Per the user's framing: does that pattern reflect
something about DGCNN specifically, or is it a general property of this dataset (source/target
label-prior mismatch — Crops3D 73% Maize vs. Pheno4D 62-63% Tomato — plus the small-N training
regime, both root-caused during C2's investigation)? A2 is a direct test: identical adversarial
method (`adapters/dann.py`, reused byte-for-byte, already carrying the post-C2 ramped-`λ_ent`
fix), different backbone. If A2 shows the same qualitative failure shape as C2 (underperforms
its own DA-0 baseline, A1, with declining quartile means and an uninformative source-loss
selection signal), that's evidence the pattern is dataset-general, not DGCNN-specific.

## Status: submitted to the cluster (2026-09-12)

## Design decisions

1. **`adapters/dann.py` reused completely unchanged** — it's already backbone-agnostic by
   design (see its own docstring: built this way specifically so A2/A4/B2/C4 could reuse it
   byte-for-byte). No edits needed for this row.
2. **`PointNet2_ClsSeg.forward` needed one additive change**: exposing `logits["feat"]` (the
   pooled 1024-dim global feature, pre-classifier) as the domain discriminator's input — the
   same key `DGCNN_ClsSeg.forward` already exposes for C2/C4. Added in
   `adapters/models_pointnet2.py`; purely additive (new dict key), so A1's existing callers
   (which only read `logits["cls"]`/`["seg_feat"]`) are unaffected.
3. **`adapters/train_a2_pointnet2_da_a.py` is a near-line-for-line copy of
   `train_c2_dgcnn_da_a.py`**, with `DGCNN_ClsSeg`/`model_args.model="dgcnn"` swapped for
   `PointNet2_ClsSeg`/plain `model_args` (matching how `train_a1_pointnet2_da0.py` already
   swapped C1's backbone the same way) — same data, same loss architecture, same DANN
   hyperparameters (`gamma=10.0`, `lambda_ent=0.1` max, `disc_dropout=0.3`), same model-selection
   protocol (lowest source val total loss, never touching target labels).
4. **Comparison baseline is A1** (`results/A1_pointnet2_da0_clsseg/run.log`), the same way C2
   compares against C1 — both A1 and A2 evaluate on the identical `pheno4d_heldout_eval.csv`
   target test set/protocol, so the A1-vs-A2 gap is directly comparable in kind to the
   C1-vs-C2 gap.
5. **Full-trajectory analysis from the start**, per explicit user instruction — not just the
   protocol-selected checkpoint. Same methodology as C2/C3/C4 (full-run mean/stdev of per-epoch
   target held-out accuracy, quartile means, correlation between source val loss and target
   acc, count of epochs collapsed to predicting one class).

## What was done and how

1. Added `logits["feat"]` to `adapters/models_pointnet2.py::PointNet2_ClsSeg.forward` (see
   design decision 2).
2. Wrote `adapters/train_a2_pointnet2_da_a.py` (see design decision 3).
3. Added `jobs/a2_pointnet2_da_a.sbatch` — same cluster settings as A1/C2 (dgx partition/qos, 1
   GPU, 4h wall time), with both `DEFREC_ROOT` and `POINTNET2_ROOT` exported (A1's job needed
   both; C2's needed only `DEFREC_ROOT` since DGCNN comes from that repo).
4. **CPU smoke test** (1 epoch, batch_size=8, real data, `--gpu -1`): ran end-to-end in ~2.5
   minutes with no shape/gradient errors. Data counts matched A1/C2 exactly (Crops3D train 263,
   val 45; Pheno4D adaptation pool 160, held-out eval 63), `logits["feat"]` wired correctly (no
   shape errors from `DomainDiscriminator(in_dim=1024)` against PointNet2's pooled feature), all
   losses non-zero and in sane ranges (`dom loss` near `ln(2)≈0.69` as expected early, `lambda_p`
   hit 0.9999 by the end of the single epoch — expected smoke-test artifact since
   `total_steps = epochs*n_batches = 1*32`, same documented behavior as C2's own smoke test, not
   a bug). One epoch predicting majority-class-only (target acc 0.365) is expected, not
   concerning. Deleted `results/_smoketest_a2/` afterward.
5. Added `jobs/a2_pointnet2_da_a.sbatch`.

---
**2026-09-12:** Submitted `jobs/a2_pointnet2_da_a.sbatch` to the `dgx` partition — job **308754**.
Queue was empty beforehand. Awaiting completion.
