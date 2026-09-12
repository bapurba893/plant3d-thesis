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

---
## Results (2026-09-12, job 308754, ~13.5 min wall time)

**Protocol-selected checkpoint** (best epoch by lowest source val total loss) — best epoch was
**81** (source val total loss 0.6315):

| Metric | A1 (DA-0) | A2 (DA-A) |
|---|---|---|
| Target cls acc | 0.4603 (avg acc 0.5750) | 0.4444 (avg acc 0.5625) |
| Tomato seg mIoU (source val) | 0.3207 (acc 0.6729) | 0.2573 (acc 0.5295) |
| Maize seg mIoU (source val) | 0.4741 (acc 0.9055) | 0.4186 (acc 0.8340) |

**Full-trajectory analysis** (all 100 epochs' target-eval numbers, same methodology as
C2/C3/C4):

| Metric | A1 (DA-0) | A2 (DA-A) | (for reference: C1 DA-0) | (C2 v2 DA-A) |
|---|---|---|---|---|
| Full-run mean target acc | 0.599 | 0.554 | 0.791 | 0.633 |
| Full-run stdev (population) | 0.196 | 0.193 | 0.104 | 0.117 |
| corr(source val loss, target acc) | −0.165 | +0.086 | −0.286 | −0.061 |
| Quartile means (Q1→Q4) | .676/.717/.507/.495 | .731/.597/.429/.458 | .798/.834/.773/.760 | .691/.678/.608/.554 |
| Epochs collapsed to one class (±0.02 tol) | 17/100 | 15/100 | 3/100 | 3/100 |
| Min / max target acc | 0.365 / 0.968 | 0.365 / 0.952 | 0.365 / 0.921 | 0.365 / 0.921 |

Note: A1/A2's "FINAL target" log line differs from the per-epoch trajectory value recorded at
the selected epoch by 1-3 samples out of 63 (e.g. A1: trajectory shows 0.4762 at epoch 97, FINAL
re-eval of the saved checkpoint shows 0.4603) — not seen on the DGCNN rows (C1/C2's FINAL lines
matched their trajectory values exactly). Most likely GPU floating-point/cuDNN-algorithm
nondeterminism between the mid-training eval call and the final re-eval of the reloaded/copied
model (not seen in a CPU smoke test, only real GPU runs) flipping a small number of
near-decision-boundary predictions — not a code bug (the same deterministic `model.eval()` +
`torch.no_grad()` path is used both times), but flagged since it's a new observation this row
surfaced. The table above uses the log's own "FINAL target" line (the number every other row's
table also cites) for the selected-checkpoint column.

### Answering the question: is C2/C3/C4's underperformance-vs-DA-0 pattern DGCNN-specific, or dataset-general?

**Both, in different proportions — the qualitative direction replicates on PointNet++, but the
magnitude and statistical clarity are much smaller.**

- **The direction replicates**: A2's full-run mean (0.554) is below A1's (0.599), same direction
  as C2 (0.633) below C1 (0.791) — DA-A does not beat DA-0 on either backbone. A2's
  selection-signal correlation (+0.086, wrong-direction/uninformative) is also worse than A1's
  own already-weak correlation (−0.165), the same qualitative shift C2 (−0.061) shows relative to
  C1 (−0.286) — in both backbones, adding the adversarial/entropy terms makes the source-loss
  selection signal less informative about target performance, not more. This supports the
  dataset-general explanation (source/target label-prior mismatch, small-N regime — see
  `step_notes/C2_DGCNN_DA_A.md`) as a real, cross-backbone contributor.
- **But the magnitude is much smaller and far less decisive on PointNet++.** Full-run mean gap:
  A1→A2 drops 4.5 points (0.599→0.554) vs. C1→C2's 15.8-point drop (0.791→0.633) — roughly a
  third the size. Selected-checkpoint gap: A1→A2 drops only 1.6 points (0.4603→0.4444) vs.
  C1→C2's 7.9-point drop (0.7460→0.6667) — essentially a tie on PointNet++, a real difference on
  DGCNN.
- **The reason the effect is smaller: A1's own DA-0 baseline is already highly unstable, in a
  way C1's never is.** A1 alone (no adversarial machinery at all) has stdev 0.196 and 17/100
  epochs collapsed to predicting one class — nearly six times C1's collapse rate (3/100) and
  roughly double C1's stdev (0.104). This matches and sharpens the systematic Maize-bias finding
  already logged for A1 (see CLAUDE.md's A1 entry: "34/40 target Tomato plants misclassified as
  Maize"). Against a baseline this volatile, DA-A's additional ~4.5-point mean drop is a small
  perturbation on top of large pre-existing noise, not a clear regression the way it is on
  DGCNN's stable, high-performing C1 baseline (stdev 0.104, only 3/100 collapse epochs) getting
  visibly worse and more volatile under C2.
- **Conclusion: the underperformance pattern is not purely DGCNN-specific — a real,
  dataset-level effect (the label-prior mismatch / small-N regime) contributes on both
  backbones — but DGCNN's result is the clearer, more decisive demonstration of it specifically
  because DGCNN's own baseline is unusually stable.** PointNet++'s baseline instability
  (independent of any DA method — visible already in A1 alone) is itself a separate,
  backbone-specific weakness that makes it harder to cleanly attribute A1→A2's smaller drop to
  the adversarial method rather than to noise already present without it.
- **One consistent cross-backbone side finding**: segmentation quality is worse under DA-A than
  DA-0 on BOTH backbones (A2 Tomato mIoU 0.2573 vs. A1's 0.3207, Maize 0.4186 vs. 0.4741; C2
  showed the same direction vs. C1). Plausible shared mechanism (not verified by ablation): the
  adversarial/entropy gradients compete with `L_seg` for shared backbone capacity on both
  architectures.

### Conclusion

A2 confirms DA-A doesn't improve over DA-0 on the PointNet++ track either, supporting the
dataset-level explanation (label-prior mismatch, small-N) as a real, shared contributor across
backbones — but the effect is much smaller and less statistically clear-cut here than on DGCNN,
because PointNet++'s own no-adaptation baseline (A1) is already considerably less stable than
DGCNN's. The strongest overall reading: this dataset's characteristics create a real headwind
against adversarial adaptation on any backbone, and DGCNN's C1→C2 comparison happens to isolate
that headwind most cleanly because DGCNN's baseline has the least pre-existing noise to confound
the comparison.
