# Row A1: PointNet++, No Domain Adaptation (DA-0), All Augmentations

## What this is, in plain language

This project's full experiment plan is a 24-row table testing combinations of 3 neural
network architectures ("backbones") × 5 domain-adaptation strategies (methods for helping a
model trained on one dataset generalize to a different, unlabeled dataset it's never seen
labels for). Row **A1** is the simplest row in Block A (the PointNet++ block): train
PointNet++ with **no domain adaptation at all** — just train it normally on the labeled
source dataset (Crops3D) and see how well it happens to generalize to the unlabeled target
dataset (Pheno4D) with zero special help. This is the "lower bound" every other PointNet++
row in the table gets compared against.

The model does two things at once: (1) classify whether a plant is a tomato or a maize plant,
and (2) label each individual point in the plant as belonging to a specific organ (leaf, stem,
fruit, etc.) — "segmentation."

**Why it matters:** A1 is the first-ever real training run on the PointNet++ backbone (see
`step_notes/PointNet2_Integration.md` for how that backbone was integrated) and the first time
this project could directly compare two different architecture families — PointNet++ vs.
DGCNN — under the exact same conditions.

## What was done and how

1. Submitted `jobs/a1_pointnet2_da0.sbatch` to the cluster's GPU queue (job **308286**) on
   2026-09-11, running `adapters/train_a1_pointnet2_da0.py` for 100 epochs, batch size 32,
   learning rate 1e-3, weight decay 5e-5, dropout 0.5 — the same training configuration
   already used for the DGCNN comparison row, C1.
2. Training used the same data split as C1: 263 Crops3D training plants (71 Tomato, 192
   Maize), 45 Crops3D validation plants (12 Tomato, 33 Maize), and 63 held-out Pheno4D
   (target) plants (40 Tomato, 23 Maize) used **only** for a hands-off evaluation each epoch —
   never trained on, matching the DA-0 protocol.
3. Combined loss: classification loss (`L_cls`, inverse-class-frequency weighted, same fix as
   C1 needed) + segmentation loss (`L_seg`, per-species — Tomato has 3 organ classes, Maize
   has 6), combined with the project's learned (Kendall) uncertainty weighting between the two
   — same loss setup as C1, only the backbone differs.
4. The next session confirmed the job had finished (no longer appearing in `squeue`), read the
   full training log (`results/A1_pointnet2_da0_clsseg/run.log`), and extracted the final
   numbers.

## Technical specifics

- **Files:** `adapters/train_a1_pointnet2_da0.py`, `adapters/models_pointnet2.py`,
  `jobs/a1_pointnet2_da0.sbatch`, log at `results/A1_pointnet2_da0_clsseg/run.log`, checkpoint
  at `results/A1_pointnet2_da0_clsseg/model.pt` (gitignored, matches `results/**/*.pt` rule)
- **Command:** `sbatch jobs/a1_pointnet2_da0.sbatch` on the `dgx` partition, A100 80GB GPU
- **Model selection:** best checkpoint chosen by lowest combined source validation loss
  (never touches target labels — correct for a genuine DA-0 lower-bound baseline), landed at
  **epoch 97**
- **Commit:** `1320d42` — "Row A1 (PointNet++, DA-0, ALL) trained: first cross-backbone
  comparison vs C1"

## Results

**A1 final numbers** (best model, epoch 97):

| Metric | Value |
|---|---|
| Source val cls acc | 1.0000 (avg acc 1.0000) |
| Target (Pheno4D) cls acc | 0.4603 (avg acc 0.5750) |
| Tomato seg mIoU | 0.3207 (acc 0.6729) |
| Maize seg mIoU | 0.4741 (acc 0.9055) |

**Comparison against C1 (DGCNN, same DA-0/ALL setup, same data, same epoch count):**

| Metric | C1 (DGCNN) | A1 (PointNet++) |
|---|---|---|
| Source val cls acc | 1.0000 | 1.0000 |
| Target cls acc | 0.7460 (avg 0.7446) | 0.4603 (avg 0.5750) |
| Tomato seg mIoU | 0.3248 (acc 0.6645) | 0.3207 (acc 0.6729) |
| Maize seg mIoU | 0.3427 (acc 0.7698) | 0.4741 (acc 0.9055) |

**Findings:**

- Both backbones fit the labeled source data perfectly (1.0000 accuracy) — expected, and not
  the interesting part of the comparison.
- **PointNet++ transfers substantially worse to the unseen target dataset for whole-plant
  classification** (0.46 vs. 0.75 for DGCNN). This isn't just noisier — the confusion matrix
  shows a systematic bias: 34 of 40 target Tomato plants were misclassified as Maize.
- **PointNet++ segments Maize plants distinctly better** than DGCNN (0.47 vs. 0.34 mIoU,
  0.91 vs. 0.77 accuracy). Tomato segmentation is essentially tied between the two backbones
  (0.32 mIoU both).
- **Same "DA-0 drift" pattern seen in C1 showed up again here.** Early in training (around
  epoch 8–9), target classification accuracy briefly peaked at 0.92–0.94 — matching or beating
  DGCNN's best — then eroded steadily over the remaining ~90 epochs while source-side metrics
  kept looking better and better, settling at the final 0.46 by epoch 97. Since nothing in
  DA-0 training is allowed to look at target labels, nothing can detect or stop this drift.
  Seeing the identical pattern on a second, architecturally unrelated backbone is stronger
  evidence that this is a general property of "no adaptation" training itself, not a
  DGCNN-specific quirk — which strengthens the motivation for the adversarial anchor rows
  (A2, C2) that come next.

**Pass/fail:** training completed successfully with no errors; the row is done and the result
is trustworthy (correct model-selection protocol, matching data split to C1). The *low* target
accuracy is not a bug — it's the expected behavior of a no-adaptation baseline, and is itself
the useful finding.

## What's next

Row A1 is done. Per the project's row ordering, the next row is **C2** (DGCNN, adversarial
domain adaptation) — see `step_notes/C2_DGCNN_DA_A.md`. Later, once the adversarial anchor
exists for the DGCNN side, the same treatment (A2) will apply to PointNet++, and this A1 result
will be the baseline it's compared against.

---
**2026-09-11 (evening):** Job 308286 confirmed finished; final numbers extracted from
`results/A1_pointnet2_da0_clsseg/run.log`, compared against C1, and committed (`1320d42`)
alongside `PROGRESS_LOG.md`/`CLAUDE.md` updates.
