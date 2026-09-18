# Row A4: PointNet++, DA-A (Adversarial), L-N Augmentation Only

## What this is, in plain language

A4 ports the adversarial anchor method (DANN + entropy minimization) to PointNet++ with
augmentation narrowed to Gaussian jitter only (L-N), instead of the full ALL mix A2 used. It is
the PointNet++-track counterpart of C4 (DGCNN, DA-A, L-N) — same augmentation isolation, same
adversarial machinery, different backbone.

## Why this row, specifically, now

Per explicit user instruction. A2 already showed adversarial adaptation underperforms A1's DA-0
baseline on PointNet++, but by a much smaller margin than DGCNN's C1→C2 gap (A1→A2: −4.5 pts
full-run mean; C1→C2: −15.8 pts) — attributed partly to A1's own baseline already being unstable
(17/100 collapse epochs) rather than purely to the adversarial method. C4 already checked, on
DGCNN, whether that underperformance is sensitive to the augmentation mix: it isn't (C2 ALL vs.
C4 L-N-only landed within ~2 points of each other, both well below C1). A4 asks the same question
on PointNet++: does narrowing to L-N change A2's picture, or replicate C4's "no meaningful
difference" finding on a second backbone?

**Note on naming vs. CLAUDE.md's strategy table**: CLAUDE.md's Block A row list actually specifies
A4 as DA-A/L-D (isolating PointNet++'s known density weakness via RandomCrop3D/CoarseDropout3D,
not jitter). Per explicit user instruction for this specific run, A4 is instead run as DA-A/L-N
(noise-only), mirroring C4's augmentation isolation exactly so A4-vs-C4 is a same-shaped
cross-backbone comparison, the same relationship A2-vs-C2 already has (identical DA-A/ALL cell).
This is flagged here explicitly as a deviation from the table's literal row spec, not a silent
substitution — if a later row still needs to specifically isolate density (L-D) on PointNet++,
that remains open and unaddressed by this row.

## Design decisions

1. **Byte-for-byte copy of `train_a2_pointnet2_da_a.py` with exactly one variable changed**:
   `augment_mode="ln_only"` on both `src_train_set` (`PlantClsSegDataset`) and `trgt_adapt_set`
   (`PlantSpeciesDataset`) — both classes already support this constructor arg (added for C4),
   no dataset-layer changes needed. Mirrors the exact C2→C4 diff pattern (confirmed by diffing
   those two files before writing this one).
2. **`dann.py` reused completely unmodified** — no new adversarial-machinery changes, so any
   A2-vs-A4 difference is attributable purely to the augmentation mix.
3. **`--time=04:00:00`**, PointNet++'s standard budget (NOT KPConv's 60h convention — CLAUDE.md's
   `--time=60:00:00` note is explicitly scoped to Block B only).
4. Model selection protocol identical to A1/A2/C1-C4: lowest source (Crops3D) val total loss
   (Kendall(cls,seg)), never touching target labels or L_dom/L_ent.
5. Given the same full-trajectory analysis as every prior row per standing user instruction, once
   results land.

## What was done and how

1. Diffed `train_c2_dgcnn_da_a.py` vs. `train_c4_dgcnn_da_a_ln.py` to confirm the exact,
   minimal change pattern for isolating L-N (one dataset constructor arg on two datasets, plus
   docstring/log-line updates — no training-loop logic changes).
2. Confirmed `PlantClsSegDataset`/`PlantSpeciesDataset` both already accept `augment_mode` (added
   for C4, reused unchanged here).
3. Wrote `adapters/train_a4_pointnet2_da_a_ln.py` as a near-line-for-line copy of
   `train_a2_pointnet2_da_a.py` with `AUGMENT_MODE = "ln_only"` applied to the two training-time
   datasets (see design decisions).
4. Added `jobs/a4_pointnet2_da_a_ln.sbatch` (`--time=04:00:00`, standard non-KPConv budget).

## Status: finished (2026-09-16), job 311737 on `cn18-dgx`, 52 minutes, no contention

Best model at epoch 46 (lowest source val total loss, 0.6892).

## Results

**Protocol-selected checkpoint** (epoch 46, lowest source val total loss):

- Source val: cls acc 1.0000 (avg acc 1.0000), seg loss 0.8774, Tomato seg mIoU 0.3743
  (acc 0.7107), Maize seg mIoU 0.4536 (acc 0.8885).
- **Target (Pheno4D held-out) test: acc 0.5397 (avg acc 0.6375), loss 2.4119** — directly
  comparable to A1/A2's same file/metric.
- Confusion matrix (rows=true, cols=pred, order Tomato/Maize): `[[11,29],[0,23]]` — the
  systematic Tomato→Maize misclassification bias seen in A1/A2 is still present (Tomato recall
  0.275) but less extreme than A2's own confusion pattern (see below).

**Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

| Metric | A1 (DA-0) | A2 (DA-A, ALL) | **A4 (DA-A, L-N only)** | C2 v2 (DA-A, ALL, ref) | C4 (DA-A, L-N only, ref) |
|---|---|---|---|---|---|
| Full-run mean | 0.599 | 0.554 | **0.586** | 0.633 | 0.613 |
| Full-run stdev | 0.196 | 0.193 | **0.170** | 0.117 | 0.138 |
| corr(selection-loss, target acc) | −0.165 | +0.086 | **−0.167** | −0.061 | +0.041 |
| Collapse epochs (±0.02 tol) | 17/100 | 15/100 | **4/100** | 3/100 | 4/100 |
| Quartile means (Q1→Q4) | .676/.717/.507/.495 | .731/.597/.429/.458 | **.773/.627/.483/.462** | .798/.834/.773/.760 | .646/.664/.582/.559 |
| Selected-checkpoint acc | 0.4603 | 0.4444 | **0.5397 (+0.095 vs A2)** | 0.6667 | 0.5556 |

## Directly answering the motivating question

**Narrowing DA-A's augmentation to L-N-only changes PointNet++'s picture substantially — unlike
DGCNN, where the same narrowing (C2→C4) made essentially no difference (both underperformed DA-0
by a similar margin, ~2 point gap between the two DA-A variants).** For PointNet++, A2→A4 shows a
real, multi-axis improvement:
- Full-run mean rises +3.2 points (0.554→0.586), nearly closing the gap to A1's own DA-0 baseline
  (0.599 — only 0.013 below, essentially a tie, vs. A2's larger 0.045-point gap below A1).
- Collapse-epoch count drops nearly 4x (15/100→4/100) — A4 is actually *more* stable by this
  measure than A1's own DA-0 baseline (17/100), the first DA-A row on either backbone to beat its
  own DA-0 baseline's collapse rate.
- **The selection-signal correlation flips from wrong-signed to correctly-signed**: A2's
  `corr(selection-loss, target acc) = +0.086` (worse than useless — a better source-val loss
  predicted worse target accuracy) becomes A4's `−0.167`, nearly identical to A1's own `−0.165`.
  This is the same kind of correlation-restoring effect seen when B5 removed the domain gap
  entirely via Oracle training — here, simply removing the point-removing/geometry-distorting
  augmentations (G-R/G-S/L-D) from the mix restores a informative, DA-0-like selection signal
  without removing the domain gap at all.
- Selected-checkpoint accuracy improves +9.5 points (0.4444→0.5397).

**This is evidence that PointNet++'s DA-A instability (documented in A2) is at least partly an
augmentation-interaction effect, not purely the same dataset-level label-prior-mismatch/small-N
story used to explain DGCNN's C2/C4 (which showed no such augmentation sensitivity).** A
plausible mechanism, flagged as an inference not verified by ablation: PointNet++'s ball-query
set abstraction is directly density-sensitive (per CLAUDE.md's Backbones section), so L-D's
RandomCrop3D/CoarseDropout3D — which directly perturbs local point density/coverage — may
interact especially badly with an already-density-sensitive backbone when combined with
adversarial training's own instability, in a way that doesn't affect DGCNN's k-NN-based (density-
insensitive by construction) graph convolution the same way. This is consistent with, though not
proof of, CLAUDE.md's stated PointNet++ weakness.

**One consistent finding shared with C4's own DGCNN result**: segmentation quality improves under
L-N vs. ALL for the same likely reason (removing L-D's point-removing augmentations makes
segmentation strictly easier) — A4's Tomato seg mIoU (0.3743) actually *exceeds* A1's own DA-0
baseline (0.3207), and its Maize mIoU (0.4536) sits much closer to A1 (0.4741) than A2's degraded
0.4186. Unlike A3's DA-S win, this segmentation improvement is NOT a tradeoff against
classification transfer — both classification and segmentation improve together under A4 vs. A2.

## Conclusion

Narrowing DA-A's augmentation mix to L-N-only is a genuinely different, more informative test on
PointNet++ than it was on DGCNN: for DGCNN it confirmed the ALL-vs-underperformance pattern was
not an augmentation artifact (C2≈C4, both below C1). For PointNet++, it reveals the opposite —
much of A2's instability and wrong-signed selection correlation is attributable to the augmentation
mix itself, not solely to the dataset-level explanations (label-prior mismatch, small-N) used for
DGCNN. A4 nearly closes the DA-0-vs-DA-A gap on this backbone while simultaneously fixing the
selection-signal problem and improving segmentation — the strongest evidence yet that PointNet++'s
adversarial-adaptation story is genuinely architecture-and-augmentation-dependent, not just a
smaller/noisier echo of DGCNN's. Full trajectory tables above; see `step_notes/A2_PointNet2_DA_A.md`
and `step_notes/C4_DGCNN_DA_A_LN.md` for the rows this compares against.

**CPU smoke test (1 epoch, batch_size=8, real data, `--gpu -1`) — passed cleanly, full epoch
completed.** Launched detached (`nohup ... &`, `disown`), per the standing precaution adopted
after B3's smoke test was lost to a session-teardown issue. Dataset sizes matched A2's exactly
(263 Crops3D train, 45 Crops3D val, 160 Pheno4D adaptation pool, 63 Pheno4D held-out eval; 32
source batches/epoch, 20 target-adapt batches cycled). All 32 training batches ran with no
shape/gradient errors; end-of-epoch summary printed sane, non-degenerate numbers (val cls acc
0.7333, target eval acc 0.3651 — expected to be noisy/uninformative after just 1 epoch). Deleted
`results/_smoketest_a4/` after confirming the pass.
