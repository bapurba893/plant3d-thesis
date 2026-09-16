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

## Status: submitted to the cluster (2026-09-16), job 311737 on `cn18-dgx`

**CPU smoke test (1 epoch, batch_size=8, real data, `--gpu -1`) — passed cleanly, full epoch
completed.** Launched detached (`nohup ... &`, `disown`), per the standing precaution adopted
after B3's smoke test was lost to a session-teardown issue. Dataset sizes matched A2's exactly
(263 Crops3D train, 45 Crops3D val, 160 Pheno4D adaptation pool, 63 Pheno4D held-out eval; 32
source batches/epoch, 20 target-adapt batches cycled). All 32 training batches ran with no
shape/gradient errors; end-of-epoch summary printed sane, non-degenerate numbers (val cls acc
0.7333, target eval acc 0.3651 — expected to be noisy/uninformative after just 1 epoch). Deleted
`results/_smoketest_a4/` after confirming the pass.
