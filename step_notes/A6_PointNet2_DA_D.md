# A6: PointNet++, DA-D (Deep CORAL) — added row

Not in CLAUDE.md's original strategy table (Block A is A1-A5: DA-0/DA-A/DA-S/DA-A-LN/DA-O — no
DA-D slot). Added per explicit user instruction, alongside its DGCNN counterpart (C6), same
category of addition as B3b was to Block B. Numbered A6 (next available slot).

## Motivating question

Shared with C6 (see that file for the full framing): does DA-D's large KPConv win (B3, +17.9
pts) generalize across backbones, or was it KPConv-specific? PointNet++'s own story differs from
DGCNN's in a relevant way: A2 (DA-A) hurts mildly (and turns out to be partly an
augmentation-mix artifact — A4 nearly closes the gap just by narrowing to L-N), while A3 (DA-S)
is this backbone's BEST non-Oracle result (+20.1 pts over A1) — so PointNet++ has already shown
one cooperative-loss win. A6 tests whether a second, mechanistically distinct cooperative method
also helps, which would strengthen "cooperative losses help this backbone" as a pattern rather
than something specific to DefRec's reconstruction mechanism.

## Implementation

`adapters/train_a6_pointnet2_da_d.py`: byte-for-byte the same plumbing as
`train_a2_pointnet2_da_a.py` (`PlantClsSegDataset`/`PlantSpeciesDataset`, `PointNet2_ClsSeg`,
same split) with C6's/B3's DA-D loss machinery substituted for A2's DANN machinery — identical
pattern to how C6 was built, itself mirroring how A2 originally ported C2's DANN machinery onto
A1's plumbing. `coral_loss` unchanged (already backbone-agnostic).

CPU smoke test (1 epoch, real data) passed cleanly before submitting — sane non-degenerate
numbers, no shape/gradient errors, coral loss values in the same tiny-but-nonzero range as C6's
and B3's own epoch-0 readings.

## Status: DONE (job 323672, 2026-09-30, 13m50s)

Best epoch 97 (source val total loss). Selected-checkpoint target test accuracy 0.5714 (avg acc
0.6625) — a real improvement over A1's own 0.4603.

**Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

| Metric | A1 (DA-0) | A6 (DA-D) |
|---|---|---|
| Full-run mean | 0.599 | **0.649 (+0.050)** |
| Full-run stdev | 0.196 | 0.166 |
| corr(selection-loss, target acc) | -0.165 | +0.024 |
| Collapse epochs (avg_acc~0.5, tol 0.02) | 17/100 | **4/100** |
| Selected-checkpoint acc | 0.4603 | **0.5714 (+0.111)** |

**Answering the motivating question for PointNet++ specifically: DA-D is a second cooperative-
loss win on this backbone, alongside DA-S (A3, +0.201 full-run mean) — both cooperative methods
now help PointNet++, while the one adversarial method tried (A2) hurts (mildly, and partly an
augmentation-mix artifact per A4).** DA-D's gain here (+0.050) is real but far more modest than
DA-S's own (+0.201) on the same backbone — cooperative losses aren't uniformly as strong as each
other even on a backbone that responds well to the category as a whole. Collapse-epoch count
drops sharply (17→4, matching the kind of stabilizing effect A4's augmentation-narrowing also
produced on this backbone's otherwise-noisy baseline) even though the selection-signal
correlation itself doesn't improve (stays near zero/wrong-signed).

## Cross-backbone DA-D verdict (all three backbones now tested, 2026-09-30)

See step_notes/C6_DGCNN_DA_D.md for the full table and discussion (duplicated here for
zero-context-reader convenience):

| Backbone | DA-0 | DA-D | Δ full-run mean | Δ selected checkpoint | Δ stdev | Δ collapse epochs |
|---|---|---|---|---|---|---|
| A (PointNet++) | 0.599 | 0.649 | +0.050 | 0.4603→0.5714 (+0.111) | 0.196→0.166 | 17→4 |
| B (KPConv) | 0.629 | 0.808 | **+0.179** | 0.4444→0.6825 (+0.238) | 0.203→0.094 | 3→0 |
| C (DGCNN) | 0.791 | 0.816 | +0.025 | 0.7460→0.8571 (+0.111) | 0.104→0.091 | 3→2 |

**Direct answer: the direction (DA-D helps) generalizes cleanly across all three backbones — the
first adaptation method in this project where that's true without exception — but the magnitude
stays strongly backbone-dependent, KPConv's gain 3.5-7x the other two.**
