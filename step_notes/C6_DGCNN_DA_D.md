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

## Status: DONE (job 323673, 2026-09-30, 16m37s)

Best epoch 98 (source val total loss). Selected-checkpoint target test accuracy 0.8571 (avg acc
0.8690) — a real jump over C1's own 0.7460.

**Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

| Metric | C1 (DA-0) | C6 (DA-D) |
|---|---|---|
| Full-run mean | 0.791 | **0.816 (+0.025)** |
| Full-run stdev | 0.104 | 0.091 |
| corr(selection-loss, target acc) | -0.286 | -0.164 |
| Collapse epochs (avg_acc~0.5, tol 0.02) | 3/100 | 2/100 |
| Selected-checkpoint acc | 0.7460 | **0.8571 (+0.111)** |

**Answering the motivating question for DGCNN specifically: DA-D is the FIRST adaptation method
that does not hurt this backbone.** DA-A (C2) cost -15.8 pts full-run mean; DA-S (C3) cost -8.8
pts; DA-D is the first of the three methods tried on DGCNN to land on the positive side, though
modestly (+0.025 full-run mean — small next to KPConv's own DA-D gain, see the cross-backbone
verdict in step_notes/A6_PointNet2_DA_D.md's counterpart section, duplicated below). Stability
also improves slightly (stdev down, one fewer collapse epoch), and the selection-signal
correlation, while still short of a fully reliable signal, moves toward zero rather than away
from it (unlike DA-A/DA-S's outright reversals on this backbone in some readings).

## Cross-backbone DA-D verdict (all three backbones now tested, 2026-09-30)

| Backbone | DA-0 | DA-D | Δ full-run mean | Δ selected checkpoint | Δ stdev | Δ collapse epochs |
|---|---|---|---|---|---|---|
| A (PointNet++) | 0.599 | 0.649 | +0.050 | 0.4603→0.5714 (+0.111) | 0.196→0.166 | 17→4 |
| B (KPConv) | 0.629 | 0.808 | **+0.179** | 0.4444→0.6825 (+0.238) | 0.203→0.094 | 3→0 |
| C (DGCNN) | 0.791 | 0.816 | +0.025 | 0.7460→0.8571 (+0.111) | 0.104→0.091 | 3→2 |

**Direct answer to the motivating question: the DIRECTION generalizes cleanly across all three
backbones — DA-D helps every single one, on every axis measured (full-run mean, selected
checkpoint, stability, collapse epochs) — but the MAGNITUDE stays strongly backbone-dependent,
with KPConv's gain 3.5-7x larger than either other backbone's.** This is a genuinely different,
and cleaner, cross-backbone story than either DA-A (never clearly helps any backbone, direction
consistent-in-failure) or DA-S (helps A/B, actively hurts C — direction itself flips). DA-D is
the first and only method in this project where the qualitative direction (helps vs. hurts)
generalizes across ALL THREE backbones without exception. This strengthens, rather than merely
repeats, CLAUDE.md's existing "cooperative losses help, adversarial doesn't" reading — that
reading was built from KPConv alone (B2 vs. B3/B3b); it now holds up as a genuine cross-backbone
pattern for DA-D specifically, not just a KPConv-specific artifact. It does NOT mean the
magnitude is architecture-independent — KPConv remains the standout, and why KPConv's gain is so
much larger than DGCNN's/PointNet++'s own DA-D gains is not resolved by this data alone (both
backbones still show real, positive, non-trivial improvement, just far more modest than KPConv's).
