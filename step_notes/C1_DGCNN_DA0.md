# Row C1: DGCNN, No Domain Adaptation (DA-0), All Augmentations

**Retroactively written 2026-09-19.** C1 was the very first row trained in this project, before
the step_notes/ convention (one detailed file per row) was adopted starting with C2 — it was
documented directly in CLAUDE.md's "Done" section instead. This file backfills that gap for
consistency with every other row (A1-A5, B1-B5+B3b, C2-C5 all have their own step_notes file);
content is reconstructed from CLAUDE.md's existing C1 entry and the run log
(`results/C1_dgcnn_da0_clsseg/run.log`, job 308239), not re-derived.

## What this is, in plain language

C1 is Block C's (DGCNN) simplest row: train DGCNN with **no domain adaptation at all** — just
train normally on the labeled source dataset (Crops3D) and see how well it generalizes, with
zero special help, to the unlabeled target dataset (Pheno4D). This is the "lower bound" every
other DGCNN row in the table gets compared against, and — since DGCNN is the field-standard
encoder in point-cloud domain-adaptation literature (PointDA-10) — it also reproduces a
published-style baseline.

The model does two things at once: (1) classify whether a plant is tomato or maize, and (2)
label each individual point as belonging to a specific organ ("segmentation").

## Design decisions

1. Backbone from `PointDA.Models.DGCNN` (DefRec_and_PCM, unmodified upstream), composed with
   per-species segmentation heads shaped after `PointSegDA.Models.segmentation` — both composed
   rather than reimplemented, per CLAUDE.md's "adapt this repo, don't reimplement" instruction.
   `adapters/models.py::DGCNN_ClsSeg`.
2. Segmentation is per-species: Tomato num_classes=3, Maize num_classes=6, following the
   per-point organ label backfill from Crops3D's `scalar_sf` PLY field (see CLAUDE.md's
   "Segmentation labels gap" entry — `Crops3D_IS` was initially the wrong source, corrected
   before this row trained for real).
3. `L_cls` uses inverse-class-frequency weighting (`w_c=(f_c+eps)^-1`, normalized) — plain
   cross-entropy in an earlier interim attempt let the model settle into a majority-class
   shortcut on Crops3D's 2.7:1 Maize:Tomato imbalance.
4. `L_seg = L_wCE + λ_lov·L_Lovász` (λ_lov=1.0, fixed per the strategy table docx).
5. `L_cls` and `L_seg` combined via Kendall uncertainty weighting (both classification-type per
   the docx's Level-2 formula, `exp(-s_j)·L_j + s_j/2`) — this project's first use of that
   mechanism.
6. Model selection: lowest combined source (Crops3D) validation total loss — never touches
   target labels, the DA-0 protocol every other DA-0 row follows.

## What was done and how

1. Submitted `jobs/c1_dgcnn_da0_clsseg.sbatch` (job **308239**) on 2026-09-11: 100 epochs,
   batch_size=32, lr=1e-3, wd=5e-5, dropout=0.5.
2. Data: 263 Crops3D train (71 Tomato, 192 Maize), 45 Crops3D val (12 Tomato, 33 Maize), 63
   held-out Pheno4D target plants (40 Tomato, 23 Maize) — evaluated each epoch as a hands-off
   domain-gap indicator only, never trained on or used for selection.
3. This run supersedes an earlier classification-only interim script (no `L_seg`) that had
   selected a different, earlier-epoch checkpoint.

## Results

**Protocol-selected checkpoint** (epoch 95, lowest source val total loss 0.6866): source val cls
acc 1.0000, target (Pheno4D) cls acc **0.7460** (avg acc 0.7446), Tomato seg mIoU 0.3248
(acc 0.6645), Maize seg mIoU 0.3427 (acc 0.7698).

**Target cls accuracy dropped vs. the earlier interim (classification-only) script's result
(0.8889 → 0.7460) — investigated, and it is NOT evidence that adding `L_seg` hurts target
transfer.** Epoch 6 of this same run (the epoch the old interim script happened to land on)
shows target acc 0.9206 — matching or beating the old result. What actually happens: target
accuracy drifts steadily downward over the full 100 epochs (settling frozen at exactly 0.7460 for
the last ~10 epochs) while source val cls stays saturated at 1.0 and seg loss keeps improving the
whole time. So under DA-0, continued training keeps helping the only signals model-selection is
allowed to see (source loss), while target generalization quietly erodes in the background with
nothing to detect or prevent it — a real, expected DA-0 characteristic (this is the drift pattern
that motivates C2's DANN anchor row), not a training bug, and not a reason to change the
selection criterion (selecting on target labels would defeat the point of DA-0 as a lower bound).

**Full-trajectory statistics** (computed later, alongside C2-C5's analyses, for consistency —
see CLAUDE.md's cross-row tables): full-run mean target acc 0.791, full-run stdev 0.104,
corr(selection-loss, target acc) −0.286, quartile means rising-then-falling
(0.798/0.834/0.773/0.760), 3/100 collapse epochs. This makes C1 the most stable of the three
backbones' own DA-0 baselines (vs. A1's 0.196 stdev/17 collapse epochs, B1's 0.203 stdev/wrong-
signed correlation) — see A1/B1's own step_notes for the cross-backbone comparison at the time
each was trained.

## Conclusion

C1 established DGCNN's DA-0 baseline (0.791 full-run mean / 0.7460 selected) as the strongest of
the three backbones' baselines — a bar that, per the project's eventual FINAL SUMMARY (see
CLAUDE.md), no adaptation method ever beat on DGCNN specifically (C2/C3/C4 all landed below C1),
unlike PointNet++ and KPConv where DA-S went on to clearly beat their own DA-0 baselines. C1 also
reproduces a published-style baseline (DGCNN being the PointDA-10 field-standard encoder),
giving this project a literature-comparable anchor point the other two backbones don't have.
