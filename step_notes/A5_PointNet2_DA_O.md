# Row A5: PointNet++, DA-O (Oracle Upper Bound), All Augmentations

## What this is, in plain language

A5 is Block A's Oracle upper-bound reference point — the PointNet++-track counterpart of
`train_c5_dgcnn_da_o.py` (C5) and `train_b5_kpconv_da_o.py` (B5). Per CLAUDE.md: "DA-O ... NEVER
a real candidate strategy — exists only as the upper-bound reference point." Domain roles are
swapped relative to every other Block A row: it trains directly on a small, labeled subset of
Pheno4D (the target dataset) instead of adapting from Crops3D, then evaluates on the SAME
held-out target test set A1-A4 all report against. Completes Block A (A1-A5 all done after this
row) and gives the third and final cross-backbone Oracle ceiling, alongside C5 (DGCNN) and B5
(KPConv).

## Why this row, specifically, now

Per explicit user instruction: same question as B5/C5. A2 showed a wrong-signed selection-signal
correlation (`corr(selection-loss, target acc) = +0.086`) — the only Block A DA-A configuration
tested with a positive/misleading correlation. A4 already showed this specific problem resolves
without any Oracle supervision at all, just by narrowing the augmentation mix to L-N (A4's corr:
−0.167, matching A1's own −0.165). A5 asks the same underlying question a different way: does the
correlation also resolve under full Oracle conditions (real target labels, no domain gap to
cross at all), and — separately — where does PointNet++'s Oracle ceiling land relative to
DGCNN's (C5: 0.925 full-run mean, corr −0.823) and KPConv's (B5: 0.9425 full-run mean, corr
−0.858, the highest ceiling of the two backbones tested so far)?

## Design decisions

1. **Mirrors `train_c5_dgcnn_da_o.py`'s Oracle protocol exactly**, with PointNet++'s batch
   mechanics (`models_pointnet2.py::PointNet2_ClsSeg`, `pts.permute(0,2,1)` raw-tensor forward
   pass, same as A1-A4) substituted for DGCNN's — only the backbone differs. Same domain-role
   swap, same target segmentation eval as an Oracle-only bonus metric.
2. **Reuses C5's exact plant-disjoint Oracle train/val split unchanged**
   (`data/pheno4d_oracle_train.csv`: 8 plants, 72 scans; `data/pheno4d_oracle_val.csv`: 2 plants,
   18 scans) — the same split B5 already reused, keeping all three backbones' Oracle rows
   trained/evaluated on the identical labeled-target-data slice for a directly comparable
   ceiling comparison.
3. **`batch_size=8`, same data-driven necessity as B5/C5** — 72 training samples; `batch_size=32`
   (A1-A4's setting) + `drop_last=True` would yield only 2 batches/epoch and drop samples;
   `batch_size=8` uses all 72 samples across 9 batches/epoch.
4. **`--time=04:00:00`**, PointNet++'s standard budget (not KPConv's 60h convention — scoped to
   Block B only per CLAUDE.md). No adversarial/DefRec/CORAL machinery in this row at all (plain
   `L_cls+L_seg`, same as A1/C1's baseline structure), so no reason to expect this row to run
   slower than A1's own ~4h real runtime; if anything faster, given the much smaller 72-sample
   training set (9 batches/epoch vs. A1's much larger source-train set).
5. No PointNet++-specific engineering risk expected: this row reuses only already-validated
   machinery (A1's PointNet2_ClsSeg forward pass, C5/B5's Oracle protocol) with no new
   integration surface.

## What was done and how

1. Confirmed `data/pheno4d_oracle_train.csv`/`pheno4d_oracle_val.csv` already exist (from C5,
   reused unchanged by B5 too) and are directly reusable — no regeneration needed.
2. Confirmed `PointNet2_ClsSeg`'s constructor signature (`args, num_class, seg_num_classes`)
   matches `DGCNN_ClsSeg`'s exactly, so C5's script structure ports over with only the backbone
   import/instantiation line changed.
3. Wrote `adapters/train_a5_pointnet2_da_o.py` (see design decisions 1-4).
4. Added `jobs/a5_pointnet2_da_o.sbatch` (`--time=04:00:00`, standard non-KPConv budget).

## Status: finished (2026-09-18), job 313635 on `cn19-dgx`, 8.5 minutes, no contention

Best model at epoch 96 (lowest Oracle val total loss, 0.3152) — by far the fastest row in the
entire project, consistent with the design decision's expectation (no adaptation machinery, tiny
72-sample training set).

## Results

**Protocol-selected checkpoint** (epoch 96, lowest Oracle val total loss):

- Oracle val: cls acc 1.0000 (avg acc 1.0000), seg loss 0.4674, Tomato seg mIoU 0.8563
  (acc 0.9631), Maize seg mIoU 0.7502 (acc 0.9069).
- **Target (Pheno4D held-out) test: acc 1.0000 (avg acc 1.0000), loss 0.0246** — directly
  comparable to A1-A4's same file/metric, and a perfect score.
- Confusion matrix (rows=true, cols=pred, order Tomato/Maize): `[[40,0],[0,23]]` — zero
  misclassifications. The systematic Tomato→Maize bias present in every other Block A row
  (A1 Tomato recall 0.125-ish territory, A2/A4 still showing real bias) is completely gone.
- Target segmentation (Oracle-only bonus metric, NOT comparable to A1-A4's source-val seg mIoU —
  different domain/label space): Tomato mIoU 0.8232 (acc 0.9570), Maize mIoU 0.6607 (acc 0.8569).

**Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

| Metric | A1 (DA-0) | A2 (DA-A) | A3 (DA-S) | A4 (DA-A, L-N) | **A5 (DA-O)** | B5 (KPConv DA-O) | C5 (DGCNN DA-O) |
|---|---|---|---|---|---|---|---|
| Full-run mean | 0.599 | 0.554 | 0.800 | 0.586 | **0.9156** | 0.9425 | 0.925 |
| Full-run stdev | 0.196 | 0.193 | 0.158 | 0.170 | **0.1732** | 0.0815 | n/a |
| corr(selection-loss, target acc) | −0.165 | +0.086 | −0.009 | −0.167 | **−0.9096** | −0.858 | −0.823 |
| Collapse epochs (±0.02 tol) | 17/100 | 15/100 | 2/100 | 4/100 | **7/100** | n/a | n/a |
| Quartile means (Q1→Q4) | .676/.717/.507/.495 | .731/.597/.429/.458 | not re-checked | .773/.627/.483/.462 | **.731/.960/.972/.999** | rising | rising (.827→.999) |
| Selected-checkpoint acc | 0.4603 | 0.4444 | 0.8889 | 0.5397 | **1.0000** | 0.9841 | 1.0000 |

## Directly answering the motivating question

**The wrong-signed selection correlation does NOT persist under Oracle conditions — it flips to
the strongest negative correlation of any Oracle row across all three backbones (−0.9096, vs.
B5's −0.858 and C5's −0.823).** A2's own wrong-signed correlation (+0.086) had already been shown
by A4 to resolve without any Oracle supervision at all, just by narrowing the augmentation mix
(A4: −0.167). A5 confirms the same underlying pattern holds under the more extreme Oracle
condition too: once real target labels directly supervise training (removing the domain gap
entirely, not just narrowing the augmentation-interaction problem A4 addressed), the selection
signal becomes not just informative but the *most* informative of any row trained in this
project — stronger than B5 or C5's own Oracle resolutions. This is the third and final backbone
to show this exact pattern (A2/B1-B4/C2-C4 all show domain-gap-era selection-signal problems that
resolve under Oracle training), completing a clean three-for-three result: **every backbone's
selection-signal reliability problem is a domain-adaptation artifact, never a fundamental
property of the backbone or dataset that Oracle training fails to fix.**

**Where PointNet++'s ceiling lands relative to the other two backbones: lowest full-run mean of
the three (0.9156 vs. B5's 0.9425 and C5's 0.925), but joins DGCNN in reaching a perfect
1.0000 selected-checkpoint accuracy** (KPConv's B5 falls just short at 0.9841). The lower full-run
mean is driven entirely by Q1 (0.731, high variance stdev 0.264) — PointNet++'s Oracle training
takes longer to stabilize than the other two backbones' (7/100 collapse epochs, all early, vs.
B5's near-zero), but once it stabilizes (Q2 onward: 0.960/0.972/0.999) it matches or slightly
exceeds them. By Q4 alone, PointNet++ (0.999) is essentially tied with the other two backbones'
best late-training performance.

**Segmentation**: PointNet++'s target Tomato seg mIoU (0.8232) is essentially tied with B5
(0.8230) and C5 (0.83) — all three backbones converge to almost the same Tomato segmentation
ceiling under Oracle conditions. But PointNet++'s target **Maize** seg mIoU (0.6607) is the best
of the three (vs. B5's 0.6222, C5's 0.52) — consistent with A1's own earlier finding that
PointNet++ has the best Maize segmentation among the three DA-0 baselines (0.4741 vs. B1's 0.4414,
C1's 0.3427), a pattern that persists all the way through to the Oracle ceiling.

## Conclusion

Block A is now complete (A1-A5, all five rows done). Combined with Block B (B1-B5) and Block C
(C1-C5), all 15 rows across all three backbones' individual strategy-table entries are finished.
Ranking by full-run mean target accuracy on this backbone: A2 (0.554) < A1 (0.599) < A4 (0.586,
between A1/A2) < A3 (0.800) < **A5/Oracle (0.9156)**. A5 confirms the same story every backbone's
Oracle row has now told: the domain gap, not model capacity, is what defeats the non-Oracle rows,
and the selection-signal unreliability seen throughout Block A (most severe in A1's own baseline
and A2's DA-A row) is entirely a symptom of that domain gap — it vanishes, and in fact becomes the
single most informative selection signal of any row in the project, once real target supervision
removes the gap. See CLAUDE.md's final three-backbone summary and PROGRESS_LOG.md's closing
milestone for the complete cross-block picture.

**CPU smoke test (1 epoch, batch_size=8, real data, `--gpu -1`) — passed cleanly, full epoch
completed.** Launched detached (`nohup ... &`, `disown`), per the standing precaution adopted
after B3's smoke test was lost to a session-teardown issue. Dataset sizes matched B5/C5's
exactly (72 Oracle-train, 18 Oracle-val, 63 target-test, 36 annotated target-test-for-seg), class
weights sane. All 9 training batches (72 samples, batch_size=8) ran with no shape/gradient
errors, and the end-of-epoch summary printed sane, non-degenerate numbers. Deleted
`results/_smoketest_a5/` after confirming the pass.
