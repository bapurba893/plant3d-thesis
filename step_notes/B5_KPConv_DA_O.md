# Row B5: KPConv, DA-O (Oracle Upper Bound), All Augmentations

## What this is, in plain language

B5 is Block B's Oracle upper-bound reference point — the KPConv-track counterpart of
`train_c5_dgcnn_da_o.py` (row C5). Per CLAUDE.md: "DA-O ... NEVER a real candidate strategy —
exists only as the upper-bound reference point. Do not treat DA-O results as 'the best model.'"
Domain roles are swapped relative to every other Block B row: it trains directly on a small,
labeled subset of Pheno4D (the target dataset) instead of adapting from Crops3D, then evaluates
on the SAME held-out target test set B1-B4 all report against — measuring how much of KPConv's
target-domain gap is closeable in principle, given real target supervision. Completes Block B
(B1-B5 all done after this row).

## Why this row, specifically, now

Per explicit user instruction: every KPConv configuration trained so far (B1 DA-0, B2 DA-A, B3
DA-D, B3b DA-S, B4 DA-0/L-D) has shown the SAME wrong-signed selection-signal correlation —
`corr(selection-loss, target acc)` positive in every single one (+0.238 to +0.470), meaning a
better-looking source-domain validation loss has never once reliably predicted better
target-domain accuracy for this backbone on this dataset, regardless of adaptation method or
augmentation mix. B5 asks whether this is specifically a symptom of the domain-adaptation
problem itself (source/target distribution mismatch corrupting what the source-side selection
signal can see) — in which case it should disappear once the model is trained and selected
*directly* on labeled target data, with no domain gap to cross at all — or a more fundamental
property of this backbone+dataset pairing that persists even under Oracle conditions.

C5 (DGCNN, DA-O) already answered this question for DGCNN: `corr` flips to strongly negative
(−0.823, the single most informative selection signal of any Block C row) once real target
supervision is available — DGCNN's own selection-signal problems (mild to begin with) are
entirely a domain-adaptation artifact for that backbone. B5 checks whether KPConv matches that
pattern (selection-signal problem is domain-adaptation-specific, resolves under Oracle) or
becomes the first row where even direct target supervision doesn't fix it.

## Design decisions

1. **Mirrors `train_c5_dgcnn_da_o.py`'s Oracle protocol exactly**, with KPConv's batch/collate
   mechanics (from `train_b1_kpconv_da0.py`) substituted for DGCNN's raw-tensor forward pass —
   only the backbone and its required batch format differ. Same domain-role swap (train/val on
   Pheno4D's own annotated subset, test on the same held-out file B1-B4 use), same target
   segmentation eval as an Oracle-only bonus metric.
2. **Reuses C5's exact plant-disjoint Oracle train/val split unchanged**
   (`data/pheno4d_oracle_train.csv`: 8 plants, 72 scans; `data/pheno4d_oracle_val.csv`: 2 plants,
   18 scans) — no reason to re-derive a KPConv-specific split; C5 already validated this split is
   disjoint from the held-out test plants, and reusing it keeps B5 directly comparable to C5 on
   the same basis A1-A5/C1-C5 rows already share within their own blocks.
3. **`batch_size=8`, same data-driven necessity as C5** — 72 training samples; `batch_size=16` +
   `drop_last=True` (B1-B4's setting) would drop 8/72 samples (11%) every epoch and yield only 4
   batches/epoch, so `batch_size=8` (9 batches/epoch, uses all 72 samples) is used instead,
   matching C5's own reasoning exactly.
4. **`neighborhood_limits` calibrated on the Oracle train set ONLY, not both domains.** Unlike
   B2/B3/B3b (which push both Crops3D and Pheno4D batches through the encoder every step and
   need both-domain calibration to avoid B1's original OOM failure mode), B5 never touches
   Crops3D at all — single-domain (Pheno4D-only) calibration is correct here, the same pattern
   B1 already established for its own single-domain (Crops3D-only) DA-0 case, not an oversight.
5. **`--time=60:00:00`**, the standing Block B convention, even though this row's tiny 72-sample
   training set and simple (no adaptation machinery) structure should make it land near B1/B4's
   fast real runtimes (2-3h) or faster — the risk the budget guards against is cluster-level
   contention (confirmed twice now, B1 and B3), not this row's own intrinsic cost.

## What was done and how

1. Confirmed `data/pheno4d_oracle_train.csv`/`pheno4d_oracle_val.csv` already exist from C5 and
   are directly reusable (no regeneration needed).
2. Verified `PlantClsSegDataset`'s `label_transform`/`num_classes` constructor signature (added
   for C5) matches the call pattern used here before writing the full script.
3. Wrote `adapters/train_b5_kpconv_da_o.py` (see design decisions 1-4).
4. Added `jobs/b5_kpconv_da_o.sbatch` (`--time=60:00:00`, standing Block B convention).
5. **CPU smoke test (1 epoch, batch_size=4, real data, `--gpu -1`, `--verbose_batches`) —
   passed cleanly, full epoch completed.** Launched detached (`nohup ... &`, `disown`) from the
   start, per the standing precaution adopted after B3's smoke test was lost to a
   session-teardown issue. Dataset sizes matched C5's expectations exactly (72 Oracle-train, 18
   Oracle-val, 63 target-test, 36 annotated target-test-for-seg), class weights sane, calibration
   completed cleanly (`neighborhood_limits = [499, 36, 29]`, consistent with prior KPConv rows).
   All 18 batches of the single training epoch (batch_size=4, 72 samples) ran with no shape/
   gradient/OOM errors, and the end-of-epoch training-summary line printed correctly. Stopped
   after the full epoch (more evidence than any prior smoke test needed) and deleted
   `results/_smoketest_b5/`.

## Status: finished (2026-09-15/16), job 311262 on `cn12-dgx`, 45 minutes, no contention

Structurally simple + tiny dataset (72 training samples) meant this row ran near-instantly
compared to the rest of Block B — the fastest KPConv row by a wide margin, consistent with the
sbatch script's own pre-run expectation. Best model at epoch **97** (lowest Oracle val total
loss, 0.4244) — landed almost at the very end of training, similar to B1's epoch-99 selection.

## Results

**Protocol-selected checkpoint** (epoch 97, lowest Oracle val total loss):

- Oracle val: cls acc 1.0000 (avg acc 1.0000), seg loss 0.6116, Tomato seg mIoU 0.8313 (acc
  0.9614), Maize seg mIoU 0.7119 (acc 0.8991)
- **Target (Pheno4D held-out) test: acc 0.9841 (avg acc 0.9875), loss 0.0738** — directly
  comparable to B1-B4's target cls accuracy, same file/metric.
- Confusion matrix (rows=true, cols=pred, order Tomato/Maize): `[[39,1],[0,23]]` — a single
  Tomato→Maize misclassification, no Maize errors at all. Essentially the systematic Maize-bias
  seen in every other Block B row (B1 Tomato recall 0.125, B2 0.375, B4 0.300) is gone entirely
  here (Tomato recall 0.975) — real target supervision removes it.
- Target segmentation (Oracle-only bonus metric, NOT comparable to B1-B4's source-val seg mIoU —
  different domain/label space): Tomato mIoU 0.8230 (acc 0.9653), Maize mIoU 0.6222 (acc 0.8524).

**Full-trajectory comparison** (target held-out cls acc, all 100 epochs):

| Metric | B1 (DA-0) | B2 (DA-A) | B3 (DA-D) | B3b (DA-S) | B4 (DA-0/L-D) | **B5 (DA-O)** | C5 (DGCNN DA-O, ref) |
|---|---|---|---|---|---|---|---|
| Full-run mean | 0.629 | 0.632 | 0.808 | 0.860 | 0.649 | **0.9425** | 0.925 |
| Full-run stdev | 0.203 | 0.141 | 0.094 | 0.086 | 0.168 | **0.0815** | not re-checked |
| corr(selection-loss, target acc) | +0.316 | +0.238 | +0.268 | +0.291 | +0.470 | **−0.858** | −0.823 |
| Collapse epochs | 3/100 | 2/100 | 0/100 | 0/100 | 1/100 | 1/100 | not re-checked |
| Quartile means (Q1→Q4) | falling | dip+recover | rising | rising | falling | **0.876→0.972→0.964→0.958** (rise then hold) | 0.827→...→0.999 (rising) |
| Selected-checkpoint acc | 0.4444 | 0.5714 | 0.6825 | 0.7143 | 0.5556 | **0.9841** | 1.0000 |

## Directly answering the motivating question

**The wrong-signed selection correlation does NOT persist under Oracle conditions — it flips to
strongly negative (−0.858), resolving the exact same way DGCNN's did in C5 (−0.823), and in fact
slightly exceeds C5 in magnitude.** Every one of B1/B2/B3/B3b/B4's positive correlations
(+0.238 to +0.470, all wrong-signed — a worse-looking source/Oracle-val loss never once reliably
predicted worse target accuracy, and vice versa) reverses completely once the model is trained
and selected directly on labeled target data with no domain gap to cross. This settles the
question this row was built to test: **KPConv's selection-signal problem across B1-B4 is a
domain-adaptation artifact specific to the source→target distribution mismatch corrupting what a
source-side (or, for B2/B3/B3b, partially-target-exposed-but-still-source-anchored) validation
loss can see — not a fundamental property of the KPConv backbone or this dataset that persists
even when the domain gap is removed.** This directly parallels C5's finding for DGCNN and rules
out the competing hypothesis (that KPConv itself has some structural quirk, e.g. related to its
grid-subsampled neighborhood mechanics or B1's own worst-of-three-backbones baseline instability,
that would make its selection signal unreliable regardless of what data it's validated against).

Two more observations reinforce this reading rather than complicate it:
- B5's quartile trend **rises then holds at a high plateau** (0.876→0.972→0.964→0.958) — the
  same qualitative "loss keeps falling, target acc keeps rising/staying high" shape C5 showed,
  the opposite of every DA-0/DA-A/DA-D/DA-S row's "early peak then erode to a low floor" pattern.
  The one collapse epoch (1/100, avg_acc≈0.5) sits early in the trajectory (part of Q1's much
  higher variance, stdev 0.129 vs. Q2-Q4's 0.021-0.050) — consistent with a brief early-training
  instability before the tiny 72-sample Oracle set is fit well, not a recurring failure mode.
- **B5's full-run mean (0.9425) edges out even C5's own Oracle ceiling (0.925)** — a genuinely
  notable result: KPConv's Oracle upper bound is not just "as resolvable as DGCNN's," it's
  marginally the higher of the two ceilings found so far in this project, even though every one
  of KPConv's non-Oracle configurations (B1-B4, full-run means 0.629-0.860) sits at or below
  DGCNN's non-Oracle configurations' range in places (e.g. C1's 0.791). The gap this closes is
  therefore not a capacity gap — KPConv is fully capable of matching or slightly beating DGCNN's
  best possible target-domain performance — it is specifically a domain-adaptation-signal gap,
  reinforcing the same conclusion as the correlation flip above from a different angle.

**Caveat carried over from C5** (same small-N regime, same Oracle train/val split): 72
training / 18 validation scans is a small pool, and the 63-scan target test set means each
additional correct/incorrect prediction moves accuracy by ~1.6 points — the exact numbers (0.9841
vs. C5's 1.0000, or the marginal full-run-mean edge over C5) should be read as "both backbones
resolve to a near-ceiling ~0.92-1.00 ceiling under Oracle conditions," not as a precise ranking
between the two.

## Conclusion

Block B is now complete (B1-B5, all five rows done). Ranking by full-run mean target accuracy:
B1 (0.629) ≈ B2 (0.632) < B4 (0.649) < B3 (0.808) < B3b (0.860) < **B5/Oracle (0.9425)**. B5
confirms the domain gap — not model capacity, not some KPConv-specific structural defect — is
what's defeating B1/B2/B4, and specifically identifies *which part* of the pipeline breaks under
that gap: the model-selection signal itself, which is only unreliable when validated across a
domain shift, not in general. This closes out the question raised as early as B1 ("does KPConv's
claimed density-robustness show up as a stable baseline") with a sharper, more specific answer
than any single earlier row could give: KPConv is not unstable or defective in general (Oracle
training is remarkably stable, stdev 0.0815, the lowest full-run stdev of any Block B row) — its
instability and misleading selection signal are entirely products of being asked to bridge
Crops3D→Pheno4D without real target labels, exactly the problem domain adaptation exists to
solve, and exactly the problem B3/B3b's cooperative-loss methods (Deep CORAL, DefRec) make real,
substantial progress on while B2's adversarial DANN does not.
