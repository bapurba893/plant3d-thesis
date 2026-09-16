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

## Status: submitted to the cluster (2026-09-15), job 311262 on `cn12-dgx`
