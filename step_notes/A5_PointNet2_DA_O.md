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

## Status: submitted to the cluster (2026-09-18), job 313635 on `cn19-dgx`

**CPU smoke test (1 epoch, batch_size=8, real data, `--gpu -1`) — passed cleanly, full epoch
completed.** Launched detached (`nohup ... &`, `disown`), per the standing precaution adopted
after B3's smoke test was lost to a session-teardown issue. Dataset sizes matched B5/C5's
exactly (72 Oracle-train, 18 Oracle-val, 63 target-test, 36 annotated target-test-for-seg), class
weights sane. All 9 training batches (72 samples, batch_size=8) ran with no shape/gradient
errors, and the end-of-epoch summary printed sane, non-degenerate numbers. Deleted
`results/_smoketest_a5/` after confirming the pass.
