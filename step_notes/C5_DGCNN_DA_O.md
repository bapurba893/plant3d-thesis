# Row C5: DGCNN, Oracle (DA-O), ALL Augmentation

## What this is, in plain language

C5 is the "Oracle" row on the DGCNN backbone — the upper-bound reference point for Block C.
Instead of training on labeled Crops3D and trying to transfer to unlabeled Pheno4D (what
C1-C4 all did), C5 trains directly on a small labeled slice of Pheno4D itself. Per CLAUDE.md,
DA-O is "NEVER a real candidate strategy — exists only as the upper-bound reference point": no
one would deploy a model that requires labeled data from every new sensor/greenhouse setup, but
knowing how well a model *could* do if it had that labeled data tells us how much of C1's
target-domain gap is closeable in principle (headroom for better domain-adaptation methods)
versus how much is just an artifact of DA-0 never having seen a single real target label.

## Status: submitted to the cluster (2026-09-12), job 308716

## Design decisions

1. **Domain roles are swapped relative to every other Block C row.** "Labeled train/val" here
   is Pheno4D's own annotated subset (not Crops3D); "target test" stays the SAME
   `data/pheno4d_heldout_eval.csv` every other row (C1-C4) reports its target cls acc against,
   so C5's classification number is directly comparable to theirs — this is the whole point of
   the row per the user's framing ("C5's gap over C1 will tell us how much headroom exists").
2. **Plant-level train/val split, not scan-level.** Pheno4D is multi-temporal (CLAUDE.md: "a
   structurally different, arguably harder domain gap" — see `step_notes/C2_DGCNN_DA_A.md`) —
   splitting by individual scan rather than by plant would let near-duplicate scans of the same
   plant leak across train/val, inflating val performance. Checked
   `data/pheno4d_adaptation_pool.csv`'s 90 annotated rows: exactly 5 Maize plants x 7 scans + 5
   Tomato plants x 11 scans, cleanly plant-disjoint from `pheno4d_heldout_eval.csv`'s 4 plants
   (M03/M04/T01/T03) already. Split the pool's 10 plants 8/2 (train: M01/M02/M05/M06 Maize +
   T02/T04/T05/T06 Tomato = 72 scans; val: M07 Maize + T07 Tomato = 18 scans), written to new
   tracked CSVs `data/pheno4d_oracle_train.csv` / `data/pheno4d_oracle_val.csv` (same schema as
   `pheno4d_adaptation_pool.csv`, pre-filtered to `is_annotated==True` only — an Oracle has no
   use for unlabeled samples).
3. **Pheno4D's own per-point labels needed a different, NEW label scheme — not Crops3D's
   `SEG_NUM_CLASSES`.** Scanned all 126 annotated Pheno4D files (both the pool's 90 and the
   held-out eval's 36) and found the raw ids are NOT a flat 3-class soil/stem/leaf scheme — they
   grow over the growing season (up to 42 distinct ids for Tomato's single label column, 5 for
   Maize's first of two label columns), consistent with individual LEAF INSTANCE ids being
   folded into the same column rather than a fixed semantic class count. Per-file scan showed
   id 0 dominates every file by a wide point-count margin (consistent with "soil"), id 1 is
   consistently the next-largest and far bigger than any higher id in both species (consistent
   with "stem" — one structure per plant vs. many smaller leaf instances), and every id >= 2 is a
   much smaller, roughly similarly-sized class (consistent with individual leaves). **Explicit
   inference, not confirmed against Pheno4D's original publication in this session**: collapsed
   raw id 0 -> soil (class 0), id 1 -> stem (class 1), id >= 2 -> a single leaf class (class 2),
   giving `PHENO4D_SEG_NUM_CLASSES = {"Tomato": 3, "Maize": 3}` — comparable in KIND (an organ
   task) but not in exact class count/id semantics to Crops3D's per-species scheme (Tomato=3,
   Maize=6). For Maize's two label columns, used the first (`label1`) and left the second
   unused — CLAUDE.md's text doesn't specify which is organ-semantic vs. instance-only, and both
   showed the same soil-dominant/stem-next/many-small-rest histogram shape, so this is a
   documented default, not a verified choice. See `adapters/dataset.py`'s
   `PHENO4D_SEG_NUM_CLASSES`/`pheno4d_collapse_organ_labels` docstrings for the full reasoning
   and histogram evidence.
4. **Model trained FROM SCRATCH on Pheno4D, not fine-tuned from a Crops3D-pretrained
   checkpoint.** CLAUDE.md's strategy table docx doesn't specify which; read "trains directly on
   labeled target data" as the conventional domain-adaptation-literature meaning of "Oracle" —
   as if the target were the only labeled dataset available — which is a from-scratch fit.
   Flagged as an inference.
5. **New target SEGMENTATION eval, which no other Block C row computes.** C1-C4 never touch any
   target label (that's the point of DA-0/DA-A/DA-S); C5 is the first row with real target
   per-point labels available at all (36 of the 63 held-out-eval scans are annotated). Reports
   this as a genuinely new metric, explicitly labeled in both the run log and here as **NOT
   comparable to C1-C4's source-val (Crops3D) seg mIoU** — different domain, different label
   space entirely (Crops3D's per-species organ ids vs. Pheno4D's collapsed soil/stem/leaf ids).
   It's an Oracle-only "how good could target segmentation be with real labels" number, useful on
   its own, not a same-axis comparison point.
6. **Same backbone/loss architecture as every other row**: DGCNN_ClsSeg (reused unmodified),
   Kendall-weighted `L_cls + L_seg` (both classification-type, same as C1-C4), Lovász `λ=1.0`,
   ALL augmentation (per this row's spec: DA-O/ALL).
7. **Hyperparameters identical to C1 except `--batch_size`** (32 -> 8): with only 72 training
   samples, batch_size=32 + `drop_last=True` would drop 8/72 samples (11%) every single epoch and
   yield only 2 batches/epoch; batch_size=8 uses all 72 samples across 9 batches/epoch. A
   data-driven necessity given the Oracle's much smaller labeled set, not a swept hyperparameter
   — everything else (epochs=100, lr=1e-3, wd=5e-5, dropout=0.5) stays identical to C1 for
   comparability of the training regime.

## What was done and how

1. Built the plant-level split (`data/pheno4d_oracle_train.csv`/`_val.csv`) directly from
   `data/pheno4d_adaptation_pool.csv`'s annotated rows — verified no plant_id overlap between
   train/val/(held-out test) anywhere.
2. Generalized `adapters/dataset.py::PlantClsSegDataset` with two new optional constructor
   args — `label_transform` (applied to raw cached labels right after loading, default None =
   no-op, so C1-C4 are unaffected) and `num_classes` (overrides the module-level Crops3D-shaped
   `SEG_NUM_CLASSES` in `seg_class_histogram`, default None falls back to the old global dict).
   Added `PHENO4D_SEG_NUM_CLASSES` and `pheno4d_collapse_organ_labels` — see design decision 3.
3. Wrote `adapters/train_c5_dgcnn_da_o.py`, structurally mirroring `train_c1_dgcnn_da0.py`'s
   `main()` almost line-for-line, with domains swapped (Pheno4D-oracle-train/val takes C1's
   Crops3D-train/val role) and one addition (the new target-segmentation eval block at the end,
   using `pheno4d_heldout_eval.csv` again but through `PlantClsSegDataset`, whose built-in "skip
   sample if cache has no 'labels' key" filter naturally restricts it to the 36 annotated scans).
4. **CPU smoke test** (1 epoch, batch_size=4, real data, `--gpu -1`): ran end-to-end in ~3
   minutes with no shape/gradient errors. Confirmed: 72/18 train/val split with correct
   per-species counts (Tomato 44/11, Maize 28/7), 36/63 target-seg filtering worked (log
   correctly warned "27 rows ... had no 'labels' key and were skipped" — matches 63-36=27
   unannotated held-out files), all losses non-zero and finite, `PHENO4D_SEG_NUM_CLASSES`
   histograms sane (soil-class dominant as expected: Tomato counts [109997, 11181, 59046], Maize
   [65914, 25156, 23618]). One epoch is far too little to judge model quality (target cls acc
   0.365 = predicting majority-class-only, as expected after a single epoch) — not a bug, just
   confirms the plumbing works. Deleted `results/_smoketest_c5/` afterward.
5. Added `jobs/c5_dgcnn_da_o.sbatch` — same cluster settings as every other row (dgx
   partition/qos, 1 GPU, 4h wall time), `--batch_size 8` per design decision 7.

---
**2026-09-12:** Submitted `jobs/c5_dgcnn_da_o.sbatch` to the `dgx` partition — job **308716**.
Queue was empty beforehand. Awaiting completion.
