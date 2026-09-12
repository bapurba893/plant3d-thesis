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

---
## Results (2026-09-12, job 308716, ~24 min wall time)

**Protocol-selected checkpoint** (best epoch by lowest *oracle* val total loss — the direct
analog of every other row's "lowest source val total loss," just with Pheno4D's own labeled
data playing the "source" role here — never touching the held-out target test set) — best
epoch was **89** (oracle val total loss 0.5666):

| Metric | Value |
|---|---|
| Oracle val cls acc | 1.0000 (avg acc 1.0000) |
| Oracle val Tomato seg | mIoU 0.8510 (acc 0.9603) |
| Oracle val Maize seg | mIoU 0.5745 (acc 0.8269) |
| **Target test cls acc (same file/metric as C1-C4)** | **1.0000 (avg acc 1.0000)** — perfect, 63/63 |
| Target test seg (annotated subset, Oracle-only metric) Tomato | mIoU 0.8330 (acc 0.9632) |
| Target test seg (annotated subset, Oracle-only metric) Maize | mIoU 0.5217 (acc 0.7974) |

**Full-trajectory analysis** (all 100 epochs' target-test cls acc, same methodology as every
other row — see `step_notes/C4_DGCNN_DA_A_LN.md` for the parsing approach reused here):

| Metric | C1 (DA-0) | C2 v2 (DA-A ALL) | C3 (DA-S) | C4 (DA-A L-N) | **C5 (DA-O, Oracle)** |
|---|---|---|---|---|---|
| Full-run mean target acc | 0.791 | 0.633 | 0.703 | 0.613 | **0.925** |
| Full-run stdev | 0.104 | 0.117 | 0.104 | 0.138 | **0.133** |
| corr(selection-loss, target acc) | −0.286 | −0.061 | +0.222 | +0.041 | **−0.823** |
| Quartile means (Q1→Q4) | .798/.834/.773/.760 | .691/.678/.608/.554 | .731/.718/.707/.657 | .646/.664/.582/.559 | **.827/.923/.951/.999** |
| Selected-checkpoint target acc | 0.7460 | 0.6667 | 0.5714 | 0.5556 | **1.0000** |
| Epochs collapsed to one class | 3/100 | 3/100 | not re-checked | 4/100 | 3/100 (early epochs only) |
| First epoch reaching acc ≥ 0.999 | never | never | never | never | epoch 23 |

**C5 is qualitatively different from every DA-0/DA-A/DA-S row in a way that matters a lot.**
Every prior row's quartile means *decline* across training (Q4 lower than Q1) and the
correlation between the model-selection signal and target accuracy is weak-to-actively-wrong
(C1: −0.29, weak; C2/C4: near zero, uninformative; C3: +0.22, actively wrong direction). C5's
quartiles *rise* monotonically (0.827→0.923→0.951→0.999) and its selection-signal correlation is
strongly negative (−0.823) — meaning that when the model has real target labels to train and
select against, more training reliably helps, and the loss the protocol watches for model
selection is actually a trustworthy proxy for target performance. That's the expected, healthy
shape for ordinary supervised learning; C1-C4 never show it, because none of them ever see a
real target label of any kind.

### Answering the user's question: how big is the gap, and what does it mean?

**The gap between C1's DA-0 baseline and C5's Oracle ceiling is large and real — C1 was NOT
already near the ceiling.**

- Reading full-run means (the fairer, trajectory-aware comparison this project has used
  throughout): **C5 0.925 vs. C1 0.791 — a 13.4-point gap.**
- Reading the protocol-selected checkpoints (what would typically get quoted as "the" number):
  **C5 1.0000 vs. C1 0.7460 — a 25.4-point gap**, with C5 reaching a perfect 63/63 on the target
  test set (both species, 100% precision/recall/F1).
- Either way, **substantial headroom exists.** This reframes C2/C3/C4's results: their failure
  to beat C1 is not "the task was already solved, nothing left to gain" — there was real,
  double-digit-point improvement available in principle, and none of the three domain-adaptation
  methods tried so far captured any of it (all three landed *below* C1, not just short of C5).
- **The limitation is a transfer problem, not a capacity or task-difficulty problem.** DGCNN
  itself is clearly capable of near-perfect target-domain species classification and reasonably
  good target-domain organ segmentation (Tomato mIoU 0.83, Maize mIoU 0.52 on Pheno4D's own real
  labels) when given actual target-domain supervision — the ~72-sample Oracle training set is
  smaller than Crops3D's 263-sample source-train set by more than 3x, and it still reaches a
  ceiling C1-C4 never approach. The sensor/acquisition-setup domain gap (see CLAUDE.md's Datasets
  section) is doing all of the damage in C1-C4, not model capacity or an intrinsically hard task.
- **Caveat on N:** the Oracle's own labeled pool is small (72 train / 18 val, 10 total plants) —
  same small-N regime flagged for C2's diagnosis. The target test set (63 scans, 4 plants) is
  also modest. A perfect 1.0000 on 63 samples is a strong result but not statistically
  bulletproof at this sample size; still, it's a genuine held-out number (the same protocol
  discipline as every other row — these 4 plants were never touched during Oracle training or
  model selection), not an artifact of leakage.
- **The target segmentation numbers (Tomato 0.83 / Maize 0.52 mIoU) are Oracle-only and not
  directly comparable to C1-C4's source-val seg mIoU** (different domain, different collapsed
  label space — see design decision 3 above) — included for completeness, not as a second
  "headroom" data point on the same axis as the classification comparison above.

### Conclusion

Block C is now complete (C1-C5). The Oracle ceiling (0.925 full-run mean / 1.0000 selected) sits
well above the DA-0 baseline (0.791 / 0.7460), confirming meaningful headroom exists for
domain-adaptation methods to claim on this task — none of DA-A (C2, C4) or DA-S (C3) claimed any
of it under the settings tried here; all three underperformed the no-adaptation baseline instead
of approaching the Oracle ceiling. Combined with C2/C4's shared diagnosis (adversarial adaptation
degrades over training regardless of augmentation mix) and C3's (self-supervised reconstruction
stays stable but doesn't transfer usefully either), the strongest current explanation is that
this project's specific domain-adaptation implementations aren't yet extracting the real,
Oracle-confirmed transferable signal that does exist between Crops3D and Pheno4D — not that no
such signal exists to extract.
