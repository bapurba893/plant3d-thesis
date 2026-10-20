# Full-workflow rerun (FW) on P4a: KPConv (B1–B5, B3b) results

Scope: the second part of the full-workflow rerun described in `jobs/fw/README.md`. Every Block B
row (B1–B5, plus the added row B3b) was retrained with seeds 1, 2, 3 on the soil-kept
preprocessing P4a (`data/preprocessed_variants/P4a/preprocessed_manifest.csv`), with each row's
original arguments. The PointNet++ and DGCNN rows are in `step_notes/FW_1_Results.md`. Block D has
not been run on P4a.

Every number below comes from `results/pp_summary/fw_summary.md` / `fw_by_row.csv`, written by
`scripts/15_fw_summarize.py` from `results/FW_<row>_s<seed>/run.log` (regenerated in commit
d19ab46, "FW: summary after R4"). The DGCNN references C1 (0.8632) and C6 (0.8687) are taken from
`FW_1_Results.md`.

Target = Pheno4D held-out set, 63 clouds (40 tomato, 23 maize). Majority-class reference (always
Tomato) = 0.635.

Metrics: **full-run mean target accuracy** (mean of the per-epoch target accuracy over the 100
epochs; primary metric), full-run mean balanced accuracy, and selected-checkpoint target accuracy
(lowest source validation loss; for the Oracle row B5 the lowest Oracle validation loss). Target
test labels are never used for selection. Std = sample std over 3 seeds.

**Augmentation.** B1, B2, B3, B3b and B5 use ALL augmentation. B4 uses L-D only (RandomCrop3D +
CoarseDropout3D). So **B1 is the like-for-like baseline for B3 and B3b**; B4 differs from them in
augmentation as well as in DA method.

## Jobs

All on Prajna (`dgx`), every seed finished with an `OK` line in its job log
(`results/logs/fw_<row>_<jobid>.log`; B3b `fw_B3b_<jobid>_<task>.log`).

| Round | Rows (job id) |
|---|---|
| 3 | B5 (355614), B4 (355615), B1 (355616), B2 (355617); C3 (353781) is in `FW_1_Results.md` |
| 4 | B3 (357295, seeds run one after another, 9 h 37 min), B3b (357296, array of 3 tasks, one seed each, 6 h 20 min – 7 h 05 min) |

---

## 1. Per-row table (P4a, 3-seed mean ± std)

| Row | DA / Aug | Full-run mean target acc (s1 / s2 / s3) | Mean ± std | Bal. acc mean ± std | Selected-checkpoint acc mean ± std |
|---|---|---|---|---|---|
| B1 | DA-0 / ALL | 0.6637 / 0.6846 / 0.6703 | 0.6729 ± 0.0107 | 0.7220 ± 0.0072 | 0.7196 ± 0.0601 |
| B2 | DA-A / ALL | 0.7275 / 0.7138 / 0.6121 | 0.6844 ± 0.0631 | 0.7169 ± 0.0462 | 0.6720 ± 0.0242 |
| B3 | DA-D / ALL | 0.6913 / 0.6902 / 0.7246 | 0.7020 ± 0.0196 | 0.7329 ± 0.0010 | 0.6455 ± 0.0799 |
| B3b | DA-S / ALL | 0.8754 / 0.9079 / 0.7114 | 0.8316 ± 0.1053 | 0.8358 ± 0.0672 | 0.7302 ± 0.2136 |
| B4 | DA-0 / L-D | 0.8302 / 0.7508 / 0.7789 | 0.7866 ± 0.0402 | 0.8057 ± 0.0173 | 0.6297 ± 0.1860 |
| B5 | DA-O / ALL (Oracle, upper bound) | 0.9603 / 0.9486 / 0.9541 | 0.9543 ± 0.0059 | 0.9416 ± 0.0085 | 0.9947 ± 0.0092 |

Source-validation Maize segmentation mIoU (FINAL best-model line of each `run.log`, 3-seed mean;
not reported by the summary script): B1 0.434, B2 0.399, B3 0.414, B3b 0.403, B4 0.465. B5's
mIoU (0.693) is on Pheno4D's own labels and is not comparable with the other rows.

---

## 2. Comparison with the original single-seed runs (old preprocessing P0)

Original = the earlier strategy-table run: **seed 1 on P0** (`data/preprocessed/...`), read from
its own `run.log` with the same parser. FW = **seeds 1–3 on P4a**. Both the data and the number of
seeds differ, so a difference is not attributable to the preprocessing alone.

| Row | Full-run mean: original (s1, P0) | FW mean (P4a) | FW − orig | FW seeds above orig | Bal. acc: orig / FW / diff | Selected: orig / FW / diff |
|---|---|---|---|---|---|---|
| B1 | 0.6294 | 0.6729 | +0.0435 | 3/3 | 0.6881 / 0.7220 / +0.0340 | 0.4444 / 0.7196 / +0.2752 |
| B2 | 0.6319 | 0.6844 | +0.0525 | 2/3 | 0.6832 / 0.7169 / +0.0337 | 0.5714 / 0.6720 / +0.1006 |
| B3 | 0.8079 | 0.7020 | **−0.1059** | 0/3 | 0.8089 / 0.7329 / −0.0760 | 0.6825 / 0.6455 / −0.0370 |
| B3b | 0.8600 | 0.8316 | −0.0284 | 2/3 | 0.8584 / 0.8358 / −0.0226 | 0.7143 / 0.7302 / +0.0159 |
| B4 | 0.6494 | 0.7866 | +0.1373 | 3/3 | 0.7127 / 0.8057 / +0.0930 | 0.5556 / 0.6297 / +0.0741 |
| B5 | 0.9425 | 0.9543 | +0.0118 | 3/3 | 0.9341 / 0.9416 / +0.0075 | 0.9841 / 0.9947 / +0.0106 |

- **B3 lost about 0.106** against its original run, with all three seeds below it (0/3). It was
  the second-best KPConv adaptation row originally (0.808); on P4a it is 0.702.
- **The original B1→B3b gain of +0.231 (0.629 → 0.860) was single-seed** (one seed of each row on
  P0). On P4a with 3 seeds the B1→B3b difference in means is +0.159 (0.673 → 0.832).
- B4 moved the most upward (+0.137, 3/3 seeds above its original).
- As in `FW_1_Results.md`, a change of a few points against one original seed can be run-to-run
  noise; B3b's own seed std on P4a is 0.105.

---

## 3. All-3-seeds rule

Rule (same as Block PP and `FW_1_Results.md`): a variant beats a reference only if **all 3** of
its seeds exceed the reference's 3-seed mean. Metric: full-run mean target accuracy.

| Comparison | Reference mean | Variant seeds | Seeds above | Beats? |
|---|---|---|---|---|
| B3 vs B1 | 0.6729 | 0.6913 / 0.6902 / 0.7246 | 3/3 | Yes |
| B3b vs B1 | 0.6729 | 0.8754 / 0.9079 / 0.7114 | 3/3 | Yes |
| B3 vs B2 | 0.6844 | 0.6913 / 0.6902 / 0.7246 | 3/3 | Yes (lowest seed 0.6902 is 0.006 above) |
| B3b vs B4 (B4: L-D only, different augmentation) | 0.7866 | 0.8754 / 0.9079 / 0.7114 | 2/3 | No |
| B3 vs always-Tomato | 0.635 | 0.6913 / 0.6902 / 0.7246 | 3/3 | Yes |
| B3b vs always-Tomato | 0.635 | 0.8754 / 0.9079 / 0.7114 | 3/3 | Yes |
| B3b vs C1 (DGCNN DA-0) | 0.8632 | 0.8754 / 0.9079 / 0.7114 | 2/3 | No |
| B3b vs C6 (DGCNN DA-D) | 0.8687 | 0.8754 / 0.9079 / 0.7114 | 2/3 | No |

---

## 4. Findings

- **B3b beats B1 (its like-for-like ALL-augmentation baseline) on all 3 seeds**, and has the
  highest non-Oracle KPConv mean (0.8316). But it has **one weak seed (s3, 0.7114)**, and because
  of that seed it **does not pass the rule against B4, C1 or C6**. Two of its seeds (0.8754,
  0.9079) are above all three references.
- **B3 beats B1 and B2 on all 3 seeds, but by small margins** (+0.029 and +0.018 on the mean), and
  is below B4. Its balanced accuracy is very consistent across seeds (std 0.001).
- **The cooperative-vs-adversarial reading is not confirmed by the rerun.** CLAUDE.md's reading
  that cooperative DA losses (DA-D, DA-S) help KPConv while the adversarial one (DA-A) does not
  rested on the original single-seed gains over B1 (B3 +0.179, B3b +0.231, B2 +0.003). On P4a
  the gains over B1 are B3 +0.029, B3b +0.159 and B2 +0.012: DA-S still gives a large gain on
  average, DA-D no longer does.
- **B4 (DA-0, L-D only) is the second-highest non-Oracle KPConv row on P4a** (0.7866). It differs
  from B1 only in augmentation.
- **B5 is the Oracle upper bound**, not a candidate (CLAUDE.md, DA-O): 0.9543 ± 0.0059.

## 5. Selected-checkpoint re-evaluation mismatch

The summary script warns when the FINAL re-evaluation of the selected checkpoint differs from the
target accuracy logged at that epoch. Among KPConv FW runs:

- Earlier rows (round 3): **8 of 12 runs** — B1 s1, s2, s3; B2 s2, s3; B4 s1, s3; B5 s1.
- **B3:** s1 (0.7302 vs logged 0.7143) and s2 (0.5714 vs 0.5397). B3 s3 has a different warning:
  the logged best epoch (83) is not the argmin of the logged val total loss (87); possibly a
  4-decimal rounding tie, not checked.
- **B3b:** s1 (0.7937 vs 0.7460) and s3 (0.4921 vs 0.4444). B3b s2 has no warning.
- The original P0 runs of B1, B3, B3b and B4 show the same kind of mismatch.

**The cause is not verified.** `jobs/fw/README.md` notes that KPConv runs are not repeatable even
with the same seed, and names `batch_grid_subsampling`'s random grid rotation as the most likely
cause, but this has not been tested. The FINAL value is used, as in all earlier summaries.

## 6. Caveats

- **Soil cue.** P4a keeps soil and pot. As in `FW_1_Results.md` section 3.3, all P4a numbers may
  partly reflect a soil/pot cue rather than plant shape; no FW row was evaluated on the
  soil-removed target cache.
- **Single-seed originals.** Every comparison in section 2 is against one P0 seed.

## 7. Not done yet

- Block D on P4a: not started (open decisions in `jobs/fw/README.md`, "Block D").
- No FW row has been cross-evaluated on the soil-removed target cache.
- Cause of the KPConv selected-checkpoint mismatch: not investigated.

## Reproducing

```bash
python scripts/15_fw_summarize.py   # reads results/FW_*/run.log -> results/pp_summary/fw_summary.md, fw_by_row.csv
```
