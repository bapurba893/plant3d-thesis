# Row B4: KPConv, DA-0 (No Adaptation), L-D Augmentation Only

## What this is, in plain language

B4 is Block B's deliberately-unadapted density-robustness check, per CLAUDE.md's own strategy
table: `B4 DA-0/L-D (deliberately unadapted — confirms claimed density robustness)`. Unlike A4/C4
(which pair L-D/L-N with an adversarial method to isolate a named weakness on top of adaptation),
B4 pairs L-D (RandomCrop3D + CoarseDropout3D — the two augmentations that directly perturb local
point density/coverage) with NO adaptation method at all. This is Block B's substitute for a
DA-S row (Block A/C have A3/C3 as their third row; Block B never did — see
`step_notes/B3b_KPConv_DA_S.md`'s "why B3b and not B3" section for the full naming history) — B4
was always meant to be Block B's own, differently-shaped test of KPConv's claimed strength
("density-robust due to grid-subsampled neighborhoods, not fixed point count," per CLAUDE.md's
Backbones section).

## Why this row, specifically, now

Per explicit user instruction: given B1's already-mixed stability story (DGCNN-like collapse
resistance, but the highest continuous variance and most misleading selection signal of any DA-0
baseline in the project — see `step_notes/B1_KPConv_DA0.md`) and B2/B3/B3b's finding that KPConv
responds well specifically to cooperative adaptation losses (DA-D, DA-S) but not adversarial ones
(DA-A) — see `step_notes/B3_KPConv_DA_D.md` — B4 asks a cleanly different question: does the raw
architecture, with no adaptation signal of any kind, hold up against a targeted density/coverage
perturbation on its own? This directly tests the architecture's claimed property in isolation,
uncontaminated by any adaptation method's effect (positive or negative).

## Design decisions

1. **Identical to B1 in every respect except one line: `augment_mode="ld_only"` on
   `src_train_set`.** Same DA-0 protocol, same `L_cls`/`L_seg` spec, same Kendall combination,
   same model-selection protocol, same eval procedure — mirrors `train_b1_kpconv_da0.py`
   line-for-line except for that one dataset-construction argument. Matches exactly how row C4
   (DGCNN, DA-A, L-N) differed from C2 by one augmentation-pipeline swap and nothing else.
2. **New `compose_pipeline_ld_only`/`compose_pipeline_ld_only_with_labels` added to
   `scripts/augmentations.py`**, following the same pattern already established by
   `compose_pipeline_ln_only` (C4's L-N-only pipeline): `normalize()` (fixed preprocessing, not
   an invariance-inducing augmentation, same treatment CLAUDE.md gives Pad3D) + `random_crop3d`
   + `coarse_dropout3d`, skipping G-R (rotate/flip/cubic symmetry), G-S (scale), and L-N (noise)
   entirely. Same `keep_frac=0.85`/`n_holes=2`/`hole_frac=0.05` defaults as the L-D steps inside
   `compose_pipeline` (isolating which augmentation is applied, not changing its own parameters).
   Registered as `"ld_only"` in `adapters/dataset.py`'s `_AUGMENT_PIPELINES` dict — purely
   additive, `"all"`/`"ln_only"` callers unaffected.
3. **No changes needed to `PlantClsSegDataset.__getitem__`'s padding logic.** Checked directly:
   `pad_if_needed3d` already runs unconditionally after any augmentation pipeline (and again as a
   safety net if point count still doesn't match `target_n`), regardless of which `augment_mode`
   was used — so L-D's point-removing crop/dropout steps are already safely handled by existing,
   unmodified code.
4. **`neighborhood_limits` calibrated on source only, same as B1** (not B2/B3/B3b's both-domain
   calibration) — DA-0 never passes target points through the encoder at all, so there's no
   analogous risk here.
5. **`--time=60:00:00`**, the standing Block B convention, even though this row is structurally
   identical in cost shape to B1 (no adaptation machinery, no deformed-batch rebuilds) and
   should land near B1/B2's real runtimes (2-3h) absent contention — B3's own 10h12m runtime
   (mostly a mid-run contention episode, not higher intrinsic cost) is a second, independent
   confirmation this risk is real and recurring on the shared `dgx` partition, not one-off.

## What was done and how

1. Added `compose_pipeline_ld_only`/`compose_pipeline_ld_only_with_labels` to
   `scripts/augmentations.py` (see design decision 2). Verified directly on synthetic data before
   running the full smoke test (4096→3935 points after crop+dropout, points-only path; labels
   stayed aligned with points in the with-labels path) — cheap, fast correctness check before
   spending smoke-test time.
2. Registered `"ld_only"` in `adapters/dataset.py`'s `_AUGMENT_PIPELINES`.
3. Wrote `adapters/train_b4_kpconv_da0_ld.py` (see design decision 1).
4. Added `jobs/b4_kpconv_da0_ld.sbatch` (`--time=60:00:00`, standing Block B convention).
5. **CPU smoke test (1 epoch, batch_size=4, real data, `--gpu -1`, `--verbose_batches`) —
   passed cleanly.** Launched detached (`nohup ... &`, `disown`) from the start this time, given
   B3's smoke test was lost to a session-teardown issue (see `step_notes/B3_KPConv_DA_D.md`) —
   the precaution wasn't needed this time (no interruption), but kept as the new standing
   practice regardless. Calibration completed in ~26s (`neighborhood_limits = [499, 42, 31]`,
   comparable to B1's own values) and 26 training batches ran with no shape/gradient/OOM errors,
   fast and consistent per-batch costs (collate 0.4-3.5s, fwd+bwd 1.6-13s) — closely matching
   B1's own original smoke-test profile, as expected for a row with no adaptation machinery.
   Stopped early (more than enough batches to trust the code, same bar B3 already used) and
   deleted `results/_smoketest_b4/`.

## Status: DONE. Job 311004 completed in 2h10m (2026-09-15) — fast, clean run, no contention.
See "Results" below.

---
## Results (2026-09-15, job 311004, 2h10m)

**Protocol-selected checkpoint** (best epoch by lowest source val total loss) — best epoch was
**73** (source val total loss 0.3560):

| Metric | B1 (DA-0, ALL) | B4 (DA-0, L-D only) |
|---|---|---|
| Source val cls acc | 1.0000 | 1.0000 |
| Target cls acc | 0.4444 | **0.5556 (+0.111)** |
| Tomato seg mIoU (source val) | 0.3808 | **0.4286** |
| Maize seg mIoU (source val) | 0.4414 | **0.4624** |

**Full-trajectory analysis** (all 100 epochs' target-eval numbers, same methodology as every
prior row):

| Metric | B1 (DA-0, ALL) | B4 (DA-0, L-D only) |
|---|---|---|
| Full-run mean | 0.629 | 0.649 (+0.020) |
| Full-run stdev | 0.203 | 0.168 |
| corr(selection-loss, target acc) | +0.316 | **+0.470 (worse)** |
| Collapse epochs (all-one-class) | 3/100 | 1/100 |
| Quartile means (Q1→Q4) | 0.750/0.617/0.667/0.484 | 0.805/0.601/0.654/0.538 |
| Selected-checkpoint acc | 0.4444 | 0.5556 (+0.111) |

### Directly answering the motivating question: does isolating L-D augmentation (no adaptation method at all) confirm KPConv's claimed density-robustness cleanly?

**No — it's another mixed result, and on the single most diagnostic measure (selection-signal
reliability) it's actually WORSE than B1, not better.** The raw trajectory numbers look like a
mild improvement: full-run mean ticks up 2 points (0.629→0.649), stdev drops modestly
(0.203→0.168), collapse epochs fall (3→1), and the selected-checkpoint accuracy improves by 11.1
points (0.4444→0.5556). Taken alone, that could read as "narrowing to density/coverage-focused
augmentation helps a bit." **But `corr(selection-loss, target acc)` gets meaningfully worse, not
better — +0.316 → +0.470, the single most positive (most actively misleading) correlation of
any row trained in this entire project so far** (surpassing B1's own +0.316, previously the
project's worst, and C3's +0.222, the previous DGCNN-track record-holder for "actively
unhelpful"). A better-looking source-domain loss under L-D-only predicts worse target accuracy
even more reliably than it did under B1's full augmentation mix. Concretely: the protocol picked
epoch 73 (lowest source val loss), landing at 0.5556 — a real improvement over B1's own selected
checkpoint, but nowhere near the trajectory's own ceiling (max target acc 1.0000, hit earlier in
training) — the selection process is, if anything, leaving MORE headroom on the table than it
did for B1, not less.

**The quartile trend also still shows the same qualitative "early peak, then erode" DA-0 shape**
(0.805→0.601→0.654→0.538) that every DA-0/DA-A row in the project has shown — narrowing
augmentation to L-D alone does not change the fundamental shape of the drift, just its exact
magnitude. **So this row does not deliver a clean verdict on KPConv's claimed density-robustness
in isolation** — the architecture's raw trajectory performance is roughly flat-to-marginally-
better without the other three augmentation categories, but the SAME row makes the backbone's
already-documented selection-signal unreliability (first found in B1, persisting across B2/B3/
B3b regardless of adaptation method) measurably worse, not better. This adds a new data point to
B1's own conclusion ("KPConv's baseline has its own distinct instability signature... this
complicates, rather than confirms, the density-sensitivity hypothesis") rather than resolving it
— even in the cleanest possible test (density-focused augmentation, zero adaptation-method
interference), KPConv's selection-signal reliability problem is not just present but sharpened.

### One real, KPConv-specific augmentation effect: segmentation improves on BOTH species

Unlike C4 (DGCNN, L-N-only vs. ALL), where narrowing augmentation improved segmentation and the
mechanism was attributed to L-N never removing points while ALL's L-D component does, **B4 still
includes the same point-removing L-D augmentation B1's ALL mix does** — yet segmentation mIoU
improves for BOTH organ classes anyway (Tomato 0.3808→0.4286, Maize 0.4414→0.4624). Since the
point-removal mechanism can't explain this (L-D is present in both B1 and B4), the more likely
explanation here is the REMOVED augmentations (G-R rotation/flip/cubic-symmetry, G-S scale, L-N
noise) were themselves making segmentation harder for KPConv specifically — a different,
KPConv-specific mechanism from C4's DGCNN-specific point-removal story, flagged explicitly as an
inference, not verified by a dedicated ablation (no experiment here isolates which of G-R/G-S/
L-N is the actual driver).

### The systematic Maize-bias eases somewhat, but doesn't disappear

B4's confusion matrix: Tomato 12/40 correct (28 misclassified as Maize), Maize 23/23 correct —
Tomato recall 0.300, meaningfully better than B1's 0.125 (5/40 correct), though still heavily
Maize-biased in absolute terms. Consistent with the modest overall accuracy improvement, not a
qualitatively different failure mode.

### Conclusion

B4 does not cleanly confirm or refute KPConv's claimed density-robustness in isolation. Raw
trajectory performance is flat-to-marginally-better without G-R/G-S/L-N augmentation, and
segmentation quality genuinely improves on both organ classes (a real, KPConv-specific
augmentation effect, mechanism unverified). But the row's most diagnostic finding is a negative
one: the selection-signal unreliability first documented in B1 gets measurably WORSE under
L-D-only augmentation, not better — now the worst `corr(selection-loss, target acc)` of any row
in the project. Taken together with B1/B2/B3/B3b, KPConv's behavior on this dataset continues to
resist a single, clean narrative — its raw architectural handling of density/coverage
perturbation (this row) looks unremarkable-to-mildly-positive, while its trainability/
selectability (every row touching this metric so far) keeps looking worse than the other two
backbones', regardless of augmentation mix or adaptation method.
