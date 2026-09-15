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

## Status: submitted to the cluster (2026-09-15), job 311004 on `cn17-dgx`
