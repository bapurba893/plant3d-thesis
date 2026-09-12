# Row C4: DGCNN, Adversarial Domain Adaptation (DA-A), L-N Augmentation Only

## What this is, in plain language

C4 asks a narrower question than C2 did: DGCNN's known architectural weakness is that its
graph-building step (dynamic k-nearest-neighbors) has no protection against noisy points —
a single noisy point can end up wired into the feature graph as if it were a real neighbor,
polluting the features around it. C4 tests whether the adversarial domain-adaptation method
(the same one C2 used) still works when the *only* augmentation applied during training is
Gaussian jitter (simulated sensor noise) — no rotation, scaling, flipping, cropping, or
dropout. Everything else (backbone, adaptation method, data, hyperparameters) is identical to
C2; only the augmentation mix changes. That isolation is the whole point of this row.

**Why it matters right now**: C2's first real run showed a serious instability problem (see
`step_notes/C2_DGCNN_DA_A.md`) that was root-caused to a source/target label-prior mismatch
interacting badly with an unramped entropy-minimization term, and fixed (entropy weight now
ramps alongside the GRL alpha, same as `λ_p`). C4 reuses that corrected `adapters/dann.py`
unchanged — so C4 becomes a direct test of whether the adversarial method's remaining behavior
(even after the fix) is a property of the method itself, or specific to the ALL augmentation mix
C2 happened to use. Per explicit user instruction, C4 gets the same full-trajectory analysis
(not just the selected checkpoint) as soon as results are in, not as an afterthought — exactly
because this comparison is only meaningful if both rows are read the same, careful way.

## Status: in progress (started 2026-09-11)

## Design decisions

1. **Byte-for-byte identical adversarial machinery to (corrected) C2.** `adapters/dann.py` is
   imported unchanged — same `GradientReversalLayer`, `DomainDiscriminator`,
   `lambda_p_schedule`, `lambda_ent_schedule` (the ramped-entropy fix). `train_c4_dgcnn_da_a_ln.py`
   is a near-line-for-line copy of `train_c2_dgcnn_da_a.py`; the only functional difference is
   the augmentation pipeline passed to the dataset classes.
2. **New augmentation plumbing, since nothing before this row needed anything but "ALL".**
   `scripts/augmentations.py` gained `compose_pipeline_ln_only`/`_with_labels` — `normalize()`
   (a fixed preprocessing step, not an augmentation category — same treatment CLAUDE.md gives
   Pad3D) followed by `gaussian_noise()` alone, skipping G-R (rotate/flip/cubic symmetry), G-S
   (scale), and L-D (crop/dropout) entirely. `adapters/dataset.py`'s `PlantClsSegDataset`/
   `PlantSpeciesDataset` gained an `augment_mode` parameter ("all" default, unchanged behavior
   for every existing caller; "ln_only" selects the new pipeline) rather than hardcoding
   `compose_pipeline` — an additive change, C1/A1/C2/C3 don't pass this argument and are
   unaffected.
3. **L-N applied to BOTH domains**, same as C2 applied ALL to both domains — consistent with
   how every other row in this project treats "the augmentation column" as something applied
   uniformly to source and target during training, not just source.
4. **Model selection**: identical protocol to C1/A1/C2 — best epoch by lowest source val total
   loss, never touching target labels.
5. **Hyperparameters identical to (corrected) C2** (epochs, batch size, lr, wd, dropout, disc
   dropout, gamma, lambda_ent) — isolating the augmentation variable means everything else must
   stay fixed.

## What was done and how

1. Added `augment_mode` plumbing (`adapters/dataset.py`) and
   `compose_pipeline_ln_only`/`_with_labels` (`scripts/augmentations.py`) — see design decisions
   above.
2. Wrote `adapters/train_c4_dgcnn_da_a_ln.py` as a near-line-for-line copy of
   `train_c2_dgcnn_da_a.py` (the corrected, ramped-`λ_ent` version), swapping only
   `augment_mode="ln_only"` for the source-train and target-adaptation-pool datasets.
3. CPU smoke test (1 epoch, batch size 8, real data, `--gpu -1`) passed end-to-end — sane
   non-zero cls/seg/dom/ent losses, `augment_mode` confirmed logged, `λ_ent` ramp confirmed
   working (`lambda_ent: 0.1000` at end of the single smoke-test epoch, same expected artifact
   as C2's smoke tests). Deleted `results/_smoketest_c4/` afterward.
4. Added `jobs/c4_dgcnn_da_a_ln.sbatch` — same cluster settings and hyperparameters as C2's
   corrected run (100 epochs, batch 32, lr 1e-3, wd 5e-5, dropout 0.5, disc dropout 0.3, gamma
   10.0, lambda_ent 0.1 max).
5. Waited for C2's entropy-ramp-fix rerun (job 308608) to finish and confirm the fix actually
   worked before spending GPU time on C4 with the same machinery — see
   `step_notes/C2_DGCNN_DA_A.md`'s "Corrected results" section. Confirmed: fix genuinely
   stabilizes training (stdev roughly halved, collapse-epochs cut 4.6x) but doesn't flip the
   DA-0-vs-DA-A comparison on its own -- so C4 is testing something real, not chasing a bug that
   was already explained by something else.

---
**2026-09-12:** Submitted `jobs/c4_dgcnn_da_a_ln.sbatch` to the `dgx` partition — job **308634**.
Queue was empty; no pre-existing `results/C4_dgcnn_da_a_ln/` directory (avoiding the log-append
issue hit on C2's rerun). Awaiting completion.

---
## Results (2026-09-12, job 308634, ~13 min wall time)

Parsed the full 100-epoch log the same way C2's corrected rerun was analyzed (`Val - Source N,
... total loss` for the selection signal, `Eval(no-train) - Target N, acc/avg acc` for the
domain-gap indicator on every epoch, never touched by training) so C4 and C2-v2 are read by an
identical, reproducible method — see the parsing done inline in this session, not a separate
script committed to the repo.

**Protocol-selected checkpoint** (same protocol as every other row: best epoch by lowest source
val total loss, never touching target labels) — best epoch was **80** (source val total loss
0.5009):

| Metric | C1 (DA-0) | C2 v2 (DA-A, ALL, fixed) | C4 (DA-A, L-N only) |
|---|---|---|---|
| Selected epoch | 95 | 66 | 80 |
| Target cls acc | 0.7460 (avg 0.7446) | 0.6667 (avg 0.7005) | 0.5556 (avg 0.6500) |
| Tomato seg mIoU (source val) | 0.3248 (acc 0.6645) | 0.2384 (acc 0.4765) | **0.4514** (acc 0.8216) |
| Maize seg mIoU (source val) | 0.3427 (acc 0.7698) | 0.3037 (acc 0.7744) | **0.4378** (acc 0.8813) |

**Full-trajectory analysis** (all 100 epochs' target-eval numbers, exactly as required before
trusting any single checkpoint — this is the comparison that actually answers the question):

| Metric | C1 (DA-0) | C2 v2 (DA-A, ALL, fixed) | C4 (DA-A, L-N only) |
|---|---|---|---|
| Full-run mean target acc | 0.791 | 0.633 | **0.613** |
| Full-run stdev (population) | 0.104 | 0.117 | **0.138** |
| corr(source val loss, target acc) | −0.286 | −0.061 | **+0.041** |
| Quartile means (Q1→Q4) | 0.798 / 0.834 / 0.773 / 0.760 | 0.691 / 0.678 / 0.608 / 0.554 | 0.646 / 0.664 / 0.582 / 0.559 |
| Min / max target acc over the run | 0.365 / 0.921 | 0.365 / 0.921 | 0.365 / 0.936 |
| Epochs collapsed to one class (acc≈23/63 or 40/63, avg_acc≈0.5, ±0.02 tol) | 3/100 | 3/100* | 4/100 |

*Recomputing C2-v2's collapse count with this session's exact tolerance (±0.02 on both `acc` and
`avg_acc`) gives 3/100, not the 5/100 figure in `step_notes/C2_DGCNN_DA_A.md` — the two counts
were produced independently and the earlier one wasn't reproduced verbatim; a plausible
explanation is a slightly different tolerance/rounding was used at the time, but this wasn't
re-derived from that session's exact code, so flagging as an unresolved minor discrepancy rather
than silently overwriting the earlier number. It doesn't change any conclusion below either way
(3 vs 5 out of 100 is noise at this sample size).

### Does noise-only augmentation help, hurt, or make no difference vs. ALL, given identical (fixed) adversarial machinery?

**Essentially no meaningful difference on classification transfer — both variants underperform
the DA-0 baseline by a similar, large margin, and both show the same qualitative failure shape.**

- Full-run mean target acc is close between the two DA-A variants (0.633 for ALL vs. 0.613 for
  L-N-only) and both sit well below C1's DA-0 mean (0.791) — a ~16-18 point gap regardless of
  which augmentation mix is used. Compared to the spread *within* either single trajectory (min
  0.365, max 0.92-0.94 — a ~55-57 point swing epoch to epoch), the ~2-point gap between the two
  DA-A variants' means is not a meaningful difference.
- The quartile pattern is qualitatively identical: both variants rise slightly then decline
  through Q3/Q4 (C2-v2: 0.691→0.678→0.608→0.554; C4: 0.646→0.664→0.582→0.559) — the same
  "early promise, later erosion" shape documented for C2, now reproduced under a completely
  different augmentation pipeline. This is evidence the decline is a property of the adversarial
  method (plus the label-prior-mismatch / small-N regime already root-caused in
  `step_notes/C2_DGCNN_DA_A.md`) rather than an artifact of the ALL augmentation mix C2 happened
  to use.
- `corr(source val loss, target acc)` is near-zero for both (−0.061 for C2-v2, +0.041 for C4) —
  in both cases the only selection signal DA-0/DA-A protocol is allowed to use (source val loss)
  carries essentially no information about target transfer. Neither augmentation choice fixes
  this.
- C4's single-checkpoint number (0.5556) looks much worse than C2-v2's (0.6667), but this is
  exactly the kind of misleading snapshot the full-trajectory analysis exists to catch (per
  C2's own diagnosis) — epoch 80 happens to land in a locally low patch of a trajectory whose
  full-run mean (0.613) is much closer to C2-v2's (0.633). Read the mean, not the selected
  checkpoint, for this comparison.
- C4's stdev is somewhat higher than C2-v2's (0.138 vs. 0.117) — L-N-only is not more stable;
  if anything marginally less stable epoch-to-epoch, though both are far more stable than the
  original unramped-entropy C2 run (0.214) discussed in `step_notes/C2_DGCNN_DA_A.md`.

**One place the augmentation choice clearly does matter: segmentation quality.** C4's source-val
seg mIoU is substantially better than both C1 and C2-v2 (Tomato 0.4514 vs. 0.3248/0.2384, Maize
0.4378 vs. 0.3427/0.3037). This is most plausibly explained by a confound, not by anything about
noise-robustness: ALL includes L-D (RandomCrop3D/CoarseDropout3D), which removes points from the
cloud, making the per-point segmentation task strictly harder regardless of domain adaptation;
L-N-only (jitter) perturbs point positions but never removes points. This is an inference about
mechanism, not a verified ablation (no run isolates "L-N-only, no adversarial" or "ALL minus L-D
only" to confirm point-removal specifically, rather than something else about the augmentation
mix, is the cause) — flagged as such rather than stated as fact.

### Conclusion

Restricting augmentation to noise-only (L-N), with the same (entropy-ramp-fixed) adversarial
machinery as C2, does **not** rescue adversarial adaptation's classification-transfer
underperformance vs. DA-0, and does not meaningfully change its qualitative failure pattern
(early-epoch peak, declining quartiles, near-zero correlation between the source-loss selection
signal and target accuracy). This is evidence that C2's post-fix behavior reflects something
about the adversarial method itself interacting with this dataset (small N, opposite
source/target class majorities, multi-temporal target — see `step_notes/C2_DGCNN_DA_A.md`),
not an artifact specific to the ALL augmentation mix C2 happened to test with. The augmentation
choice does have a real, large effect on segmentation quality, but that is best explained by
point-count preservation (no cropping/dropout in L-N-only) rather than by DGCNN's noise-weakness
narrative this row was originally designed to probe — the classification-transfer question this
row was built to answer comes back essentially a null result: no meaningful difference from ALL.
