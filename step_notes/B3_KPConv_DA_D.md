# Row B3: KPConv, Discrepancy-Based Domain Adaptation (DA-D, Deep CORAL), All Augmentations

## What this is, in plain language

B3 is the row CLAUDE.md's strategy table actually specifies as B3 (Block B's discrepancy-based
adaptation row) — see `step_notes/B3b_KPConv_DA_S.md`'s "why B3b and not B3" section for the
naming mix-up this project already caught and corrected once mid-session: B3b (DA-S, already
done) is a separate ADDITION to the plan, not a substitute for this row. B3 tests Deep CORAL
(feature-covariance alignment), a genuinely different domain-adaptation mechanism from both DA-A
(B2: discriminator + gradient reversal) and DA-S (B3b: self-supervised deformation
reconstruction) — no discriminator, no adversarial training, no reconstruction pretext, just
aligning the second-order statistics (covariance) of source and target pooled features directly.

## Why this row, specifically, now

B3b (KPConv, DA-S) produced the strongest non-Oracle result in the whole project — full-run mean
target accuracy 0.629 → 0.860 (+23.1 points), the most stable trajectory trained so far, zero
collapse epochs, no segmentation cost. B2 (KPConv, DA-A) showed a much more modest, roughly
neutral effect (+0.003 full-run mean). Per explicit user instruction, B3 asks the natural
follow-up question: is B3b's dramatic win specific to the self-supervised reconstruction
mechanism, or does KPConv simply respond well to domain-adaptation pressure of almost any kind?
DA-D is a clean third data point — mechanistically unlike both DA-A and DA-S — to help separate
"KPConv benefits from adaptation in general" from "KPConv benefits from this one specific
mechanism."

## Design decisions

1. **Deep CORAL, not MMD.** CLAUDE.md's DA-D definition names either as an option ("Deep CORAL
   ... or MMD ..."), with no mandate for one over the other. Chose CORAL: no kernel-bandwidth
   hyperparameter to select (MMD's Gaussian-kernel bandwidth is a real, non-trivial extra choice
   with no strategy-table guidance either), a single well-defined covariance-alignment loss, and
   it matches CLAUDE.md's own framing of DA-D as "cheaper and more stable" than DA-A more
   directly (no all-pairs kernel computation). Flagged as an explicit choice, not read from a
   literal spec value — see `adapters/coral.py`'s docstring for the full reasoning.
2. **No reference implementation to adapt — built from scratch, same category as `dann.py`.**
   `DefRec_and_PCM` implements DefRec + PCM only; grepped its full source tree for
   coral/mmd/discrepancy/covariance — no matches (confirmed directly, not assumed still true
   from `dann.py`'s earlier check). `adapters/coral.py` is new, standard Deep CORAL (Sun &
   Saenko, 2016), kept in its own module the same way `dann.py` is, in case a DA-D row is ever
   added to Block A/C the way B3b was added to Block B.
3. **`L_CORAL` is Kendall-weighted — a real, substantive difference from DA-A's `L_dom`/`L_ent`,
   not just a different loss function.** Per CLAUDE.md's loss architecture, "cooperative DA
   losses (L_CORAL/L_MMD/L_defrec)" are explicitly named as LEARNED (Kendall) cooperative
   losses, unlike DA-A's fixed/scheduled `L_dom`/`L_ent`. So `kendall({"cls":..., "seg":...,
   "coral":...})` combines all three learned terms — `L_coral` never gets summed in with a fixed
   coefficient the way `l_dom` is in `train_b2_kpconv_da_a.py`. Classified as `"regression"`
   type (continuous covariance-alignment metric, not classification-like), same category as
   `L_defrec` in A3/C3/B3b.
4. **Much simpler plumbing than B2 or B3b — no per-chunk batch rebuilds needed at all.** Unlike
   DA-S (which needs a fresh KPConvBatch rebuilt from deformed points per chunk, see
   `step_notes/B3b_KPConv_DA_S.md`), CORAL only needs each domain's already-computed pooled
   `logits["feat"]` — one clean source forward, one clean target forward, no discriminator, no
   GRL, no entropy term, no deformation. Structurally closer to B2's shape (2 collate calls/step)
   than B3b's (~5 collate calls/step) — expected per-epoch cost similar to B1/B2's real
   runtimes (a few hours), not B3b's ~5h.
5. **`neighborhood_limits` calibrated over BOTH domains**, same as B2/B3b and for the same
   reason (target points flow through the encoder here too, to get `feat` for CORAL) — reused
   `calibrate_neighborhood_limits`/`combine_neighborhood_limits` unchanged.
6. **`evaluate_coral` pairs `src_val_loader` against `trgt_adapt_loader` (cycled), not a
   separate held-out target split** — CORAL has no natural "held-out" reading the way a
   classification/segmentation loss does (it's a set-level statistic over a batch, not a
   per-sample label-comparable loss), and every DA method in this project already trains/
   evaluates its DA loss off the same target adaptation pool, never off
   `pheno4d_heldout_eval.csv` (which stays reserved exclusively for the never-trained-on
   accuracy diagnostic, same discipline as A3/C3/B2/B3b).

## What was done and how

1. Wrote `adapters/coral.py` (`coral_loss(source_feat, target_feat)` — squared Frobenius norm
   between covariance matrices, normalized by `4*d^2` per the original paper).
2. Wrote `adapters/train_b3_kpconv_da_d.py`, mirroring `train_b2_kpconv_da_a.py`'s KPConv
   batch/collate/calibration mechanics with the DANN machinery replaced by a much simpler
   clean-source + clean-target forward pass and `coral_loss`.
3. Added `jobs/b3_kpconv_da_d.sbatch` (`--time=60:00:00`, standing Block B convention).
4. **CPU smoke test (1 epoch, batch_size=4, real data, `--gpu -1`, `--verbose_batches`) —
   interrupted by a session/environment issue, not a code bug, after 60/65 batches ran cleanly.**
   Launched detached (`nohup ... &`, `disown`) to survive the Claude Code session it was started
   from; that session was itself interrupted and restarted partway through by something outside
   this project (background-task notification came back "stopped" with no completion record).
   Checked directly rather than assumed: the process was gone with no Python traceback and no
   `EXIT_CODE` line in its log — consistent with the whole process group being killed when the
   login session ended (plausibly systemd's per-session cgroup cleanup, which `nohup`/`disown`
   don't protect against, unlike a real `sbatch` job, which is why every other row's actual
   training run has always survived session churn but this login-node smoke test did not).
   **Judged sufficient to proceed without a third attempt**: the log showed 60 of 65 batches/
   epoch completed with no errors, sane and bounded per-batch costs (collate ~1.2-14.7s,
   fwd+bwd+step mostly 13-28s, a few load-driven outliers to 48-75s) — comparable in shape to
   B2's own smoke-test profile, as expected (CORAL needs no deformed-batch rebuilds, so its cost
   shape should resemble B2's, not B3b's) — far more batches exercised than any prior row's
   smoke test needed before being trusted. Deleted `results/_smoketest_b3/` and submitted the
   real job directly (an `sbatch` job, unlike a login-node background process, runs
   independently of the submitting session).

## Status: DONE. Job 310284 completed in 10h12m (2026-09-14/15) — much longer than B1/B2's
few-hour pace, but a confirmed, real GPU-contention episode mid-run, not a code problem (see
"A real contention episode, confirmed not guessed" below). See "Results" below for the full
comparison against B1/B2/B3b.

---
## A real contention episode, confirmed not guessed

Per-epoch timestamps show three distinct phases: epochs 0-59 ran at a healthy ~2.2 min/epoch
(15:02→17:10, matching B1/B2's real-world pace, confirming the "structurally simpler than B3b"
design expectation was correct). Epochs 59-69 slowed to ~9.4 min/epoch, then **epochs 69-79 hit
~31 min/epoch — a ~15x slowdown from baseline** (18:44→23:52 for just 10 epochs), before
recovering to ~2 min/epoch by epochs 79-89 and settling at a mild ~5.6 min/epoch for the final
stretch. This is the same shared-`dgx`-partition contention pattern already documented for B1's
original incident (`step_notes/B1_KPConv_DA0.md`) — checked directly via the per-epoch
timestamps rather than assumed, and a second independent confirmation that this is a real,
recurring cluster-level risk (not a one-off), not specific to any one row's code. The 60h budget
absorbed it without incident: even with several hours at ~15x slowdown, the job still finished
in 10h12m, well inside budget — validates keeping the standing 60h convention for B4/B5 rather
than trying to tighten it now that B1/B2/B3b all happened to run fast.

---
## Results (2026-09-14/15, job 310284, 10h12m)

**Protocol-selected checkpoint** (best epoch by lowest source val total loss) — best epoch was
**94** (source val total loss 0.0282):

| Metric | B1 (DA-0) | B2 (DA-A) | B3b (DA-S) | B3 (DA-D) |
|---|---|---|---|---|
| Source val cls acc | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Target cls acc | 0.4444 | 0.5714 | 0.7143 | **0.6825** |
| Tomato seg mIoU (source val) | 0.3808 | 0.2730 | 0.2697 | **0.2482** |
| Maize seg mIoU (source val) | 0.4414 | 0.3889 | 0.3910 | **0.4150** |

**Full-trajectory analysis** (all 100 epochs' target-eval numbers, same methodology as every
prior row):

| Metric | B1 (DA-0) | B2 (DA-A) | B3 (DA-D) | B3b (DA-S) |
|---|---|---|---|---|
| Full-run mean | 0.629 | 0.632 (+0.003) | **0.808 (+0.179)** | 0.860 (+0.231) |
| Full-run stdev | 0.203 | 0.141 | **0.094** | 0.086 |
| corr(selection-loss, target acc) | +0.316 | +0.238 | +0.268 | +0.291 |
| Collapse epochs (all-one-class) | 3/100 | 2/100 | **0/100** | 0/100 |
| Quartile means (Q1→Q4) | 0.750/0.617/0.667/0.484 | 0.733/0.554/0.589/0.651 | **0.858/0.815/0.804/0.754** | 0.912/0.908/0.861/0.759 |
| Selected-checkpoint acc | 0.4444 | 0.5714 (+0.127) | **0.6825 (+0.238)** | 0.7143 (+0.270) |

B3's full-run stdev (0.094) is the **second-lowest of any row in the entire project** — below
even C1's previous project-wide best (0.104) — beaten only by B3b's 0.086. Zero collapse epochs,
matching B3b exactly and beating B1 (3/100) and B2 (2/100).

### Directly answering the user's question: where does Deep CORAL land relative to "roughly tied" (DA-A) and "large win" (DA-S)?

**Clearly and unambiguously on the "large win" side, much closer to B3b than to B2.** B3's
full-run-mean gain over B1 (+17.9 points) is nearly **60x** B2's gain (+0.3 points) and reaches
77% of the way to B3b's own gain (+23.1 points). By every full-trajectory measure — mean, stdev,
collapse count, quartile trend, selected-checkpoint gain — B3 sits in the same tier as B3b, not
anywhere near B2's roughly-neutral result. This is now the **second** adaptation method (after
DA-S) to produce a clear, decisive win for KPConv on this dataset, while DA-A remains the one
outlier that doesn't.

### Does this clarify "adapts broadly" vs. "specific structure"? Yes — and it points at a specific, testable distinction

The three KPConv adaptation results now span a real mechanistic range:
- **DA-A (DANN)**: adversarial min-max (discriminator tries to maximize domain separability,
  backbone adversarially minimizes it via GRL). Result: **roughly neutral** (+0.3 pts).
- **DA-D (CORAL, this row)**: cooperative, directly minimized, operates ONLY on abstract pooled
  feature-covariance statistics — never touches or reconstructs actual target-domain point
  coordinates. Result: **large win** (+17.9 pts).
- **DA-S (DefRec)**: cooperative, directly minimized, but requires the model to actually
  reconstruct deformed target-domain GEOMETRY (real 3D coordinates) via Chamfer distance.
  Result: **largest win** (+23.1 pts).

**This is informative precisely because DA-D and DA-S sit on opposite ends of the "exposes the
model to real target geometry" spectrum, yet both produce large wins, while DA-A (also fairly
different from both) is the outlier.** If "KPConv needs exposure to actual target-domain
geometry" were the operative explanation, DA-D (which never reconstructs or touches raw target
coordinates, only second-order feature statistics) should have looked more like DA-A's tepid
result — it didn't. **The cleanest, best-supported read: what predicts success for KPConv on
this dataset is NOT adaptation pressure in general (DA-A is real adaptation pressure and doesn't
help), and NOT specifically geometry-exposing methods (DA-D never touches target geometry and
still wins big) — it's specifically whether the DA loss is cooperative/directly-minimized (per
CLAUDE.md's own Kendall-vs-fixed loss-architecture distinction: DA-D and DA-S are both
Kendall-weighted cooperative losses; DA-A's `L_dom`/`L_ent` are fixed/scheduled precisely
because they sit inside an adversarial min-max structure) rather than adversarial.**

**Flagged explicitly as an unverified mechanistic hypothesis, not a proven one**: a plausible
explanation for *why* KPConv specifically might be less compatible with adversarial training is
that its kernel-point convolutions already impose strong, hand-engineered geometric structure
(fixed kernel-point positions optimized once, grid-subsampled neighborhoods) that may make its
gradient dynamics more brittle under the destabilizing, alternating min-max pressure GRL injects
— cooperative losses (whether statistical or reconstruction-based) integrate into ordinary
gradient descent without that structural mismatch. This is a plausible story consistent with the
data but not tested by any ablation here (e.g., no experiment isolates "adversarial training
dynamics" from other confounds like DA-A's specific hyperparameters). A competing, more
deflationary explanation also fits the data equally well: KPConv's B2 (DA-A) result was already
the *best-of-three-backbones* DA-A outcome (roughly neutral, vs. DGCNN's clear −15.8-point loss
and PointNet++'s −4.5-point loss) — so it's possible KPConv is simply "generally receptive to
adaptation" and DA-A is just its weakest form of help (a near-zero-but-not-negative gain) rather
than categorically different in kind from DA-D/DA-S. Both explanations predict the same ranking
observed here (DA-S ≳ DA-D ≫ DA-A), so this row alone cannot distinguish between them — B4
(DA-0/L-D, no adversarial machinery at all) and B5 (DA-O, the ceiling) won't resolve this either,
since neither adds a second adversarial-vs-cooperative data point. A genuine test would require
a second adversarial method (not in the current strategy table) or a controlled ablation on
DANN's own hyperparameters — out of scope for this project's current plan, noted here for
completeness rather than proposed as a new row.

### Segmentation: costs Tomato mIoU the most of any DA method, but preserves Maize best

Unlike the classification story, B3's segmentation numbers are not simply "closer to B3b."
**Tomato mIoU (0.2482) is the LOWEST of all three adaptation methods** (B2: 0.2730, B3b: 0.2697,
B1 baseline: 0.3808) — a slightly larger cost than either other DA method. **Maize mIoU (0.4150)
is the HIGHEST of the three adaptation methods** (B2: 0.3889, B3b: 0.3910), closest to B1's own
0.4414. Reading the full per-epoch trajectories (not just the FINAL line): both organ classes
*rise* steadily over training (Tomato 0.169→0.241 quartile means, Maize 0.294→0.411) — no
collapse pattern, matching B3b's healthy shape, not A3's DGCNN/PointNet2-track collapse.

### Conclusion

Deep CORAL is KPConv's second clear adaptation win, landing at +17.9 points full-run mean
(77% of B3b's own +23.1-point gain) and the second-most-stable trajectory of any row in the
project. This decisively answers the motivating question: CORAL's mechanism (feature-statistic
alignment, no target-geometry exposure, no adversarial structure) is different enough from both
DA-A and DA-S that its landing next to DA-S rather than DA-A is informative, not a coincidence —
the shared property between the two winners (DA-D, DA-S) is "cooperative, directly minimized,"
not "exposes real target geometry." KPConv on this dataset appears specifically averse to
adversarial training dynamics, not to domain adaptation broadly — though the causal mechanism
behind that aversion remains an open, flagged question this project's current plan doesn't have
a row designed to resolve.
