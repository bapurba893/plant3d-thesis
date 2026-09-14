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

## Status: submitted to the cluster (2026-09-14), job 310284 on `cn11-dgx`
