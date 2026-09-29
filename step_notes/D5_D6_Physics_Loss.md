# D5/D6: Physics-Informed Growth Loss (Logistic/Gompertz ODE)

Block D rows D5 (`D4 + Logistic ODE physics loss + L_mono`) and D6 (`D4 +
Gompertz ODE physics loss + L_mono`), per CLAUDE.md's Block D definition
and loss-architecture section. Only D5 (Logistic) is implemented and
committed so far (`adapters/physics_loss.py`,
`adapters/train_d5_fusion_physics_logistic.py`); D6 (Gompertz) reuses the
same `physics_loss.py` machinery (already `ode_family`-parameterized) and
has not been built yet.

Zero-context-reader summary: this file documents (1) why L_phys uses a
finite-difference derivative instead of the docs' literal "derivative via
autodiff" instruction, (2) why rho/beta are softplus-bounded, and (3) why
rho and K/alpha now get separate, very different learning rates — all
three findings came from synthetic recovery tests with a KNOWN ground-truth
ODE solution, run before trusting the loss on real data, per CLAUDE.md's
"flag inferences vs. confirmed facts" style rule.

## Spec (from the primary docs)

- `docs/plant_3d_pipeline_v6_corrected.drawio` (node id="19"): "derivative
  via autodiff", "rho,K / alpha,beta trainable".
- CLAUDE.md loss architecture: Logistic ODE (raw trait space)
  `eps(t) = dy/dt - rho*y*(1 - y/K)`; Gompertz ODE (log space, `y=ln V`)
  `eps(t) = dy/dt - (alpha - beta*y)`; `L_mono` applies ONLY to
  height/stem_diameter (never volume, since leaf area/volume/count can
  legitimately decrease via senescence); `L_phys`/`L_mono` are
  fixed/scheduled, NEVER Kendall-weighted (unlike `L_trait`/`L_growth`,
  which are cooperative and Kendall-weighted).

## Design decisions

1. **PHYSICS_TERMS = {height, stem_diameter} only, never volume** — matches
   `L_mono`'s explicit scope in both docs, and D0's `growth_curves.py`
   already only ever fits Logistic/Gompertz to those two traits (see
   `step_notes/D0_Trait_Extraction_Pipeline.md`).
2. **rho/K (or alpha/beta) are GLOBAL trainable `nn.Parameter` scalars**,
   not per-plant (no stated mechanism to generalize to held-out test
   plants) and not per-species (no such categorical structure in either
   doc) — matches this codebase's own Kendall-uncertainty-weighting
   precedent of a small number of global learned scalars. Separate scalars
   per trait (height vs. stem_diameter), matching D0's own convention of
   fitting them as two independent curves per plant. Initialized from D0's
   population-average fitted values (`load_d0_population_init`) — a
   reasonable starting point, not a fixed target; fully trainable
   throughout.
3. **The rate parameter (rho for Logistic, beta for Gompertz) is
   softplus-bounded** (`effective_rate_param` = `softplus(raw) + 1e-4`).
   Found necessary via an early synthetic recovery test (2026-09-28): with
   an UNCONSTRAINED trainable rho, a `lambda_phys` sweep (1e-4 to 10.0)
   showed the ODE residual dropping toward 0 (0.20→0.0008) while rho ALSO
   dropped toward 0 (0.028→0.001) instead of the true generating value
   (0.30) — the optimizer was minimizing `L_phys` by making the physics
   law itself vacuous (`dydt - rho*y*(1-y/K)` collapses to just `dydt` as
   `rho→0`), not by genuinely recovering the growth law. Necessary but NOT
   sufficient on its own — see finding 1 below.
4. **L_phys/L_mono are fixed-weight, never Kendall-weighted** — per
   CLAUDE.md's two-regime rule.
5. **L_phys and L_mono are both computed ONCE PER EPOCH**, not per
   mini-batch, from a SINGLE shared full-table forward pass
   (`run_physics_mono_step`) — both need ordered same-plant scan pairs,
   which D1-D4's per-scan-independent `DataLoader` batching can't provide.
   `L_mono` uses consecutive pairs within the D4-filtered table as-is (may
   skip an excluded scan — an already-documented simplification, harmless
   for a monotonicity check). `L_phys` is stricter: a pair is DROPPED if
   any real scan of that plant (checked against the full, unfiltered
   timeseries table) falls strictly between the two `elapsed_days` values
   — an excluded scan straddling the pair would silently inflate `dt` and
   distort the finite-difference rate estimate.

## Finding 1: autodiff constrains the wrong quantity (2026-09-29)

The docs literally say "derivative via autodiff". Implemented first
(`physics_loss.py::compute_physics_residual` /
`compute_l_phys_autodiff_ISOLATED_PARTIAL_retired`, kept in the file, not
deleted, clearly marked RETIRED): rebuilds the same per-term input slice
`PerTermFusionHeadD4.forward` uses internally, but with `elapsed_days`
replaced by a fresh `requires_grad=True` leaf, everything else (including
`prev_height`, D4's own lag-1 feature) detached, then
`torch.autograd.grad` differentiates the model's output w.r.t. that leaf.

Verified correct in isolation to machine precision (the derivative
computation itself is right). But a synthetic recovery test with a KNOWN
Logistic solution showed this constrains the WRONG quantity: autodiff
computes `dy/d(elapsed_days)` **holding `prev_height` fixed** — an
"isolated partial derivative". But `prev_height` is itself a function of
time along the real trajectory (moving from scan `t_k` to `t_{k+1}`
changes `prev_height` too), so the isolated partial derivative and the
TOTAL trajectory derivative (finite difference between the model's own
predictions at real consecutive scans) are DIFFERENT quantities.

Measured directly: isolated-partial RMS residual 0.12 (height) vs. TOTAL
trajectory RMS residual 4.72 (height) — an ~40x gap. The optimizer could
trivially minimize the isolated-partial residual (via rho collapsing
toward its floor) without the model's REAL predicted growth rate ever
becoming physically consistent.

**Fix**: `L_phys` now uses `compute_l_phys_finite_diff` — a finite
difference between the model's own predictions at real consecutive scans
(same once-per-epoch computation style as `L_mono`), not autodiff. This is
a recorded, documented departure from the docs' literal instruction, not a
silent substitution.

## Finding 2: rho recovers, K doesn't, under a shared learning rate (2026-09-29)

First synthetic recovery test of the finite-difference redesign: rho
recovery improved substantially vs. the autodiff version (65.8%/58.8%
relative error across the two physics terms, down from ~90%+ under
autodiff) — confirming the finite-difference fix works directionally. But
K barely moved from its initialization.

Hypothesis going in: K's gradient in the Logistic residual scales with
rho (`d(eps)/dK = rho*y^2/K^2` from differentiating
`rho*y*(1-y/K)` w.r.t. `K`), and rho started small (near the softplus
floor) — so while rho is still recovering, K's gradient is proportionally
weak too. rho and K shared a single learning rate (`lr_ode=2e-2`, applied
to both via one `trainable_ode_params` list / one optimizer param group).

## Finding 3: separate learning rates fix BOTH parameters (2026-09-29)

Built a standalone follow-up test (`scripts/d5_synthetic_recovery_test.py`
— diagnostic only, not part of the training pipeline, same category as
`scripts/inspect_crops3d_sf.py`) to test the hypothesis directly:

- 12 synthetic plants, 7 scans each (days `[0,4,8,13,18,24,30]`, uneven
  spacing like real Pheno4D data), generated from the exact analytic
  Logistic solution (`rho_true=0.30`, `K_true=100`, per-plant variation in
  `K`/`y0` to mimic real inter-plant variability) plus observation noise
  (`std=1.5`).
- A small MLP (`elapsed_days → predicted y`) trained jointly against the
  noisy synthetic observations (an `L_fit` term standing in for
  `L_trait`) plus `lambda_phys * L_phys`, using the SAME
  `effective_rate_param`/`logistic_residual` functions
  `physics_loss.py` uses, and the same finite-difference-on-consecutive-
  real-scans formula as `compute_l_phys_finite_diff`.
- rho and K both initialized far from truth (`rho0=0.05` vs. true 0.30,
  `K0=40` vs. true 100) to reproduce the same "stuck" regime.
- Compared `lr_K in {0.02 (=lr_ode baseline), 0.1 (5x), 0.2 (10x)}`, 3000
  epochs each, `lambda_phys=0.1`.

**Results:**

| `lr_K` | final rho | rho rel. err | final K | K rel. err |
|---|---|---|---|---|
| 0.02 (1x, shared with rho) | 0.0002 | 99.9% | 47.84 | 52.2% |
| 0.10 (5x) | 0.2800 | 6.7% | 102.20 | 2.2% |
| 0.20 (10x) | 0.2800 | 6.7% | 102.20 | 2.2% |

At the shared/baseline rate, rho doesn't just fail to recover — it
actively COLLAPSES toward 0 (99.9% relative error), while K stays stuck at
52% error. This is the SAME vacuous-law failure mode `effective_rate_param`
was built to prevent (finding 3, above), just resurfacing via a different
path: a slow-moving, still-wrong K makes the residual easiest to shrink by
crushing rho instead, since `rho*y*(1-y/K)` collapses toward `0` either way
`rho→0` or `K` stays wrong in a way that keeps the product small over the
observed data range. K staying wrong wasn't just cosmetic — it was actively
blocking rho's recovery too.

At 5x and 10x lr_ode, BOTH parameters recover well and identically (5x and
10x converge to the same values) — so **5x (`lr_K=0.1`) is used as
sufficient**, not the largest multiplier tried.

**Decision, per explicit user instruction**: since K DID catch up under the
higher learning rate, this passes the check — proceed to real D5 training
with `lr_K=0.1` wired in as a separate optimizer param group (implemented
in `train_d5_fusion_physics_logistic.py::main`, alongside the existing
`lr_ode` group for rho). Had K NOT caught up, the instruction was to treat
that as informative rather than a hard gate and proceed anyway, given D0's
own finding below.

## Caveat carried into real D5 training: real Tomato K may not have a clean answer

`step_notes/D0_Trait_Extraction_Pipeline.md` already found that **4 of 7
real Tomato plants' height fits hit K's upper bound (5x the plant's own
observed max) during D0's own per-plant curve fitting** — a genuine
finding, not a fitting bug: those plants' height had not plateaued within
Pheno4D's observed scanning window, so the population itself doesn't have
one clean "true K" the way the synthetic test's ground truth does. This
means D5's real training run should not be judged by whether the single
GLOBAL K converges to a narrow value — real population heterogeneity
(including plants that never plateaued) makes that a much less
well-posed question than the synthetic test, which had an exact single
generating value by construction. K's real-data behavior is being tracked
as informative diagnostic signal, not a pass/fail gate on D5's validity.

## Remaining scope for D6 (Gompertz, not yet built)

`physics_loss.py` is already `ode_family`-parameterized
(`"logistic"`/`"gompertz"`) and `ode_params[term] = (raw_rate, other)`
already generically covers `(rho, K)` for Logistic and `(beta, alpha)` for
Gompertz — the RATE parameter (rho/beta) is always `raw_rate` (softplus-
bounded), and the non-rate parameter (K/alpha) is always `other`. The
`lr_K`-style fix found here applies identically to Gompertz's `alpha` when
D6 is built: `alpha`'s gradient in `eps = dydt - (alpha - beta*y_log)` is
`d(eps)/d(alpha) = -1` (NOT scaled by beta, unlike K) — so this specific
coupling may not reproduce for Gompertz's alpha the same way, and should be
checked with its own synthetic recovery test before assuming the same
5x multiplier applies, rather than copying the Logistic result over
unverified.
