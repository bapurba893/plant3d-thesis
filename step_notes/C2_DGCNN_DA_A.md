# Row C2: DGCNN, Adversarial Domain Adaptation (DA-A), All Augmentations

> **⚠ NUMBERS BELOW SUPERSEDED 2026-09-11 — see "Root-cause diagnosis and fix" at the bottom
> of this file for the corrected rerun.** Everything from here down through the original
> "Pass/fail" verdict is the *first* C2 run (unramped `λ_ent`, job 308389) and its full
> investigation, kept in full rather than deleted or rewritten — the diagnosis process (how the
> instability was traced to a source/target label-prior mismatch interacting with unramped
> entropy minimization) is itself a valuable, correct piece of work, and remains true as an
> explanation of *why that run behaved the way it did*. What changed is the fix that followed
> from it, and the corrected numbers that fix produced. Read the bottom section for the current,
> official C2 result.

## What this is, in plain language

C2 is the "anchor" adversarial row on the DGCNN backbone. Instead of training only on the
labeled source dataset (Crops3D) and hoping it happens to generalize (that's what C1/DA-0 did),
C2 actively forces the network to build features that look the same whether they came from
Crops3D or from unlabeled Pheno4D. It does this with a classic trick (DANN): a small second
network (a "domain discriminator") tries to guess which dataset a plant's features came from,
while the main network is trained to fool it. A "Gradient Reversal Layer" (GRL) is what makes
this a single, ordinary-looking training loop instead of two competing optimizers — it just
flips the sign of the discriminator's gradient on its way back into the main network, so one
`loss.backward()` call does both the "discriminator learns to detect domain" step and the "main
network learns to hide domain" step at once. On top of that, C2 also nudges the network to make
confident (not wishy-washy) predictions on target data it's never labeled ("entropy
minimization").

This is the row every other DA-A row in the table (A2, A4, B2, C4) copies exactly, so that any
difference between backbones is attributable to the architecture, not to the adaptation method
itself.

## Status: in progress (started 2026-09-11)

## Design decisions confirmed before writing code

Per CLAUDE.md's "Loss architecture" two-regime rule, `L_dom` and `L_ent` were built to be
**fixed/scheduled, never learned** — they stay completely outside the
`KendallUncertaintyWeighting` module that already combines `L_cls`/`L_seg` (unchanged from C1).
The reasoning CLAUDE.md gives (a learnable weight would let the network inflate its uncertainty
to silently disable adaptation) is why: total loss = `Kendall(L_cls, L_seg) + L_dom + λ_ent·L_ent`,
added on top of the Kendall term, not passed into it.

## What was done and how

1. **Checked the reference repo first.** Grepped all of `DefRec_and_PCM` for
   adversarial/domain_cls/dann/grad_rev/GRL/discriminator — no matches. Read `PointDA/trainer.py`
   in full: it implements DefRec (self-supervised deformation reconstruction) + PCM (point cloud
   mixup), not DANN — there is no domain-classification branch anywhere in the repo. So there was
   nothing to "adapt" for the adversarial machinery itself; only the DGCNN backbone (already
   wrapped by `adapters/models.py::DGCNN_ClsSeg`, reused unmodified) comes from the reference
   repo. This mirrors exactly how C1 handled the DA-0 gap (see that script's own docstring).
2. **Found the exact `λ_p` formula** by extracting the strategy table docx's raw text
   (`docs/Domain_Invariance_Strategy_Table.docx` → `word/document.xml`, stripped of XML tags) and
   grepping it, rather than guessing: `λ_p = 2/(1+e^(−10p)) − 1`, described as covering
   "Adversarial, physics, and distillation terms" that are "never assigned a learned σ" — this is
   the standard Ganin & Lempitsky (2016) DANN schedule, `p` = training progress in `[0,1]`.
3. **Two inference calls, both flagged explicitly** (the docx doesn't fully specify these — see
   `adapters/dann.py`'s docstring for the full reasoning, not repeated here):
   - `λ_p` is applied as the Gradient Reversal Layer's `alpha` only (ramping how hard the
     reversed gradient pushes the backbone), with `L_dom`'s own coefficient left at 1 in the
     total loss. Applying `λ_p` a second time as an outer multiplier would scale the backbone's
     adversarial gradient by `λ_p²` while only scaling the discriminator's gradient by `λ_p¹` —
     an asymmetric double-count. Single-application-via-GRL-alpha is standard practice in
     essentially every reference DANN implementation.
   - `L_ent`'s weight (`λ_ent`) has no formula anywhere in the docx (the sentence naming `λ_p`
     doesn't mention entropy minimization specifically). Used a fixed, CLI-configurable constant
     (default 0.1) instead of ramping it with `λ_p`, since ramping entropy minimization up
     alongside a still-unreliable discriminator early in training risks reinforcing
     confidently-wrong target pseudo-predictions.
4. **Built `adapters/dann.py`** — new, backbone-agnostic module (not folded into
   `losses.py`/`models.py`) holding the Gradient Reversal Layer (`_GradReverse` autograd
   Function + `grad_reverse` helper), `lambda_p_schedule`, `DomainDiscriminator` (binary MLP:
   1024→256→128→2, BatchNorm+LeakyReLU+Dropout), and `entropy_loss`. Kept separate and
   backbone-agnostic on purpose: CLAUDE.md says DA-A "appears identically across all 3 backbones
   so cross-backbone differences are attributable to architecture, not method" — this file will
   be reused byte-for-byte by A2/A4 (PointNet++) and B2 (KPConv), and C4 (DGCNN/L-N) later.
5. **Exposed the pooled global feature from `DGCNN_ClsSeg.forward`** (`adapters/models.py`):
   added `logits["feat"] = x5_pooled` (the same 1024-dim pre-classifier vector `self.C` already
   consumes) as an additive key — C1's evaluation code, which only reads `logits["cls"]`/
   `["seg_feat"]`, is unaffected. This is the domain discriminator's input.
6. **Wrote `adapters/train_c2_dgcnn_da_a.py`**, built directly on top of `train_c1_dgcnn_da0.py`
   (same backbone, same `L_cls`/`L_seg` setup, same source data/splits/protocol) with exactly the
   adversarial machinery added on top, per CLAUDE.md's row-ordering note ("isolated,
   one-variable-at-a-time increment on top of C1 rather than a rewrite"). New pieces:
   - Loads `data/pheno4d_adaptation_pool.csv` (160 plants) as a second, target-side data loader
     — **points only**, species labels are read (required by `PlantSpeciesDataset`'s return
     signature) but never touched in the loss. This is separate from
     `data/pheno4d_heldout_eval.csv` (63 plants), still used purely as a hands-off
     epoch-by-epoch domain-gap indicator exactly as in C1/A1 — never trained on, never used for
     model selection.
   - Epoch length is source-loader-driven (32 batches at batch size 8 in the smoke test, will be
     8 batches at batch size 32 for the real run); the target adaptation loader is `cycle()`d
     against it, matching `PointDA/trainer.py`'s own `zip(src_loader, trgt_loader)` pattern for
     pairing unevenly-sized domains.
   - Single `Adam` optimizer over backbone + discriminator + Kendall log-vars — the GRL, not a
     separate optimizer/min-max alternation, is what turns one `backward()` call into both "the
     discriminator learns to detect domain" and "the backbone learns to hide it," standard DANN
     practice.
   - Model selection: identical protocol to C1/A1 — best checkpoint by lowest **source** val
     total loss (`Kendall(cls,seg)` on Crops3D val only). `L_dom`/`L_ent` are deliberately
     excluded from the selection criterion (documented in the script's own docstring): they
     involve target features/predictions and are adversarial/entropy terms, not accuracy
     measures, so a lower `L_dom` doesn't mean a better model — and including them would either
     need target labels (defeating the point) or select on an unreliable signal. Also matches
     `PointDA/trainer.py`'s own stated convention ("save model according to best source model,
     since we don't have target labels").
7. **CPU smoke test** (2026-09-11): 1 epoch, batch size 8, real Crops3D/Pheno4D data, `--gpu -1`.
   Ran end-to-end in ~12 minutes with no shape/gradient errors. All losses/metrics came back
   non-zero and in sane ranges for a single epoch (`cls loss 0.47`, `seg loss 2.24`, `dom loss
   0.73` — near `ln(2)≈0.69` as expected early on when the discriminator is still weak, `ent loss
   0.36`). Confirmed `λ_p` schedule uses `args.epochs` (not the smoke test's local epoch count)
   for its denominator, so it ramps gradually across the full 100-epoch real run rather than
   hitting 1.0 within the smoke test's single epoch by construction — the smoke test showing
   `λ_p: 0.9999` at end of epoch 0/1 is the *expected* smoke-test artifact of `total_steps =
   1*32`, not a bug; the real run's `total_steps = 100*8` ramps it properly. Deleted the smoke
   test's `results/_smoketest_c2/` output afterward (scratch, not part of the real result).
8. **Added `jobs/c2_dgcnn_da_a.sbatch`** — same cluster settings as C1's job (dgx
   partition/qos, 1 GPU, 4h wall time), 100 epochs / batch 32 / lr 1e-3 / wd 5e-5 / dropout 0.5
   (identical to C1, isolating the DA-A variable), plus `--disc_dropout 0.3 --gamma 10.0
   --lambda_ent 0.1`.

## Technical specifics

- **New files:** `adapters/dann.py`, `adapters/train_c2_dgcnn_da_a.py`, `jobs/c2_dgcnn_da_a.sbatch`
- **Modified:** `adapters/models.py` (`DGCNN_ClsSeg.forward` now also returns `logits["feat"]`)
- **Not modified:** `adapters/losses.py` (`KendallUncertaintyWeighting` reused as-is, still only
  ever sees `{"cls", "seg"}` — never `L_dom`/`L_ent`)

## What's next

Submit `jobs/c2_dgcnn_da_a.sbatch` to the cluster (100 epochs, ~4h budgeted, same wall-clock
ballpark as C1/A1). Once it lands, compare against C1 (DA-0) on the same backbone: does the
adversarial anchor actually close the target-classification gap C1/A1 both showed drifting
downward over training? That comparison is the whole point of the anchor row existing.

---
**2026-09-11:** Submitted `jobs/c2_dgcnn_da_a.sbatch` to the `dgx` partition — job **308389**.
Queue was empty beforehand (no conflicting jobs). Awaiting completion.

---
**2026-09-11 (later):** Job 308389 finished (~13.5 min wall time). Full results below, including
a deeper trajectory analysis beyond just the selected checkpoint's numbers — the headline number
alone turned out to be misleading, so both are reported.

## Results

**Protocol-selected checkpoint** (same protocol as C1/A1: best epoch by lowest source val total
loss, never touching target labels) — best epoch was **49** (source val total loss 1.0958):

| Metric | C1 (DGCNN, DA-0) | C2 (DGCNN, DA-A) |
|---|---|---|
| Best/selected epoch | 95 | 49 |
| Source val cls acc | 1.0000 | 1.0000 |
| Source val cls avg acc | 1.0000 | 1.0000 |
| Target cls acc (selected epoch) | 0.7460 | **0.9365** |
| Target cls avg acc | 0.7446 | 0.9130 |
| Tomato seg mIoU (acc) | 0.3248 (0.6645) | 0.2416 (0.5210) |
| Maize seg mIoU (acc) | 0.3427 (0.7698) | 0.1996 (0.5305) |

Read naively, this says DA-A "worked" — the selected checkpoint's target accuracy jumped from
0.75 to 0.94. **That naive reading does not survive looking at the full 100-epoch trajectory,
and the honest answer to "does DA-A close the drift" is no, not in this run.**

### Full-trajectory analysis (why the headline number is misleading)

Parsed every epoch's `Eval(no-train) - Target` line (never used for training/selection, purely
a diagnostic logged every epoch in both C1 and C2) from both run logs and compared the full
trajectories, not just the one selected checkpoint:

| Quartile (epochs) | C1 mean target acc | C2 mean target acc |
|---|---|---|
| 0–24 | 0.798 | 0.789 |
| 25–49 | 0.834 | 0.683 |
| 50–74 | 0.773 | **0.456** |
| 75–99 | 0.760 | **0.446** |
| Full-run stdev | 0.104 | **0.214** (2x C1) |
| corr(source val loss, target acc) | −0.29 | **−0.04** (~none) |

What this shows:

- **C1's trajectory is the already-documented smooth drift**: peaks early (~epoch 6–9, ~0.92),
  decays gradually, settles into a shallow plateau around 0.75–0.76 for the back half of
  training. Relatively low variance (stdev 0.10) — it's a drift, not noise.
- **C2's trajectory is not a smoothed/stabilized version of that — it's a substantially worse and
  more volatile one.** Target accuracy oscillates violently throughout the entire run (between
  ~0.37 and ~0.95, stdev more than double C1's) and never settles into any plateau. Crucially,
  in the back half of training (epochs 50–99) — which is where C1 was still holding a
  respectable ~0.76–0.77 — C2 averages only **0.456**, i.e. *worse than C1 in the same epoch
  range*, not better.
- **This instability lines up exactly with the GRL's `λ_p` schedule saturating.** Extracted
  `λ_p` per epoch from the training log: it crosses 0.9 around epoch 30 and is ≥0.98 by epoch 50
  — precisely where C2's target accuracy shifts from "volatile but still often high" (epochs
  0–49, mean 0.79→0.68) to "volatile and mostly low" (epochs 50–99, mean ~0.45). Once the
  adversarial gradient reaches full strength, training does not settle into a domain-invariant
  equilibrium — it destabilizes.
- **The selection protocol's own signal confirms this isn't a fluke of one bad quartile boundary
  choice**: `corr(source val loss, target acc)` across all 100 epochs is essentially zero for C2
  (−0.04) versus already-weak-but-present for C1 (−0.29). Source validation loss — the *only*
  signal DA-0/DA-A's model selection is allowed to look at — carries almost no information about
  where C2's target accuracy actually is at any given epoch. Epoch 49 happening to be both a
  genuine local minimum in source val loss *and* a lucky spike in target accuracy is close to
  coincidence, not a sign the selection is tracking anything real about domain alignment. (Epochs
  84–99, for comparison, have similarly low source val loss (~1.1–1.3) but target accuracy stuck
  at 0.36–0.62 — the same "good source loss" region the selector favors does *not* reliably mean
  good target accuracy for C2, unlike the somewhat-more-reliable relationship in C1.)
- **Segmentation also got worse at the selected checkpoint** (Tomato mIoU 0.24 vs C1's 0.32,
  Maize 0.20 vs 0.34): epoch 49 is much earlier in training than C1's epoch 95, and seg loss was
  still declining at that point (1.44 at epoch 49 vs ~1.15–1.2 by epoch 90+) — the selection
  protocol traded segmentation convergence for a favorable, but not representative, cls/domain
  snapshot.

**Conclusion: no, the adversarial anchor does not close or reduce the target-accuracy drift seen
in C1/A1 — in this run, under this configuration, it makes target-domain behavior markedly more
unstable, and worse on average once the adversarial term reaches full strength.** The one
strong-looking number (0.9365) is real (it's what the correct, target-label-blind selection
protocol legitimately picked), but it is not evidence of a systemically better or more
domain-invariant model — it's a high-variance process landing on a good draw. This is a known,
well-documented failure mode of vanilla DANN-style adversarial training (the same gradient
pressure that in principle pushes features toward domain invariance can equally destabilize the
shared feature extractor), not a bug in this implementation — the CPU smoke test already
confirmed the GRL/discriminator/entropy wiring itself is correct, and this behavior is consistent
with what plain, un-stabilized DANN is known to do. No code changes were made in response to this
finding: C2 is specified as the literal, unmodified anchor method (per the strategy table, this
row exists specifically so cross-backbone comparisons are attributable to architecture, not
method-level tweaks) — adding stabilization tricks (e.g. a separate/lower discriminator learning
rate, gradient clipping, spectral normalization, slower `γ`) would deviate from that spec for
this row. Worth flagging as a discussion point for the thesis write-up and as a candidate
follow-up ablation, not a required fix.

**Pass/fail:** training completed successfully with no errors, and the correct model-selection
protocol was followed — the row is done and its numbers are trustworthy *as reported*, but the
result itself is a genuine negative/cautionary finding about vanilla adversarial adaptation on
this data, not a success story.

## What's next

Row C2 is done. Per the project's row ordering, block C continues with **C3** (DGCNN,
self-supervised/DefRec, comparable to PointDA-10) next, then **C4** (DGCNN, adversarial +
L-N/jitter-noise augmentation only, isolating the noise weakness). The instability finding above
is also directly relevant to **A2** (PointNet++, same DA-A anchor method) — worth watching for
the same `λ_p`-saturation-linked destabilization there, since `adapters/dann.py` is shared
unmodified across both.

---
# Root-cause diagnosis and fix (2026-09-11, before starting C4)

Before spending more GPU time on C4 (which reuses C2's exact adversarial machinery), the user
asked for an honest read on why *neither* C2 (DA-A) nor C3 (DA-S) beat the DA-0 baseline's
full-trajectory mean, and whether anything in `dann.py`/the DefRec integration was worth
re-examining first. This section is that investigation, done by actually re-reading the logs and
data rather than speculating.

## What was verified, concretely (not assumed)

1. **Source and target have opposite class majorities.** `data/crops3d_train.csv`: Maize 192 /
   Tomato 71 (73% Maize). `data/pheno4d_adaptation_pool.csv`: Tomato 100 / Maize 60 (62%
   Tomato). `data/pheno4d_heldout_eval.csv`: Tomato 40 / Maize 23 (63% Tomato). Source and
   target don't just differ in sensor noise — their label priors point in opposite directions.
   Vanilla DANN's theory only aligns marginal *feature* distributions; it has no mechanism for a
   `P(y)` mismatch between domains, a known gap in the original formulation discussed in
   follow-up DA literature critiquing plain DANN.
2. **C2 (the original run) repeatedly collapsed to predicting the source's majority class on
   target data.** Parsed every epoch's `avg_acc`: 23 of 100 epochs land within 0.02 of exactly
   0.5 — the signature of one class at 0% recall, the other at 100%. The specific accuracy those
   epochs show is 0.3651 = 23/63, i.e. the model was predicting **all-Maize** (source's
   majority) on a target set where Maize is the *minority*. Not random noise — a systematic,
   source-biased collapse, recurring throughout the run (23 epochs, not clustered only at the
   start).
3. **Entropy loss fell from 0.32 to 0.05 over the run** (max possible for 2 classes is
   `ln(2)=0.693`). Entropy minimization was doing exactly what it's designed to do — forcing
   very confident predictions — it just was frequently confident in the wrong, source-biased
   direction. With only 2 classes, "confident" effectively means "collapsed to one class," a
   much blunter failure mode than in a many-class benchmark (e.g. PointDA-10's 10 classes),
   where the same entropy-minimization mechanism has more room to be selectively confident.
4. **`dann.py`'s own original docstring already stated the risk, but the implementation didn't
   act on it.** It reasoned that ramping entropy minimization up early "risks reinforcing
   confidently-wrong target pseudo-predictions" — and then used a *fixed* `λ_ent` from epoch 0
   anyway, applying exactly that risk at full strength before the shared features had any
   adversarial pressure behind them. Points 2–3 above show this playing out, not just being a
   hypothetical risk.
5. **Scale**: only 263 source-train / 160 target-adapt samples, batch 32 → 8 source batches / 5
   target batches per epoch, ~800 total optimizer steps across the full 100-epoch run. DANN's
   `λ_p` schedule and adversarial min-max dynamics were validated in regimes with orders of
   magnitude more steps; small-N adversarial training is independently well-known to be
   oscillation-prone. This compounds points 1–4, it doesn't replace them.
6. **Crops3D vs. Pheno4D structural asymmetry, checked not assumed**: `data/crops3d_train.csv`
   has `scan_date`/`plant_id` entirely empty (single snapshot per plant); Pheno4D's manifest has
   them populated (genuinely multi-temporal, multiple growth stages per plant, per CLAUDE.md).
   So the target domain spans an axis of variation (developmental stage) source never has any
   examples of at all — a structurally different, and arguably harder, gap than "same shapes,
   different sensor," which is what DANN/DefRec were originally validated against.

C3's DA-S result is separately explained by the DefRec paper's own ablation (both-domain DefRec
underperforms target-only — see `step_notes/C3_DGCNN_DA_S.md`) plus these same small-N/label-shift
factors; it isn't re-derived here since C3 doesn't use `dann.py` at all.

## The fix

Given the analysis above, the user chose (via an explicit decision, not a unilateral change) to:
**ramp `λ_ent` alongside `λ_p` instead of holding it at a fixed constant**, matching the
reasoning `dann.py` already stated but hadn't implemented. `adapters/dann.py` gained
`lambda_ent_schedule(p, lambda_ent_max, gamma) = lambda_ent_max * lambda_p_schedule(p, gamma)` —
rides the exact same DANN ramp as the GRL alpha. `adapters/train_c2_dgcnn_da_a.py` now computes
`lambda_ent = lambda_ent_schedule(p, args.lambda_ent, gamma=args.gamma)` per batch and uses it in
place of the old fixed `args.lambda_ent` when combining `total = task_total + l_dom + lambda_ent
* l_ent`; `--lambda_ent` is now documented as the *max* weight, reached only once `λ_p`
saturates, not the constant weight from batch 1. Both files' docstrings were rewritten (not just
appended to) to state the corrected design as current, with a "REVISED 2026-09-11" note
explaining what changed and why, per this project's style convention of flagging inferences vs.
confirmed facts.

**This changes the anchor method itself**, so per CLAUDE.md's requirement that DA-A "appears
identically across all 3 backbones" for valid cross-backbone comparison, C2 needed to be
**rerun** with the fix before C4 (which reuses the same `dann.py`) starts, rather than letting C2
and C4 diverge on an unrelated methodological fix. CPU smoke test (1 epoch, real data) passed
end-to-end with the fix before resubmitting; `λ_ent` was confirmed ramping correctly in the log
(`lambda_ent: 0.1000` at the end of the smoke test's single epoch — the same expected
end-of-ramp artifact `λ_p` showed in the original C2 smoke test, not a bug).

*(Corrected rerun numbers to be added below once the job completes.)*
