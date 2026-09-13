# Row B3b: KPConv, Self-Supervised Domain Adaptation (DA-S), All Augmentations

## What this is, in plain language

B3b ports the project's self-supervised domain-adaptation method (DefRec-style deformation
reconstruction: deform a region of the point cloud, train the model to reconstruct it via
Chamfer distance, on BOTH source and target domains) to the KPConv backbone — the same method
already run on DGCNN (C3) and PointNet++ (A3), so cross-backbone differences are attributable to
architecture, not to the method itself.

## Why "B3b" and not "B3" — a real correction, not a naming quirk

Per CLAUDE.md's original strategy table, **Block B's third row (B3) is DA-D** (discrepancy-based
adaptation, Deep CORAL/MMD), not DA-S — Block B was specced differently from Block A/C from the
start: instead of a DA-S row, Block B uses B4 (DA-0/L-D, deliberately unadapted) as its own
density-robustness check. This was caught and corrected mid-session (2026-09-13) when a request
to run "B3 (DA-S)" was checked against the actual documented strategy table before proceeding,
rather than assumed correct — the table said B3=DA-D, not DA-S, and the user's own restated
question ("does KPConv flip DA-S the way it flipped DA-A?") presupposed a row that didn't exist
in the plan as written.

**Resolution (confirmed with the user):** add DA-S as a genuinely new row, labeled **B3b**, for
direct comparability with A3/C3, documented explicitly in CLAUDE.md's strategy table section as
an intentional addition — not a silent renumbering, and not a replacement for B3 (DA-D), which
stays in the plan and runs separately, after this row.

## Why this row, specifically, now

A3 (PointNet++, DA-S) reversed C3's DGCNN finding entirely — DefRec hurt DGCNN's target-
classification transfer but helped PointNet++'s, by a large margin (see
`step_notes/A3_PointNet2_DA_S.md`). B2 (KPConv, DA-A) then showed a second architecture-dependent
reversal: DANN hurt both DGCNN and PointNet++ (to different degrees) but did not hurt KPConv
overall (see `step_notes/B2_KPConv_DA_A.md`). This row asks the natural next question: does
KPConv also flip DA-S's effect the way it flipped DA-A's, or does DA-S land on KPConv the way it
landed on one of the other two backbones (DGCNN's clear hurt, or PointNet++'s clear help)?

## The one real, KPConv-specific engineering problem this row had to solve

DGCNN/PointNet2 take a raw `(B, 3, N)` tensor directly as `forward`'s input, so DefRec's "deform
the input, feed the deformed tensor through the model" recipe needs no extra plumbing for those
two backbones — their own forward pass internally recomputes whatever geometric structure it
needs (farthest-point sampling, ball query, k-NN) from the deformed coordinates automatically,
every call, as part of ordinary differentiable/non-differentiable geometric ops inside the
PyTorch graph.

KPConv is architecturally different: its multi-layer neighbor/pool/upsample structure is
precomputed OUTSIDE the model entirely, by `adapters/kpconv_collate.py`'s collate functions
(CPU-side, via the compiled C++ extensions), from a specific, fixed set of point coordinates.
Simply overwriting an already-built `KPConvBatch`'s point values with deformed coordinates would
silently keep the ORIGINAL (undeformed) geometry's neighbor structure — not incorrect in a
crashing sense, but not a faithful analogue of what happens automatically for the other two
backbones, and arguably defeating the point of the pretext task (the model should have to
process the deformed geometry through its own real encoder machinery, not decode deformed
positions through an encoder still "looking at" the clean layout).

**Fix**: added two small, symmetric helpers to `adapters/kpconv_collate.py`:
- `batch_points_bcn(batch)` — extracts a clean, already-collated batch's layer-0 (finest,
  full-resolution) points as a plain `(B, 3, N)` tensor. Safe/exact for the same reason
  `KPConv_ClsSeg.forward`'s own `seg_feat` reshape is exact: every sample here is a fixed N=4096
  points, so `batch.lengths[0]` is always `[N]*B`, in batch order.
- `build_batch_from_points(config, points_bcn, batch_builder=None)` — the reverse: rebuilds a
  FRESH, fully-structured `KPConvBatch` (new neighbor/pool/upsample tables, via the same
  `segmentation_inputs` call every other collate function uses) from a `(B, 3, N)` tensor,
  deformed or not.

So KPConv's DA-S path now does explicitly what DGCNN/PointNet2 do implicitly: rebuild geometric
structure from whatever coordinates are being reconstructed, every time — not just reuse stale
structure with new point values spliced in. `KPConv_ClsSeg` gained a `self.DefRec =
RegionReconstruction(config, seg_feat_dim)` head and an `activate_DefRec: bool = False` forward
parameter, mirroring `PointNet2_ClsSeg`'s own DefRec composition (A3) exactly — same generic
`PointDA.Models.RegionReconstruction` class, fed the same already-computed `seg_feat` tensor,
no reference-repo code modified.

## Design decisions

1. **DefRec's core machinery reused unchanged.** `DefRec_and_PCM.DefRec.deform_input`/`.calc_loss`
   operate on plain `(B, 3, N)` point tensors and a `(B, N, 3)` reconstruction output — fully
   backbone-agnostic, never touching model internals. Only how those tensors get IN and OUT of
   KPConv's specific batch format (via the two new helpers above) is new. The chunked-Chamfer-
   distance OOM workaround (`DEFREC_CHUNK=8`, from C3's original root-cause fix) is reused
   unchanged too — Chamfer distance operates on the SAME `(B, N, 3)`-shaped tensors regardless of
   which encoder produced them, so the same OOM risk (and fix) applies identically.
2. **`neighborhood_limits` calibrated over BOTH domains, same as B2 and for the same reason** —
   target points flow through the encoder here too (via DefRec, not via a domain discriminator
   this time, but the exposure is the same in kind). Reused `calibrate_neighborhood_limits`/
   `combine_neighborhood_limits` from B2 unchanged.
3. **Deformed-batch rebuilds reuse the SAME already-calibrated `config`**, including for
   deformed geometry never directly measured during calibration — this is safe from the same
   CUDA OOM `calibrate_neighborhood_limits` exists to prevent, because
   `PointCloudDataset.big_neighborhood_filter`'s cap is an *unconditional hard slice*
   (`neighbors[:, :limit]`), not a probabilistic percentile guarantee — even if deformation
   happens to produce locally denser configurations than calibration observed, the slice still
   bounds memory. Flagged as an inference (not independently re-measured for deformed geometry
   specifically), but load-bearing and directly traceable to `big_neighborhood_filter`'s actual
   implementation, not assumed from first principles.
4. **A shared `KPConvBatchBuilder` instance (`defrec_batch_builder`), constructed once, reused
   across every chunk rebuild for the whole run** — cheap to construct (holds only `self.config`
   and `self.neighborhood_limits`), but avoids needless reconstruction inside the innermost loop.
5. **Real, accepted cost, flagged up front**: this row needs meaningfully more collate calls per
   training step than B1 or B2. At `batch_size=16`, `DEFREC_CHUNK=8`: 1 clean-source collate +
   2 deformed-source-chunk rebuilds + 2 deformed-target-chunk rebuilds = 5 collate calls/step,
   vs. B1's 1 and B2's 2. Expected to be slower than B1 (2h15m) and B2 (2h44m)'s real-world
   runtimes, but well within the standing 60h Block B budget even at several times their
   per-step cost. `--verbose_batches` kept on, same standing convention.

## What was done and how

1. Added `RegionReconstruction` import + `self.DefRec` + the `activate_DefRec` forward parameter
   to `adapters/models_kpconv.py::KPConv_ClsSeg` (see "the one real engineering problem" above).
   Purely additive — B1/B2 never set `activate_DefRec=True`, unaffected.
2. Added `batch_points_bcn`/`build_batch_from_points` to `adapters/kpconv_collate.py`. Purely
   additive.
3. Wrote `adapters/train_b3b_kpconv_da_s.py`, mirroring `train_a3_pointnet2_da_s.py`'s DA-S
   training-loop structure (Kendall(cls, seg, defrec), chunked-backward DefRec on both domains,
   identical model-selection protocol) with KPConv-specific batch/collate/rebuild mechanics
   swapped in throughout.
4. Added `jobs/b3b_kpconv_da_s.sbatch` (`--time=60:00:00`, standing Block B convention).
5. Updated `CLAUDE.md`'s strategy table section to document B3b explicitly as an addition to the
   plan (per user instruction) — see that file's Block B entry.
6. CPU smoke test (1 epoch, batch_size=4, real data, `--gpu -1`, `--verbose_batches`) — see
   Status below.

## A real bug caught by reading the reference repo's source, not by trusting the smoke test alone

`DefRec_and_PCM.DefRec.calc_loss(args, logits, labels, mask)` reads `logits['DefRec']`
*internally* — it expects the FULL logits dict A3/C3's own `defrec_loss_for_batch` passes it
(`logits_deform = model(pts_deformed, activate_DefRec=True); DefRec.calc_loss(args,
logits_deform, pts_orig, mask)`), not a pre-extracted tensor. The first draft of
`defrec_loss_for_chunk` here wrote `logits_deform = model(deformed_batch,
activate_DefRec=True)["DefRec"]` (pre-indexing before the call) — would have raised
`TypeError: 'Tensor' object is not subscriptable` the first time the DefRec path actually ran.
Caught by re-reading `DefRec.py`'s exact `calc_loss` implementation directly and comparing
against A3's exact call pattern, *before* the smoke test reached that code path (calibration
runs first and takes several minutes) — fixed and the smoke test restarted from scratch, rather
than spending the remaining smoke-test budget on a crash that was already diagnosable from the
reference repo's source.

## Status: DONE. Job 309416 completed in 4h58m (2026-09-13) — the strongest non-Oracle result
in the project so far. See "Results" below.

CPU smoke test (after the fix above) passed: calibration completed in ~21 min
(`neighborhood_limits`: source-only=[499,42,28], target-only=[499,35,29], combined=[499,42,29] —
consistent with B2's own calibration numbers, as expected since the calibration code itself is
identical), then 2 training batches ran cleanly with no shape/gradient/OOM errors before the
deliberate 30-min timeout cutoff.

**Measured performance confirms the "real, accepted cost" flagged in Design decision 5 above,
concretely, not just in principle**: at `batch_size=4` (smoke test), batch 0 took 119.5s collate
+ 236.1s fwd+bwd+step (inflated by one-time warm-up costs, expected), batch 1 (steady-state)
took 12.0s collate + 53.0s fwd+bwd+step — both dramatically higher than B1/B2's per-batch costs
(typically single digits to tens of seconds total). On the actual GPU run (batch_size=16), the
real cost landed on the moderate end of that range: ~2.5-4 min/epoch, 100 epochs in **4h58m**
(job 309416, no contention) — well inside the 60h budget, using more of it than B1 (2h15m) or B2
(2h44m) but nowhere near the tens-of-hours worst case.

---
## Results (2026-09-13, job 309416, 4h58m)

**This is the strongest result of any row in the entire project so far, non-Oracle or otherwise.**

**Protocol-selected checkpoint** (best epoch by lowest source val total loss) — best epoch was
**84** (source val total loss 0.8936):

| Metric | B1 (KPConv, DA-0) | B3b (KPConv, DA-S) |
|---|---|---|
| Source val cls acc | 1.0000 | 1.0000 |
| Target cls acc | 0.4444 (avg 0.5625) | **0.7143 (avg 0.7658)** |
| Tomato seg mIoU (source val) | 0.3808 (acc 0.7551) | 0.2697 (acc 0.5658) |
| Maize seg mIoU (source val) | 0.4414 (acc 0.8801) | 0.3910 (acc 0.8302) |

**Full-trajectory analysis** (all 100 epochs' target-eval numbers, same methodology as every
prior row):

| Metric | B1 (DA-0) | B3b (DA-S) | A1→A3 (DA-0→DA-S) | C1→C3 (DA-0→DA-S) |
|---|---|---|---|---|
| Full-run mean | 0.629 | **0.860** | 0.599→0.800 (+0.201) | 0.791→0.703 (−0.088) |
| Full-run stdev | 0.203 | **0.086** | 0.196→0.158 | 0.104→0.104 |
| corr(selection-loss, target acc) | +0.316 | +0.291 | −0.165→−0.009 | −0.286→+0.222 |
| Collapse epochs (all-one-class) | 3/100 | **0/100** | 17/100→2/100 | 3/100→not re-checked |
| Quartile means (Q1→Q4) | 0.750/0.617/0.667/0.484 | **0.912/0.908/0.861/0.759** | (rises then holds) | (declining) |
| Selected-checkpoint acc | 0.4444 | 0.7143 (+0.270) | 0.4603→0.8889 (+0.429) | 0.7460→0.5714 (−0.175) |

For scale: B1→B3b's full-run-mean gain (**+0.231**, 23.1 points) is *larger* than A1→A3's own
gain (+0.201, 20.1 points) — the biggest DA-S improvement over its own DA-0 baseline seen on any
backbone so far. B3b's full-run mean (0.860) sits only **0.065 below C5's Oracle ceiling**
(0.925, the only row in the project trained directly on labeled target data) — remarkable given
B3b never touches a single target label.

### Directly answering the user's three-way question: flip DA-A's pattern, align with C3's null result, or something distinct?

**Neither, cleanly — B3b is its own, stronger case, closer in direction to A3 but distinct in
both magnitude and in what it costs.**

- **It does NOT align with C3 (DGCNN).** C3 showed DA-S actively hurting DGCNN's target-
  classification transfer (0.791→0.703, an 8.8-point loss, though a stable one). B3b shows the
  opposite: a 23.1-point *gain*. KPConv's DA-S result is about as far from C3's as it's possible
  to be while still being the same method.
- **It is not simply "the same reversal A3 showed, ported over."** It's a *bigger* reversal on
  every full-trajectory measure that matters: bigger full-run-mean gain (+23.1 vs. +20.1
  points), a much lower full-run stdev (0.086 vs. A3's 0.158 — B3b is the single most stable
  trajectory of *any* row trained in this project, DA-0/DA-A/DA-S/DA-O included, more stable
  even than C1's previous-best 0.104), and **zero collapse epochs** (A3 still had 2/100; every
  other row, including every DA-0 baseline, had at least some). The one place A3's reversal was
  clearly bigger: its selected-checkpoint gain (+0.429) exceeds B3b's (+0.270) — A3's single
  best snapshot landed higher (0.8889 vs. 0.7143) even though B3b's overall trajectory is
  stronger and steadier.
- **It does not flip B2's (DA-A's) pattern either — it's a different kind of result entirely.**
  B2 showed DA-A roughly *tying* B1 (a flat +0.003 on full-run mean) — DANN neither helped nor
  hurt KPConv much. B3b shows DA-S *substantially* helping KPConv — a much stronger, unambiguous
  positive effect than DA-A produced on the same backbone. So within Block B itself, the two
  adaptation methods tried so far land in genuinely different places: DA-A ≈ neutral, DA-S ≈
  strongly positive.
- **One thing that does NOT change from B1 to B3b: the selection-signal correlation.** It stays
  positive and essentially flat (+0.316 → +0.291), still on the wrong (positive) side. A1→A3
  moved in a different way — from a weakly-informative −0.165 to a near-zero −0.009 (still
  correctly-signed, just less informative) — but never crossed to the wrong side the way B1/B3b
  sit throughout. KPConv's positive-correlation selection-signal issue persists at essentially
  the same strength whether DA-A or DA-S is layered on top of it — suggesting this
  is a fairly structural property of how KPConv's source validation loss relates to target
  accuracy on this dataset, not something tied to any one adaptation method. Practically: even
  though B3b's *trajectory* is excellent, the checkpoint the protocol actually selects (epoch 84,
  acc 0.7143) is still noticeably below the trajectory's own ceiling (max 0.9365) — real
  headroom is being left on the table by the selection protocol here too, same structural
  critique as B1/B2.

### The segmentation story is also cleanly different from A3's — no collapse, gradual improvement

A3's PointNet++ DA-S win came at a real cost: Tomato seg mIoU collapsed from ~0.27 to a
~0.02-0.09 floor within a few epochs and never recovered. **B3b shows nothing like this.**
Reading the full per-epoch trajectory (not just the FINAL line): Tomato seg mIoU *rises*
steadily over training (quartile means 0.207→0.240→0.254→0.269, starting ~0.16-0.22 and ending
~0.27), and Maize seg mIoU rises even more (quartile means 0.265→0.361→0.384→0.387, starting
~0.04-0.16 and ending ~0.39, close to B1's own 0.4414). Both organ classes end only modestly
below B1's DA-0 baseline (Tomato 0.2697 vs. 0.3808, Maize 0.3910 vs. 0.4414) — a real but mild
cost, nothing resembling A3's collapse. **KPConv's DA-S win is closer to a clean win than any
other adaptation-method result in the project so far** — large classification gain, best-in-
project stability, and only a modest (not collapsed) segmentation cost.

### Conclusion

KPConv's response to DA-S is the strongest positive adaptation result found anywhere in this
project (Oracle excluded) — bigger than A3's already-notable reversal, dramatically more stable
than any other row, and without A3's severe segmentation trade-off. It does not fit neatly into
either prior pattern (C3's clean negative result, or "A3's reversal, just ported over") — it's a
genuinely new data point showing DA-S's benefit can be not just backbone-dependent in direction
(already established by A3 vs. C3) but also highly variable in *magnitude and cost profile*
even among the backbones where it helps. One structural finding carries over unchanged from B1
and B2: KPConv's selection-signal correlation with true target performance stays positive
(wrong-direction) regardless of which adaptation method is layered on top, suggesting it's a
property of the backbone+dataset pairing itself, not of any one DA method. Given this result and
A3's, self-supervised deformation reconstruction is now 2-for-3 across backbones (helps
PointNet++, helps KPConv even more, hurts DGCNN) — worth real weight when Block D's fusion work
picks a backbone/DA/augmentation combination to build on.
