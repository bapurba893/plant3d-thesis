# Row B2: KPConv, Adversarial Domain Adaptation (DA-A), All Augmentations

## What this is, in plain language

B2 ports the project's adversarial domain-adaptation anchor method (DANN: a domain discriminator
+ Gradient Reversal Layer + entropy minimization on unlabeled target predictions) to the KPConv
backbone — the same anchor method already run identically on DGCNN (C2) and PointNet++ (A2), so
cross-backbone differences are attributable to architecture, not to the method itself.

**Why this row, specifically, next:** per explicit user instruction. B1 (KPConv, DA-0) showed the
worst model-selection-signal reliability of any DA-0 baseline trained so far across all three
backbones — `corr(source val loss, target acc) = +0.316`, *positive*, meaning a better-looking
source-domain loss actually predicted *worse* target accuracy (A1: −0.165, C1: −0.286, both
negative/informative in direction even if weak). B2 asks whether that already-poor reliability
gets worse under DANN the way both A2 and C2 showed a real (if backbone-dependent-in-magnitude)
DA-A underperformance vs. their own DA-0 baselines (see `step_notes/A2_PointNet2_DA_A.md`,
`step_notes/C2_DGCNN_DA_A.md`), or whether KPConv's already-distinct instability signature
(DGCNN-like collapse resistance, but the highest continuous variance of the three backbones)
behaves differently once adversarial pressure is added.

## Status: DONE. Job 309232 completed in 2h44m (2026-09-13) — see "Results" below. KPConv is the
first backbone where DANN does NOT show a clear DA-0→DA-A underperformance pattern.

## Design decisions

1. **`adapters/dann.py` reused byte-for-byte, unchanged** — already backbone-agnostic by design
   (operates on plain feature tensors via `logits["feat"]`, never touches model internals), the
   same way A2 already reused it unmodified from C2. Includes the ramped-`lambda_ent` fix found
   during C2's investigation (`step_notes/C2_DGCNN_DA_A.md`) — B2 starts from the corrected
   version, never the buggy v1.
2. **`adapters/train_b2_kpconv_da_a.py` mirrors `train_a2_pointnet2_da_a.py`'s DA-A training-loop
   structure** (source+target batches per step, `lambda_p`/`lambda_ent` schedule, GRL, domain
   discriminator, entropy loss, Kendall(cls,seg) + `l_dom` + `lambda_ent * l_ent` combination,
   identical model-selection protocol) with the KPConv-specific batch/collate mechanics from
   `train_b1_kpconv_da0.py` swapped in (calling `model(batch)` on a `KPConvBatch`, not
   `model(pts)` on a raw tensor; `reshape_seg_labels` for the flat-to-`(B,N)` seg-label reshape;
   `make_kpconv_collate_fn`/`make_kpconv_collate_fn_cls_only` for source/target loaders).
3. **`DomainDiscriminator(in_dim=256, ...)` — NOT 1024.** KPConv_ClsSeg's pooled bottleneck
   feature (`logits["feat"]`) is 256-wide, traced directly from `KPConv_ClsSeg.__init__`
   (`PlantKPConvConfig.first_features_dim=64`, doubled twice by the two `resnetb_strided` blocks:
   64→128→256), not assumed or copied from A2/C2's DGCNN/PointNet2 value of 1024. Using 1024 here
   would have silently mismatched the discriminator's first `Linear` layer's expected input width
   — a real, backbone-specific detail this row had to get right, not a copy-paste default.
4. **`neighborhood_limits` calibrated over BOTH domains, not source-only like B1.** B1 was a true
   DA-0 row — it never passed target points through the encoder at all, so calibrating only on
   Crops3D was sufficient. B2 (like every DA-A/DA-D/DA-O row from here on) pushes target-domain
   batches through the identical shared encoder every training step, and
   `step_notes/A3_PointNet2_DA_S.md`'s still-unverified density-sensitivity hypothesis already
   flags a real, checked (not assumed) raw point-density gap between Crops3D and Pheno4D
   (`data/dataset_statistics.csv`: ~2-13x depending on species, before FPS downsampling) —
   calibrating on source alone risks under-covering target-domain neighbor counts, which is
   exactly the failure mode that caused B1's original CUDA OOM incident
   (`step_notes/B1_KPConv_DA0.md`). Rather than assume this risk away, `adapters/
   kpconv_collate.py` gained two small, backward-compatible additions:
   - `calibrate_neighborhood_limits(..., collate_fn_factory=...)` — defaults to
     `make_kpconv_collate_fn` (unchanged behavior for B1's existing call), but can be pointed at
     `make_kpconv_collate_fn_cls_only` to calibrate against Pheno4D's cls-only 2-tuple dataset
     instead of Crops3D's cls+seg 3-tuples (the two datasets return structurally different tuples,
     so a single calibration call can't cover both without this).
   - `combine_neighborhood_limits(*limits_lists)` — elementwise max across the two domains'
     per-layer caps, so whichever domain has the wider neighbor distribution at a given layer
     determines the applied cap.
   Verified this wasn't wasted caution, not just theoretical: the CPU smoke test's actual
   calibration output was `source-only=[499, 42, 28], target-only=[499, 35, 29], combined=
   [499, 42, 29]` — target-only calibration gave a *different* (higher) cap than source-only at
   layer 2 (29 vs 28), a real, measured instance of the exact risk this design decision exists to
   cover, not a hypothetical one.
5. **Batch size kept at 16 (B1's setting), not A2/C2's 32.** KPConv's collate cost scales with
   total points in a flat batch; doubling to 32 would meaningfully increase both collate time and
   worst-case neighbor-matrix memory pressure for no benefit specific to this row (KPConv is
   density-robust by claimed design, not because of batch size). Matches B1's own precedent.
6. **`--num_workers` defaults to 0**, same as B1 — the multi-worker `DataLoader` attempt that
   made things worse on B1 (deadlock/severe slowdown with these compiled C++ extensions) is a
   backbone-level finding, not row-specific, so B2 never attempts it.
7. **`--time=60:00:00`**, the standing Block B convention from `CLAUDE.md`'s Backbones section,
   applied from the start regardless of B1's own fast (2h15m) actual runtime once it landed on an
   uncontended node — the risk that motivated the 60h budget (shared-GPU contention on the `dgx`
   partition) is about the cluster, not about B1 specifically, so there's no reason to assume B2
   will get as lucky with node placement.

## What was done and how

1. Added `collate_fn_factory` param to `calibrate_neighborhood_limits` and a new
   `combine_neighborhood_limits` helper to `adapters/kpconv_collate.py` (see design decision 4) —
   purely additive, B1's existing call site (`train_b1_kpconv_da0.py`) unaffected (default
   `collate_fn_factory=None` resolves to the same `make_kpconv_collate_fn` it always used).
2. Wrote `adapters/train_b2_kpconv_da_a.py` (see design decisions 2-3, 5-6).
3. Added `jobs/b2_kpconv_da_a.sbatch` (see design decision 7).
4. **CPU smoke test** (1 epoch, batch_size=4, real data, `--gpu -1`, `--verbose_batches`): ran
   cleanly for the full bounded window (590s) — dual-domain calibration completed correctly (see
   design decision 4's measured output), 8 training batches ran with no shape/gradient/OOM
   errors, all losses (`cls`, `seg`, `dom`, `ent`) in sane non-zero ranges. Deleted
   `results/_smoketest_b2/` afterward.

## Results (2026-09-13, job 309232, 2h44m — no contention this run either)

**Protocol-selected checkpoint** (best epoch by lowest source val total loss) — best epoch was
**65** (source val total loss 0.5045):

| Metric | B1 (KPConv, DA-0) | B2 (KPConv, DA-A) |
|---|---|---|
| Source val cls acc | 1.0000 | 1.0000 |
| Target cls acc | 0.4444 (avg 0.5625) | **0.5714 (avg 0.6440)** |
| Tomato seg mIoU (source val) | 0.3808 (acc 0.7551) | 0.2730 (acc 0.5663) |
| Maize seg mIoU (source val) | 0.4414 (acc 0.8801) | 0.3889 (acc 0.8022) |

**Full-trajectory analysis** (all 100 epochs' target-eval numbers, same methodology as every
prior row — `avg_acc == 0.5` exactly identifies a total-collapse-to-one-class epoch on this
63-sample, Tomato=40/Maize=23 target set):

| Metric | B1 (DA-0) | B2 (DA-A) | A1→A2 (DA-0→DA-A) | C1→C2 v2 (DA-0→DA-A) |
|---|---|---|---|---|
| Full-run mean | 0.629 | **0.632** | 0.599 → 0.554 (−0.045) | 0.791 → 0.633 (**−0.158**) |
| Full-run stdev | 0.203 | **0.141** | 0.196 → 0.193 (≈flat) | 0.104 → 0.117 (+0.013) |
| corr(selection-loss, target acc) | +0.316 | **+0.238** | −0.165 → +0.086 (flips, worse) | −0.286 → −0.061 (worse) |
| Collapse epochs (all-one-class) | 3/100 | 2/100 | 17/100 → 15/100 | 3/100 → 5/100 |
| Quartile means (Q1→Q4) | 0.750/0.617/0.667/0.484 | **0.733/0.554/0.589/0.651** | (declining) | 0.798→0.834→0.773→0.760 to 0.691/0.678/0.608/0.554 |
| Selected-checkpoint acc | 0.4444 | **0.5714 (+0.127)** | 0.4603 → 0.4444 (−0.016) | 0.7460 → 0.6667 (−0.079) |

### Directly answering the user's question: does B1's poor selection-signal reliability get worse under DANN, the way it did for the other two backbones, or does KPConv respond differently?

**KPConv responds differently — on nearly every axis, not just one.** Both other backbones show
a real DA-0→DA-A *underperformance* (C1→C2: −15.8 points full-run mean, the clearest case; A1→A2:
−4.5 points, smaller but same direction) and both show the selection signal getting *more*
misleading under DANN (C1's already-informative −0.286 weakens to −0.061; A1's weakly-informative
−0.165 flips sign entirely to +0.086). **B1→B2 does neither of these things:**

- **Full-run mean target accuracy is essentially unchanged** (0.629 → 0.632, +0.003) — not the
  clear decline both other backbones show. KPConv's DA-A row is, on this measure, a dead heat
  with its own DA-0 baseline, not an underperformer.
- **Selected-checkpoint accuracy actually improved** (0.4444 → 0.5714, +12.7 points) — the
  opposite direction from both A1→A2 (−1.6 points, essentially flat) and C1→C2 (−7.9 points, a
  real decline).
- **Full-run stdev decreased** (0.203 → 0.141) — DANN *stabilized* KPConv's baseline, which had
  been the least stable of the three backbones under DA-0. This is the opposite of what C1→C2
  showed (stdev rose slightly, 0.104→0.117) and unlike A1→A2 (essentially flat, 0.196→0.193).
- **The selection-signal correlation stays positive (wrong-direction) but moves toward zero**
  (+0.316 → +0.238) rather than getting worse. So the specific, narrow question — "does the
  already-bad positive correlation get more positive under DANN?" — the answer is **no, it
  improves slightly, but does not cross to the correct (negative) side.** It remains the same
  general phenomenon as B1 (a source-loss-based selection signal that is not reliable for this
  backbone), just marginally less pronounced. Contrast: A2 also lands on the wrong (positive)
  side (+0.086), so post-DANN, two of the three backbones (A2, B2) now have positive/misleading
  correlations, while only C2 stays on the informative side (albeit weakly, −0.061).

**One more real difference: the quartile trajectory shape itself is different.** B1 and every
other DA-0/DA-A row so far shows a monotonic-ish decline from an early peak (the "erode then
freeze low" pattern). B2's quartiles (0.733 → 0.554 → 0.589 → 0.651) dip in Q2 and *partially
recover* through Q3/Q4, ending on the second-highest quartile of the run, with the tightest
within-quartile variance (Q4 stdev 0.016, vs. B1's Q4 stdev 0.037) — consistent with the GRL/
entropy schedule reaching full strength (`lambda_p≈1.0`, `lambda_ent=0.1` by epoch 99) and
settling into a moderately-good, low-variance regime late in training, rather than decaying into
a low floor the way B1, A1/A2, and C1/C2 all do.

**The systematic Maize-bias also partially eased, not worsened.** B1's confusion matrix showed
Tomato recall 0.125 (35/40 misclassified as Maize); B2's shows Tomato recall 0.375 (25/40
misclassified) — still Maize-leaning, but the bias is markedly less extreme under DANN, again the
opposite direction from what a "DANN makes things worse for KPConv too" hypothesis would predict.

### Segmentation: the one place B2 clearly costs something

Source-val segmentation mIoU dropped under DA-A relative to B1's DA-0 baseline — Tomato 0.3808→
0.2730, Maize 0.4414→0.3889 — the same qualitative pattern A2/C2 both showed vs. their own DA-0
baselines (plausibly shared backbone capacity competing between `L_seg` and the adversarial/
entropy terms, per A2's step_notes). So the "adversarial training costs segmentation quality" side
finding *does* generalize to KPConv, even though the classification-transfer story does not.

### Conclusion

**KPConv is the first backbone where DANN does not show a clear DA-0→DA-A underperformance
pattern.** Every measure that showed a real cost on DGCNN (clear, C1→C2) and a smaller-but-real
cost on PointNet++ (A1→A2) — full-run mean, selected-checkpoint accuracy, systematic bias
severity — moves flat-to-positive for KPConv instead. The one place DANN costs KPConv something
real is segmentation quality, matching both other backbones. The selection-signal reliability
question specifically: B1's problem does not get *worse* under DANN, it improves marginally but
remains on the wrong (positive-correlation) side — the same general unreliability, not a new or
deepened one. Taken together with A3's finding that DA-S helps PointNet++ but hurts DGCNN, this
is now the **second row where an adaptation method's effect is genuinely architecture-dependent
in a way that isn't just "same direction, different magnitude"** — DA-A's underperformance,
previously assumed (after A2) to be a shared, dataset-driven property showing up on both
backbones just to different degrees, does not extend to KPConv at all. Whether this is really
about KPConv's architecture (e.g. its density-robust grid-subsampled neighborhoods producing
features less prone to the kind of biased, over-confident adversarial collapse DGCNN/PointNet++
show) or a re-run of B1's own already-unusual instability signature landing differently under a
new loss combination is not distinguished by this row alone — worth keeping in mind for B4
(KPConv, DA-0, L-D only), which offers another, differently-motivated read on this backbone's
behavior without adversarial training in the mix.
