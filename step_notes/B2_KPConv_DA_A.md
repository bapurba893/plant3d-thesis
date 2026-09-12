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

## Status: submitted to the cluster (2026-09-13), job 309232 on `cn19-dgx`

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

## Results

*(pending — job 309232 submitted 2026-09-13, `--time=60:00:00`, awaiting completion)*
