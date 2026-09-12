# Row A3: PointNet++, Self-Supervised Domain Adaptation (DA-S), All Augmentations

## What this is, in plain language

A3 is the DGCNN-track's C3 row, ported to the PointNet++ backbone: same self-supervised
deformation-reconstruction method (DefRec-style — deform a region of the point cloud, train the
model to reconstruct it via Chamfer distance, on BOTH source and target domains per CLAUDE.md's
DA-S definition), same augmentation mix, same data/protocol — only the backbone changes.

**Why this row, specifically, before Block B:** per explicit user instruction, C3 was DGCNN's
least-bad domain-adaptation result so far — its full-run mean target accuracy (0.703) sits closer
to C1's no-adaptation baseline (0.791, an 8.8-point gap) than either DA-A variant did (C2: 15.8
points; C4: 17.8 points — see `step_notes/C4_DGCNN_DA_A_LN.md`). A3 tests whether "DA-S is the
closest of the three methods tried so far" is a genuine, backbone-general property of the method,
or just something true of DGCNN specifically — the same kind of cross-backbone question A2 already
asked of DA-A (see `step_notes/A2_PointNet2_DA_A.md`: DA-A's underperformance direction replicated
on PointNet++ but the magnitude was much smaller, because A1's own baseline was already unstable).
Answering this before adding a third backbone (KPConv, Block B) matters more right now because the
current two-backbone picture is still incomplete on this specific question.

## Status: submitted to the cluster (2026-09-12)

## Design decisions

1. **DefRec's core machinery (`DefRec_and_PCM.DefRec.deform_input`/`.calc_loss`,
   `utils.pc_utils`) is fully backbone-agnostic** — it operates on raw point tensors and reads a
   generic `logits['DefRec']` key, never touching the backbone's internals directly. Reused
   unchanged from C3, including the chunked-Chamfer-distance OOM workaround
   (`DEFREC_CHUNK=8`/`defrec_chunked_backward`) — PointNet2 produces the same dense `(B, 4096,
   ...)` per-point tensors DGCNN does, so the same memory pressure that OOM'd an 80GB A100 on
   C3's first attempt applies here too; no reason to expect otherwise, so the fix was carried
   over proactively rather than waiting to hit the same OOM.
2. **One real gap needed filling: PointNet2_ClsSeg had no reconstruction head at all** (A1/A2
   never needed one). Unlike A2's adversarial machinery (which had to be built from scratch,
   backbone-agnostically, in `adapters/dann.py`), PointNet++'s DefRec support only needed
   composing ONE existing reference-repo class it hadn't used yet:
   `PointDA.Models.RegionReconstruction` — the exact same class `DGCNN_ClsSeg` already uses via
   its `DGCNN` base class, a generic per-point Conv1d stack over an arbitrary input feature
   width (identical in kind to `PointSegDA.Models.segmentation`, already reused for both
   backbones' seg heads). Added to `adapters/models_pointnet2.py::PointNet2_ClsSeg`:
   - `self.DefRec = RegionReconstruction(args, seg_input_size)` in `__init__` (same
     `seg_input_size=1152` already used for the seg heads).
   - `forward(..., activate_DefRec=True)` now sets `logits["DefRec"] = self.DefRec(logits["seg_feat"])`
     — reusing the already-computed `seg_feat` tensor (per-point features concatenated with the
     pooled global feature) as DefRec's input, since that's exactly DefRec's expected input shape.
     This is a minor efficiency improvement over DGCNN's own `forward`, which builds an identical
     tensor a second time from scratch for `DefRec_input` — not a behavioral difference, just
     avoiding a redundant duplicate computation this repo's composition doesn't need to repeat.
   - No reference-repo code modified — same "adapt this repo" pattern as everything else in this
     project.
3. **`adapters/train_a3_pointnet2_da_s.py` is a near-line-for-line copy of
   `train_c3_dgcnn_da_s.py`**, with `DGCNN_ClsSeg`/`model_args.model="dgcnn"` swapped for
   `PointNet2_ClsSeg`/plain `model_args` (same swap pattern A1 used for C1, A2 used for C2) — same
   data, same loss architecture (Kendall over cls/seg/defrec), same DefRec hyperparameters
   (`DefRec_dist=volume_based_voxels`, `num_regions=3`, `DefRec_weight=1.0` neutral), same
   model-selection protocol.
4. **Comparison baseline is A1** (`results/A1_pointnet2_da0_clsseg/run.log`), the same way C3
   compares against C1 and A2 compares against A1 — all four rows (A1/A2/A3, C1) share the
   identical `pheno4d_heldout_eval.csv` target test set/protocol.
5. **Full-trajectory analysis from the start**, per explicit user instruction, same methodology
   as A2/C2/C3/C4.

## What was done and how

1. Added `RegionReconstruction` import + `self.DefRec` + the `activate_DefRec=True` branch to
   `adapters/models_pointnet2.py::PointNet2_ClsSeg` (see design decision 2). Purely additive —
   A1/A2 never set `activate_DefRec=True`, so they're unaffected.
2. Wrote `adapters/train_a3_pointnet2_da_s.py` (see design decision 3).
3. Added `jobs/a3_pointnet2_da_s.sbatch` — same cluster settings as A1/A2/C3 (dgx partition/qos,
   1 GPU, 4h wall time), both `DEFREC_ROOT` and `POINTNET2_ROOT` exported (needed for the
   PointNet++ backbone, same as A1/A2's jobs).
4. **CPU smoke test** (1 epoch, batch_size=8, real data, `--gpu -1`): ran end-to-end in ~9.5
   minutes (slower than A1/A2's smoke tests, expected — Chamfer distance is the most expensive
   op in this whole project's training loops, same as C3's own smoke test) with no shape/gradient
   errors. `RegionReconstruction` composed correctly (no shape mismatch feeding it PointNet2's
   1152-wide `seg_feat`), all losses non-zero and in sane single-epoch ranges (`defrec loss`
   substantially larger than `cls`/`seg` loss, expected — same pattern C3 showed). Deleted
   `results/_smoketest_a3/` afterward.
5. Added `jobs/a3_pointnet2_da_s.sbatch`.

---
**2026-09-12:** Submitted `jobs/a3_pointnet2_da_s.sbatch` to the `dgx` partition — job **308789**.
Queue was empty beforehand. Awaiting completion.

---
## Results (2026-09-12, job 308789, ~33 min wall time)

**This is the first row in the entire project where a domain-adaptation method clearly beats its
own backbone's no-adaptation baseline, on the full-trajectory reading, not just a lucky
checkpoint.** Given how much this differs from every prior adaptation result (C2, C3, C4, A2 all
underperformed their DA-0 baseline), the numbers below were checked for leakage/bugs before being
taken at face value — see the "Sanity checks" section.

**Protocol-selected checkpoint** (best epoch by lowest source val total loss) — best epoch was
**95** (source val total loss 1.5076):

| Metric | A1 (DA-0) | A3 (DA-S) |
|---|---|---|
| Target cls acc | 0.4603 (avg acc 0.5750) | **0.8889** (avg acc 0.9033) |
| Tomato seg mIoU (source val) | 0.3207 (acc 0.6729) | **0.0594** (acc 0.0853) |
| Maize seg mIoU (source val) | 0.4741 (acc 0.9055) | 0.3404 (acc 0.7015) |

**Full-trajectory analysis** (all 100 epochs' target-eval numbers, same methodology as
A2/C2/C3/C4):

| Metric | A1 (DA-0) | A3 (DA-S) | C1 (DA-0, ref) | C3 (DA-S, ref) |
|---|---|---|---|---|
| Full-run mean target acc | 0.599 | **0.800** | 0.791 | 0.703 |
| Full-run stdev | 0.196 | 0.158 | 0.104 | 0.104 |
| corr(source val loss, target acc) | −0.165 | −0.009 | −0.286 | +0.222 |
| Quartile means (Q1→Q4) | .676/.717/.507/.495 | **.851/.683/.780/.885** | .798/.834/.773/.760 | .731/.718/.707/.657 |
| Epochs collapsed to one class | 17/100 | **2/100** [24, 30] | 3/100 | not re-checked |
| Min / max target acc | 0.365 / 0.968 | 0.365 / 0.968 | 0.365 / 0.921 | — |

### Sanity checks (given how surprising this result is)

- **No target-label leakage**: `train_a3_pointnet2_da_s.py` discards target species labels the
  same way C3/A2 do (`_trgt_species_unused = trgt_batch[1]`, never referenced again) — this is
  the same, already-vetted code path C3 used without producing a similarly dramatic result, so
  the mechanism isn't new or untested.
- **Model selection never touches target data**: `val_total_loss` combines
  `Kendall(cls, seg, defrec)` computed entirely on Crops3D val (source); `evaluate_defrec` is
  also called with `src_val_loader` only, matching C3's protocol exactly.
- **The improvement is not a single-epoch fluke**: full-run mean (0.800) is close to the
  selected-checkpoint number (0.8889) and every quartile except Q2 is well above A1's
  corresponding quartile (Q1 0.851 vs 0.676, Q3 0.780 vs 0.507, Q4 0.885 vs 0.495) — a real,
  mostly-consistent improvement across nearly the whole run, not a spike right before the
  selected epoch.
- **Collapse-epoch count corroborates it independently**: A3 collapses to predicting one class
  in only 2/100 epochs vs. A1's 17/100 — a large, structural stability improvement that doesn't
  depend on how accuracy itself is read.

### A genuine, real tradeoff found alongside the improvement: Tomato segmentation collapses

Reading the full per-epoch Tomato seg mIoU trajectory (not just the FINAL line): it starts around
0.25-0.28 in the first 2-3 epochs, then collapses to a 0.02-0.09 range for essentially the entire
rest of training (mean well under 0.1) — a real, sustained degradation, not a selected-checkpoint
artifact. **Maize segmentation does NOT show this pattern** — it starts low (~0.03-0.16) and
steadily *improves* to a 0.34-0.39 range by the end (mean 0.277), a healthy training curve.

So A3's story is not simple "DA-S is unambiguously better on PointNet++" — it is
**a real classification-transfer improvement bought at a real cost to Tomato-species
segmentation specifically.** Plausible contributing factor, flagged as an inference (not verified
by ablation): Tomato is already the minority species in Crops3D training data (71/263 = 27%) and
has a severely imbalanced organ-label distribution (the "fruit" class is present in only 12% of
Tomato files, per CLAUDE.md's Crops3D labeling note) — under joint training with strong,
competing `cls`+`defrec` gradients (and Kendall's learned weighting visibly favoring `cls`: by
epoch 99, `s_cls=-0.415` (heavily up-weighted) vs. `s_seg=0.243`/`s_defrec=0.192`, both
down-weighted relative to `cls`), Tomato's already-fragile segmentation may be the first casualty
of DefRec's added training pressure. Maize (73% of training data, better-represented organ
classes) doesn't show the same fragility.

### Answering the user's question: does "DA-S is DGCNN's least-bad result" hold on PointNet++?

**No — it doesn't just fail to hold, it flips direction entirely, and dramatically so.** On
DGCNN, DA-S underperforms DA-0 (C1 0.791 → C3 0.703, an 8.8-point full-run-mean decline; 17.5
points at the selected checkpoint). On PointNet++, DA-S substantially *improves* over DA-0 (A1
0.599 → A3 0.800, a 20.1-point full-run-mean gain; 42.9 points at the selected checkpoint) — the
opposite direction, and a much larger swing.

This is a genuinely different finding from A2's (DA-A): A2 showed the *same direction* as C2 on
both backbones, just a smaller magnitude on PointNet++ (see `step_notes/A2_PointNet2_DA_A.md`).
A3 shows the *opposite direction* from C3. **DA-A's underperformance looks like a
dataset-general property (shows up on both backbones, differing only in how clearly the
already-noisy PointNet++ baseline lets it show); DA-S's effect looks architecture-dependent in a
much stronger sense — it actively helps one backbone and actively hurts the other.**

A plausible (unverified) explanation: A1's baseline instability (17/100 collapse epochs,
systematic Maize-bias documented in CLAUDE.md's A1 entry) may stem from PointNet++'s
farthest-point-sampling/ball-query set-abstraction encoder learning geometric features that are
less robust to the sensor-domain shift than DGCNN's dynamic-graph features are by default. DefRec's
self-supervised reconstruction task, forced on both domains, directly trains the encoder to
recover clean geometry from deformed input — plausibly compensating for exactly the kind of
representational fragility that made A1 so unstable in the first place. DGCNN's C1 baseline had
much less of that fragility to begin with (stdev 0.104, 3/100 collapse epochs), so DefRec had
less room to help and, per C3's own investigation, instead diluted the model-selection signal
(`corr(source val loss, target acc)` flipped from C1's −0.286 to C3's +0.222 — actively wrong
direction) without a compensating stability gain. This mirrors, in reverse, A2's finding that a
noisier baseline changes how a DA method's effect shows up — here, PointNet++'s noisier baseline
is precisely what DA-S has room to fix, whereas DGCNN's stabler baseline gives DA-S nothing to
fix and only a diluted selection signal to pay for it. **This is an inference about mechanism, not
a verified one** — no ablation isolates "DefRec's contribution to feature robustness" from other
factors.

### Conclusion

DA-S's usefulness is real but strongly architecture-dependent on this dataset — it substantially
improves classification transfer on PointNet++ (reversing C3's DGCNN-track finding entirely) while
also introducing a genuine, sustained cost to Tomato-species segmentation quality that DGCNN's
DA-S run didn't show at anywhere near this severity. Both directions are worth reporting together:
this is not a case where one backbone's result can be assumed to generalize to the other, for
either the direction or the magnitude of DA-S's effect — a materially different conclusion than
A2 found for DA-A (same direction, different magnitude only).

### Proposed mechanism for the reversal (2026-09-12) — NOT VERIFIED, logged as a hypothesis only

Discussed with the user after this row landed; captured here so it isn't lost, but explicitly
flagged per CLAUDE.md's "flag inferences vs. confirmed facts" rule — nothing below was tested by
an ablation, only reasoned from existing project facts plus one new data check (the raw density
numbers).

**The hypothesis**: DefRec's region-deformation-reconstruction pretext (redraw a region's points
from a local Gaussian, train the backbone to reconstruct clean geometry from context) is
fundamentally a "learn features robust to locally perturbed point density/arrangement" exercise.
CLAUDE.md names a different specific architectural weakness for each backbone: PointNet++'s is
"fixed-radius ball query is density-sensitive," DGCNN's is "unrestricted k-NN lets noisy points
become spurious graph edges." DefRec's pretext task maps far more directly onto fixing the first
than the second — a dynamic k-NN graph always finds exactly k neighbors regardless of local
density, so it's comparatively insensitive to exactly the kind of perturbation DefRec practices
against, while a fixed real-world-radius ball query (radius=0.2/0.4 in
`adapters/models_pointnet2.py`) is precisely the kind of operation that perturbation would
stress-test and improve. On this view, DefRec has a real architectural weakness to fix on
PointNet++ and comparatively little to fix on DGCNN — consistent with A1's baseline being far
less stable to begin with (17/100 collapse epochs vs. C1's 3/100) and DefRec both stabilizing it
(A3) and, on DGCNN's already-stable baseline, only diluting the selection signal without a
compensating gain (C3's `corr` flipping from C1's −0.286 to +0.222).

**Supporting data point checked, not merely asserted**: `data/dataset_statistics.csv` shows a
real, large RAW point-density gap between domains before any downsampling — Crops3D Maize
averages ~82K points vs. Pheno4D Maize's ~1.1M (~13x), Tomato is smaller but still a 2-3x gap.
This is concrete evidence the sensor/domain gap has a real density component for this dataset,
not just a hypothetical one.

**Caveat that keeps this at "plausible," not "likely confirmed"**: checked `scripts/
preprocessing.py` and found both domains go through voxel pre-decimation *then* true
farthest-point sampling (FPS) down to the shared target_n=4096 — FPS specifically maximizes
spatial coverage, which partially (not necessarily fully) equalizes local point spacing
regardless of the original raw density. So the 13x raw gap likely overstates how different the
final *cached* N=4096 clouds actually are; FPS may not fully erase sensor-specific fine-scale
sampling artifacts (structured-light/RGB-D grid-like patterns vs. laser-triangulation scan lines),
but this wasn't measured directly.

**One thing this hypothesis rules out**: it cannot be merely "DA-S exposes the backbone to
target points during training that DA-0 never sees" as the differentiator, since C3 (DGCNN) runs
the identical both-domains DefRec protocol and shows no comparable benefit — the target-domain
exposure is identical in both C3 and A3, so whatever differs has to be about *which* weakness the
pretext task happens to counteract, not mere exposure.

**The "PointNet++ had more room to be stabilized" framing (the user's own alternative/companion
hypothesis) is treated here as the statistical symptom of the same mechanism, not a competing
explanation** — if A1's instability is itself caused by ball-query behaving inconsistently under
whatever residual density/sampling difference survives FPS, then "more room to improve" and "DefRec
fixes PointNet++'s specific weakness" are the same explanation at two different levels (statistical
vs. mechanistic), not two alternatives to choose between.

**Future work, not run now (deprioritized in favor of starting Block B — breadth over depth given
the deadline)**:
1. Measure actual local neighbor-spacing/density statistics directly on the final cached N=4096
   point clouds, per domain and per species, to check whether a real residual density-texture
   difference survives the voxel+FPS pipeline (would directly test the "supporting data point"
   above rather than relying on pre-downsampling raw counts).
2. An ablation swapping PointNet++'s fixed-radius ball-query grouping for a k-NN-based grouping
   (matching DGCNN's neighbor-selection strategy) and rerunning A3, to see whether the DA-S
   benefit shrinks or disappears once the specific named weakness this hypothesis targets is
   removed.

Block B (KPConv) is a natural, complementary angle on the same question without spending more
budget on dedicated ablations: KPConv's own claimed strength is density robustness via
grid-subsampled neighborhoods rather than a fixed point count (see CLAUDE.md's Backbones
section) — directly relevant to this hypothesis, and worth observing organically as Block B
proceeds rather than as a purpose-built test.
