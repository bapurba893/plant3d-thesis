# PointNet++ eval-time nondeterminism (found 2026-09-30, while building report figures)

Zero-context-reader summary: the vendored PointNet++ backbone (`Pointnet_Pointnet2_pytorch`,
unmodified upstream, see CLAUDE.md's Reference codebase section) is NOT deterministic at eval
time, even with the model in `.eval()` mode and weights frozen, because its set-abstraction
layers pick a RANDOM starting point for farthest-point sampling on every forward call. This
affects every Block A row (A1-A6) — measured directly, not just theorized — and this file pins
down exactly how much it matters before any single-run PointNet++ comparison is trusted.

## Root cause

`Pointnet_Pointnet2_pytorch/models/pointnet2_utils.py::farthest_point_sample`:

```python
farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
```

Every call samples a fresh random starting centroid via the ambient (process-global) torch RNG
state — there is no seeding inside this function, and `model.eval()` only disables
dropout/switches BatchNorm to running stats; it does NOT make FPS deterministic. `PointNet2_
ClsSeg.forward` calls this 3 times per forward pass (`sa1`/`sa2`/`sa3`, see
`adapters/models_pointnet2.py`). Neither DGCNN (uses k-NN graph over ALL points, no downsampling
at all) nor KPConv (grid subsampling — deterministic spatial binning, not random point selection)
use farthest-point sampling anywhere in their architecture, so this specific mechanism is
believed NOT to apply to either other backbone — a reasoned inference from reading both
architectures, not something separately verified by direct repeated-eval measurement the way
this file does for PointNet++ below. Flagged as an inference, not a confirmed fact, per this
project's own discipline.

## How it was found

Building `scripts/plot_confusion_matrices.py`'s A3 confusion matrix produced a target test
accuracy of 0.9048 — but `results/A3_pointnet2_da_s/run.log`'s own originally-logged value is
0.8889 (already committed, already cited throughout CLAUDE.md/step_notes as A3's headline
result). Same checkpoint file, same held-out eval set, same code path (`model.eval()`,
`torch.no_grad()`) — a genuine discrepancy, not explained by a weight-loading bug (checked: no
unexpected/missing state_dict keys were reported for A3 beyond the already-understood DefRec.*
mismatch, see `_load_state_dict_lenient`'s warning mechanism in `scripts/tsne_feature_plot.py`).
0.8889→0.9048 on n=63 is exactly a 1-sample flip (1/63=0.0159), consistent with a single
borderline prediction changing due to a different random FPS draw, not a systematic error.

## Quantified measurement (2026-09-30)

10 repeated evaluations of the SAME frozen checkpoint (weights never change across repeats;
`torch.manual_seed(i)` for `i` in 0..9 set once before each repeat, so each repeat's FPS draws
are reproducible individually but different from each other — deliberately NOT holding one fixed
seed throughout, to sample the actual spread a naive, unseeded re-run would show):

| Row | Mean acc | Std | Min | Max | Range (≈ samples flipped, n=63) |
|---|---|---|---|---|---|
| A1 (DA-0) | 0.4524 | 0.0146 | 0.4286 | 0.4762 | 0.0476 (≈3 samples) |
| A3 (DA-S) | 0.8937 | 0.0102 | 0.8730 | 0.9048 | 0.0317 (≈2 samples) |
| A6 (DA-D) | 0.5603 | 0.0266 | 0.5079 | 0.5873 | 0.0794 (≈5 samples) |

**The originally-logged, already-reported value for each row falls inside its own measured
range** (A1: 0.4603, appears 4/10 times in the repeat set; A3: 0.8889, appears 4/10 times; A6:
0.5714, appears 2/10 times) — every previously-reported number is a typical, representative draw
from this distribution, not an outlier or a sign of a deeper bug. Nothing already reported is
contradicted by this finding.

## What this does and does NOT call into question

**Does NOT meaningfully affect**: the full-trajectory statistics (full-run mean, stdev, collapse-
epoch counts) that are this project's PRIMARY basis for every "this backbone is
unstable/stable" claim. Those come from ~100 independent per-epoch evaluations across a single
training run, where the model's WEIGHTS genuinely change epoch to epoch — e.g. A1's own
documented full-run stdev is 0.196, roughly 7-13x larger than the pure eval-noise std measured
here (0.015-0.027) on a fixed checkpoint. Under an independence assumption, eval noise this size
would contribute at most ~1-3% of A1's observed variance — the vast majority of every documented
PointNet++ trajectory's instability is genuine training-dynamics behavior, not this artifact.

**DOES require a caveat**: any comparison between two PointNet++ rows' SELECTED-CHECKPOINT
accuracy where the gap is small (roughly within 1-3 samples / 0.016-0.048 on this n=63 test set).
Concretely, re-reading this project's own already-published Block A selected-checkpoint deltas
with this in mind:
- A1→A2 (0.4603→0.4444, Δ=-0.0159, exactly 1 sample): **within noise, should not be read as a
  real difference** on its own — though A2's separate full-trajectory mean (0.554 vs. A1's 0.599)
  is a 100-epoch-averaged number, far more robust to this effect, and that comparison stands.
- A1→A4 (0.4603→0.5397, Δ=+0.0794, ≈5 samples): borderline — sits at the edge of A6's own
  measured range width, plausibly still a real effect given A4's full-trajectory story
  independently supports the same direction, but a single-run selected-checkpoint number alone
  would not be strong evidence by itself.
- A1→A3 (0.4603→0.8889, Δ=+0.4286, ≈27 samples) and A1→A5 (0.4603→1.0000, Δ=+0.5397, ≈34
  samples): **far beyond this noise band, fully robust conclusions**, unaffected.
- A1→A6 (0.4603→0.5714, Δ=+0.1111, ≈7 samples): A6 itself showed the WIDEST measured spread of
  the three checkpoints tested (std 0.0266, range 0.0794) — this delta is about 1.4x A6's own
  full measured range, so plausibly a real effect but with less margin than the large A3/A5 gaps;
  the cross-backbone DA-D verdict in step_notes/A6_PointNet2_DA_D.md leans primarily on the
  full-trajectory mean (+0.050, a 100-epoch-averaged, noise-robust number) rather than the
  selected-checkpoint delta alone, which is the more defensible way to have stated it.

## Practical fix for future report figures — IMPLEMENTED and verified

`scripts/plot_confusion_matrices.py` and `scripts/tsne_feature_plot.py` now call
`torch.manual_seed(<fixed value>)` immediately after `model.eval()` in every PointNet2 path, so
regenerating a given figure later reproduces the SAME confusion matrix/t-SNE embedding every time
(run-to-run reproducibility of the figure itself). Verified directly: `confusion_pointnet2` on
A3's checkpoint returned 0.9048 identically across 3 repeated calls after this fix (previously
varied per call). This does not "fix" the underlying nondeterminism (a different seed still gives
a different, equally valid draw), it only makes a
given figure's exact numbers reproducible on demand.
