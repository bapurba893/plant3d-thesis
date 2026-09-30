# KPConv eval-time nondeterminism (found 2026-09-30, while building report figures)

Zero-context-reader summary: like PointNet++ (see `step_notes/PointNet2_Eval_Nondeterminism.md`),
the vendored KPConv backbone is NOT deterministic at eval time even with weights frozen — but
for TWO separate reasons, neither of which is the PointNet2 mechanism (KPConv has no
farthest-point sampling at all). Both were found, isolated, and fixed in the same working
session. This affects every Block B row (B1-B5, B3b) whenever a checkpoint is re-evaluated in a
fresh process (as opposed to read once from that row's own original training run.log).

## The first fix attempt was incomplete — flagged, not hidden

The first fix (adding `torch.manual_seed(EVAL_SEED)` after `model.eval()`) was verified against
this project's own reproducibility bar and FAILED it: 4 repeated calls on the same frozen B1
checkpoint still gave 0.5079/0.4921/0.4921/0.4286 after that fix — proof the fix was incomplete,
not a false alarm. Investigated further rather than accepted as "good enough."

## Root cause 1: `calibrate_neighborhood_limits`'s own calibration DataLoader

`adapters/kpconv_collate.py::calibrate_neighborhood_limits` builds a calibration `DataLoader`
with `shuffle=True` and no seed of its own — every call samples a different random subset of
batches to build its neighbor-count histogram, which can produce a different
`neighborhood_limits` cap per layer. This consumes torch's global RNG. Fixed by moving
`torch.manual_seed(EVAL_SEED)` to BEFORE this call (the original, incomplete fix had it AFTER —
too late).

## Root cause 2: kernel-point random rotation (the one that actually mattered)

`KPConv-PyTorch/kernels/kernel_points.py::load_kernels` (vendored, unmodified upstream) loads or
generates a BASE set of kernel-point positions (cached to a `.ply` file after the first
generation, per-`(num_kpoints, fixed, dimension)` combination), but then ALWAYS applies a FRESH
RANDOM ROTATION to that base pattern on every call:

```python
theta = np.random.rand() * 2 * np.pi
...
kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape)
```

This runs inside the KPConv layer's `__init__` — i.e. on every `KPConv_ClsSeg(...)` construction,
independent of which trained weights are subsequently loaded. The rotated kernel positions are
NOT part of the saved `state_dict` (they're recomputed geometry, not a learned parameter), so
loading a checkpoint's weights does nothing to make this deterministic. Critically, this uses
**NumPy's global RNG** (`np.random.rand`/`np.random.normal`), NOT torch's — so the first fix
(`torch.manual_seed` only) had no effect on this source at all, exactly matching the observed
"still varies after the first fix" symptom.

## Quantified measurement

Two batches, both on B1's checkpoint (weights frozen, only the eval process differs):

1. **Both bugs present** (fully unseeded): 4 repeats gave `[0.5238, 0.5079, 0.4603, 0.4921]` —
   range 0.0635 (~4/63 samples).
2. **Calibration seeded, kernel rotation NOT yet fixed** (isolates root cause 2 alone): 4 repeats
   gave `[0.5079, 0.4921, 0.4921, 0.4286]` — range 0.0793 (~5/63 samples).

Both magnitudes are comparable to (slightly larger than) PointNet2's own measured range
(0.032-0.079, see the sibling step_notes file) — this is a real, non-trivial source of noise on
this project's small (n=63) target test set, not a rounding-level artifact.

## Verified fix

Both `torch.manual_seed(EVAL_SEED)` (before calibration) AND `np.random.seed(EVAL_SEED)` (before
model construction — placed at the same point, before calibration, since it must also precede
`KPConv_ClsSeg(...)`) are required together. Verified, not just asserted: 4 repeated calls with
DIFFERENT ambient torch/numpy RNG states before each call (`torch.manual_seed(i*137)`,
`np.random.seed(i*991)` for `i` in 0..3, deliberately varied to prove the function's OWN internal
seeding — not a lucky starting state — is what fixes it) all returned exactly `0.5397`.

## Does this change any previously-reported number? No — same reasoning as the Oracle bug

Every Block B row's OWN training script (`train_b1_kpconv_da0.py` etc.) calibrates
`neighborhood_limits` and constructs its model exactly ONCE, at the start of that single training
run, then uses that one instance consistently for training, validation, and the final target
eval — all within one process, one kernel-point rotation, fully self-consistent. This
nondeterminism only becomes visible when a checkpoint is loaded into a FRESH process later (a
new report-figure script, or a manual re-run) that reconstructs the model from scratch. Every
number currently cited in CLAUDE.md/PROGRESS_LOG/other step_notes came from each row's own
original single-process run and is unaffected.

## Practical consequence for this project's report figures

`scripts/plot_confusion_matrices.py` and `scripts/tsne_feature_plot.py`'s KPConv paths now
reproduce a fixed seed's numbers exactly, but that seeded number will generally NOT exactly match
the row's own originally-logged run.log value (different kernel rotation than whatever the
original training run happened to draw) — expected, not a bug, exactly the same situation
already documented for PointNet2. Confusion-matrix/t-SNE figures for B1/B3b/B5 were regenerated
after this fix and may show accuracy numbers that differ slightly (within the measured ~0.06-0.08
range) from each row's headline reported number; captions/figures should note this is expected
seeded-reproduction variance, not a discrepancy to resolve.
