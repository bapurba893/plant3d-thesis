# Illustrated Write-Up: Figure Generation Log

Tracks the illustrated report being built for the professor (training curves, t-SNE plots,
confusion matrices per row) — one entry per figure/design decision as it's produced, same
zero-context-reader bar as every other step_notes file. Report scope/prioritization decided
2026-09-29 (see PROGRESS_LOG.md for the plain-language summary): a curated set of headline rows
(DA-0 baseline / DA-A anchor / each backbone's best adaptation method / Oracle ceiling, per
backbone), not the full illustrated treatment for all 22 trained rows.

## Design decision: grayscale marker/fill scheme for t-SNE plots (confirmed 2026-09-29)

Per CLAUDE.md's style rule ("Documents/reports: black and white only, no color"), t-SNE plots
need two categorical dimensions (species, domain) encoded WITHOUT color. Validated on a single
example (`scripts/tsne_feature_plot.py`, row C1/DGCNN/DA-0) before committing to generating the
full set, per explicit user instruction:

- **species → marker shape**: circle = Tomato, triangle = Maize
- **domain → marker fill**: solid black = source (Crops3D), hollow/white with black edge = target
  (Pheno4D)

User-confirmed legible, including in the plot's one dense overlapping cluster (target Tomato).
**This scheme is now the standard for every t-SNE plot in the report** — no further per-plot
design review needed.

## Bug: Oracle rows' segmentation-head config, found and fixed 2026-09-30

**What was wrong**: `scripts/plot_confusion_matrices.py`'s first version constructed every row's
model with `seg_num_classes=SEG_NUM_CLASSES` (the Crops3D-shaped scheme — Tomato=3, Maize=6)
unconditionally. Oracle rows (A5/B5/C5) don't use that scheme at all — per CLAUDE.md's DA-O
description and `adapters/dataset.py::PHENO4D_SEG_NUM_CLASSES`'s own docstring, Oracle training
trains directly on Pheno4D's OWN per-point organ labels, collapsed to a DIFFERENT 3-class
soil/stem/leaf scheme for BOTH species (`{"Tomato": 3, "Maize": 3}`), a different label space
entirely from Crops3D's. Each Oracle row's segmentation head shape (`seg_heads.Maize.conv4`,
etc.) was therefore saved with a different width than the script assumed.

**Which rows are affected**: only A5/B5/C5 (the three Oracle rows) — every other row (A1-A4/A6,
B1-B4/B3b, C1-C4/C6) always uses the Crops3D-shaped `SEG_NUM_CLASSES`, correctly, with no config
branching needed.

**How it was caught**: `model.load_state_dict(..., strict=False)` (via
`tsne_feature_plot.py::_load_state_dict_lenient`) raised a hard `RuntimeError` immediately on A5
— `size mismatch for seg_heads.Maize.conv4.weight: copying a param with shape torch.Size([3,
128, 1]) from checkpoint, the shape in current model is torch.Size([6, 128, 1])` — a loud,
unambiguous failure, not a silent one. The script's `main()` loop has no try/except around each
row, so the whole run halted at A5; B5/C5 (also in the confusion-matrix script's row list) were
NEVER REACHED, so no incorrect number was ever produced for any of the three Oracle rows.

**Does this change any previously-reported number? No.** Confirmed by reading each Oracle row's
OWN training script directly (`adapters/train_a5_pointnet2_da_o.py`,
`train_b5_kpconv_da_o.py`, `train_c5_dgcnn_da_o.py`) — all three already construct their model
with `seg_num_classes=PHENO4D_SEG_NUM_CLASSES` correctly; this was always right in the actual
training code that produced every Oracle number currently cited in CLAUDE.md/PROGRESS_LOG/other
step_notes files. The bug existed ONLY in this brand-new, not-yet-finished report-figure script
(`plot_confusion_matrices.py`), written and caught within the same working session, before it
ever produced a single figure or logged number for any Oracle row. Nothing downstream was
silently corrupted because the bug never got far enough to silently produce anything — it failed
loudly on the very first Oracle row it touched.

**Fix**: `_seg_num_classes_for(dirname)` — returns `PHENO4D_SEG_NUM_CLASSES` when the row's
results-directory name starts with `A5_`/`B5_`/`C5_`, `SEG_NUM_CLASSES` otherwise. Not yet
re-run against B5/C5 with the fix as of this writing (A1/A3 confusion matrices were already
generated successfully before this bug surfaced on A5; the fixed script still needs a full re-run
across all 8 curated rows, including a re-run of A5 itself, before any confusion-matrix figures
are treated as final) — tracked as the next step before finalizing the confusion-matrix set.

## Related, separately-documented findings: eval-time nondeterminism (PointNet++ AND KPConv)

While root-causing the Oracle bug above, the A1/A3 confusion-matrix numbers this script produced
were independently checked against each row's own already-logged `run.log` accuracy and found to
differ slightly (A3: 0.9048 here vs. 0.8889 logged). This is NOT a second instance of the Oracle
bug above (unrelated code path, no state_dict mismatch was reported) — it is a genuine, separate
finding about the vendored PointNet++ backbone being non-deterministic at eval time even with
weights frozen. Fully measured and written up in `step_notes/PointNet2_Eval_Nondeterminism.md`.

**A second, mechanistically distinct instance of the same class of problem was then found in
KPConv** while fixing the first one and re-verifying against this project's own reproducibility
bar (a fix that doesn't actually reproduce on repeated calls doesn't count as fixed) — TWO
separate unseeded-RNG sources (`calibrate_neighborhood_limits`'s calibration `DataLoader`, and
the vendored KPConv-PyTorch kernel-point layer's random rotation, which uses NumPy's global RNG,
not torch's, and required a second, separate fix after the first `torch.manual_seed`-only attempt
was verified to NOT resolve it). Full investigation, both root causes, and the final verified fix
in `step_notes/KPConv_Eval_Nondeterminism.md`.

**Both fixes are now actually wired into the inference paths `plot_confusion_matrices.py` and
`tsne_feature_plot.py` use** (`torch.manual_seed`/`np.random.seed` set immediately before
calibration and model construction in every KPConv function, immediately after `model.eval()` in
every PointNet2/DGCNN function) — verified by repeated-call tests for both backbones, not just
asserted. DGCNN needs no such fix (no known eval-time randomness source in that architecture,
confirmed by its confusion-matrix numbers matching `run.log` exactly).

**Correction to an earlier claim about the B1-vs-B3b t-SNE pair**: before this investigation, the
B1/B3b t-SNE plots (generated unseeded, same as everything else at that point) were described as
showing source and target Tomato "sitting directly adjacent" under B3b (DA-S) — read as a clean
visual confirmation of the quantitative win. After regenerating both with the verified fix
(`results/report_figures/tsne_B1_kpconv_da0.png`, `tsne_B3b_kpconv_da_s.png`), **that specific
description does not hold as cleanly**: B3b's Tomato source and target now form two distinct
clusters, not one merged one (Maize, by contrast, does form one clean, tightly-separated cluster
in the corrected B3b plot — visibly better species separation than B1's more scattered picture).
Flagged here explicitly rather than left as a stale claim: the honest reading of the corrected
figure is that DA-S visibly improves species (Tomato vs. Maize) separability in feature space,
not that it visibly merges the two domains' Tomato distributions into one — a real, more modest
claim than what was said in chat/the original commit message for that figure. The KPConv
kernel-rotation randomness (see `step_notes/KPConv_Eval_Nondeterminism.md`) plausibly explains
why the unseeded draw looked more merged than a "typical" draw — a single random kernel geometry
can shift the fine embedding structure even when the coarse classification behavior is stable;
this has not been separately verified by running multiple seeds to check how consistent the
"two distinct clusters" reading is, and should be treated as one data point, not a fully
characterized finding, the same caution now applied to any single seeded t-SNE draw for this
backbone.

## Figure 1: C1 (DGCNN, DA-0) — source vs. target feature space

`results/C1_dgcnn_da0_clsseg/tsne_example.png`, generated by `scripts/tsne_feature_plot.py`.
t-SNE of the pooled 1024-dim global feature (`logits["feat"]`, pre-classifier) for 45 Crops3D
(source) validation scans and all 63 Pheno4D (target) held-out eval scans, from C1's trained
checkpoint (no retraining — checkpoint was already on disk from the original training run,
gitignored but not deleted).

**Observation, flagged explicitly per user instruction as a real finding, not just a visual
detail**: Maize source and target features land in the SAME cluster (substantial overlap), while
Tomato source and target features separate into two distinct, non-overlapping sub-clusters. This
is species-asymmetric domain-gap evidence visible directly in feature space, under DA-0 (no
adaptation at all).

**Why this is independent, corroborating evidence, not a restatement of an already-known
finding** (the distinction the user asked to make explicit): every prior species-asymmetry finding
in this project came from either (a) classification OUTCOMES — e.g. A1's confusion matrix showing
34/40 target Tomato plants misclassified as Maize, B1's Tomato recall of 0.125 vs. Maize recall
1.00, both "systematic Maize-bias" — or (b) growth-curve/measurement-level diagnostics — D0's
finding that 4/7 real Tomato plants hadn't plateaued within Pheno4D's scanning window (K hitting
its bound) and that `stem_diameter` fits poorly across the board, later showing up again in D5/D6
as `stem_diameter`'s physics-loss-ablation delta being noise-level while `height`'s wasn't. Neither
of those is a feature-space observation. C2's own investigation (`step_notes/C2_DGCNN_DA_A.md`)
diagnosed a MECHANISM for the classification-outcome asymmetry — Crops3D (source) is 73% Maize
while Pheno4D (target) is 62-63% Tomato, an opposite-majority label-prior mismatch that made
entropy minimization collapse toward the source's majority class in some epochs — but that
diagnosis was also derived from training dynamics/outcomes, not from directly looking at where
the two domains' learned features land relative to each other.

**This t-SNE plot is a different kind of evidence for the same underlying asymmetry**: it shows,
directly in the trained model's own representation space, that Tomato's source and target
distributions are further apart from each other than Maize's are — a geometric/representational
signature that is consistent with (not merely a repeat of) both the label-prior-mismatch mechanism
(C2) and the Tomato-specific data-quality/heterogeneity issues (D0, D5/D6). Three independent
lines of evidence (classification outcomes, growth-curve/measurement diagnostics, and now feature-
space geometry) now converge on the same conclusion: something about Tomato specifically — not
just the domain gap in general — is harder for this project's models to handle consistently across
the Crops3D→Pheno4D sensor gap. Flagged as convergent evidence across independent analyses, not
proof of a single root cause — the three lines of evidence are consistent with several
non-exclusive contributing mechanisms (label-prior mismatch, real biological non-plateau
heterogeneity, stem_diameter measurement noise) rather than isolating one.
