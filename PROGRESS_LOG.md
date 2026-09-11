# Progress Log

A plain-language, chronological record of this project's progress — for people, not for an
AI assistant. (For dense technical context, see `CLAUDE.md`.) Each section below is a real
milestone: a finished training run, a bug found and fixed, a task completed. Not every small
step gets an entry — only things that actually moved the project forward or changed our
understanding of it.

---

## 1. Data Foundation (Stage 0)

Both datasets — Crops3D (the labeled source, Tomato + Maize, structured-light/RGB-D sensors)
and Pheno4D (the unlabeled target, Tomato + Maize, laser triangulation scanner) — were
downloaded, checked for completeness, and preprocessed into a single cached format the
training code can load quickly.

Along the way: Pheno4D's `.xyz`/`.txt` files needed a parsing fix (different files have
different column counts depending on whether they're annotated and which species they are),
and the point-sampling step used during preprocessing (farthest-point sampling, a standard way
to downsample a point cloud to a fixed size while keeping it representative) was too slow in
its original form and had to be sped up.

End result: 531 files preprocessed with zero failures, cached as `.npz` files plus manifest
CSVs, all committed to the repo (~25MB — small enough to track directly, unlike the raw PLY/
txt files which stay outside the repo).

## 2. Cluster & GPU Setup

Development happens on a laptop with no GPU, so real training needed a GPU machine. Got access
to IIT Bombay's Prajna HPC cluster and confirmed an A100 80GB GPU actually works end-to-end
through the cluster's job submission system (`sbatch`, not interactive sessions — this
account isn't allowed to run interactive GPU sessions, only submitted batch jobs). Confirmed
the environment has PyTorch 2.6.0 with CUDA 12.4 support.

## 3. Base Codebase Integrated

Rather than writing domain-adaptation training code from scratch, we're building on an
existing, published research codebase (Achituve et al., WACV 2021 — "DefRec", self-supervised
domain adaptation for point clouds) that already implements the training protocols this
project needs. It's kept as an unmodified external dependency (like a library), not copied
into our repo, so it's clear what's "borrowed" versus what's ours.

Before adapting it to our own data, we first ran the reference codebase's own built-in example
(a standard benchmark: ModelNet → ShapeNet) completely unchanged, to confirm the environment
and integration were solid. It ran clean end-to-end on the cluster GPU. Only after that did we
start adapting it to Crops3D/Pheno4D.

## 4. First Real Result: Row C1 (DGCNN, DA-0)

Trained the first real strategy-table row: DGCNN backbone (one of our three model
architectures), no domain adaptation (the "no adaptation" lower-bound baseline), classification
only at this stage (is a plant a tomato or maize plant?).

Found and fixed a real bug: with plain cross-entropy loss, the model was taking a shortcut —
just always predicting the majority class (maize, which outnumbers tomato roughly 2.7 to 1 in
the training data) — and still scoring deceptively well by accuracy alone. Fixed by weighting
the loss by inverse class frequency, which forced the model to actually learn to distinguish
the two species rather than guess the majority.

## 5. Finding the Real Segmentation Labels

The next goal was per-point segmentation (labeling each point in a plant as, e.g., leaf vs.
stem vs. fruit), not just whole-plant classification. The original plan pointed at a dataset
variant called `Crops3D_IS`, but investigating it turned up two problems: it has no Tomato
plants at all, and its labels are the wrong granularity anyway (which *plant* a point belongs
to, not which *organ* of the plant).

The real answer was hiding in plain sight: the base Crops3D files we'd already downloaded and
already used for the classification result contain an unused per-point property (`scalar_sf`)
that turns out to be exactly the organ-category label we needed. No published source (paper or
either GitHub mirror) documents what the numeric codes mean by name, so we use the raw numeric
IDs as class labels rather than guessing names.

Confirmed by scanning all 308 cached Crops3D files: Tomato has 3 label classes, Maize has 6.
Two of the label meanings were confirmed by color-checking the raw point colors (soil is
visibly brown; the most common label per species is leaf); the rest remain unconfirmed by name.

## 6. Row C1 Retrained with Full Segmentation

Retrained row C1 with the complete intended loss (classification + segmentation combined, with
a learned weighting between the two rather than a fixed guess). Segmentation quality landed
around 0.32–0.34 mIoU (a standard segmentation quality score; higher is better, 1.0 is
perfect) for both species.

One number looked concerning at first: classification accuracy on the unlabeled target dataset
dropped compared to an earlier interim run (0.89 → 0.75). Investigated it properly rather than
shipping the number as-is: it turns out that under "no adaptation" training, accuracy on the
target dataset quietly erodes the longer training continues, even while everything the model
*is* allowed to check (source-domain performance) keeps looking better and better. This isn't
a bug — it's the expected, and even useful, behavior of a no-adaptation baseline: it has no way
to notice or prevent this drift, which is exactly the problem the next step (an adversarial
domain-adaptation method) is designed to solve.

## 7. PointNet++ Integration — 2026-09-11

Started work on the second model architecture (PointNet++), which required adapting an
external, well-established implementation since the base reference codebase only comes with
the first two architectures (PointNet, DGCNN) built in.

Checked it over carefully before trusting it, the same way the base codebase was checked in
Milestone 3: confirmed it runs on our current software versions, and — importantly — confirmed
it needs no special compiled GPU code to work (one of our three planned architectures, KPConv,
will need that later, so this was worth checking rather than assuming). Hit one real snag along
the way — a naming collision between one of the external library's internal files and one of
our own files, which caused confusing import errors — tracked it down and fixed it.

Verified the integration works correctly with a full trial run on real data (on the laptop's
CPU, since it just needed to prove correctness, not be fast) before considering it done, then
handed it off to the cluster to actually train for real.

## 8. Row A1 Trained — First Cross-Backbone Comparison — 2026-09-11

Row A1 (PointNet++, no adaptation) finished training on the cluster GPU, same 100-epoch
setup, same data split, same "no adaptation" strategy as row C1 — so this is the project's
first real apples-to-apples comparison between two of the three model architectures.

| Metric | C1 (DGCNN) | A1 (PointNet++) |
|---|---|---|
| Source classification accuracy | 1.00 | 1.00 |
| Target (unseen dataset) classification accuracy | 0.75 | 0.46 |
| Tomato segmentation quality (mIoU) | 0.32 | 0.32 |
| Maize segmentation quality (mIoU) | 0.34 | 0.47 |

Both architectures fit the source data perfectly (expected — that's not the hard part). The
interesting split shows up on the parts that are actually hard:

- **PointNet++ transfers noticeably worse to the unseen (target) dataset for whole-plant
  classification** — it isn't just noisier, it's systematically biased: it misclassifies most
  Tomato plants as Maize on the target set, something DGCNN doesn't do nearly as much.
- **PointNet++ segments Maize plants distinctly better** than DGCNN, while the two are
  essentially tied on Tomato segmentation.
- The same "quiet erosion over training" pattern seen in Milestone 6 for DGCNN showed up again
  here for PointNet++ (best target accuracy came very early in training, then declined even as
  source-side numbers kept improving) — confirming this is a general property of "no
  adaptation" training, not something specific to one architecture. That strengthens the case
  for the next step: adding an adversarial domain-adaptation method (rows A2/C2) to see if it
  fixes this erosion for both architectures.

## 9. Adversarial Domain Adaptation Built and Submitted — 2026-09-11

Started the next milestone: teaching the model to actively make its features look the same
regardless of which dataset they came from, instead of just hoping a plain model happens to
generalize (which Milestones 6 and 8 showed doesn't hold up over long training). This uses a
well-known technique (DANN — adversarial domain adaptation): a small second network tries to
guess whether a set of features came from Crops3D or Pheno4D, while the main network is trained
to fool it, plus a separate nudge that encourages confident (not wishy-washy) predictions on the
unlabeled target data. None of this existed in the base reference codebase (which only
implements a different technique, self-supervised reconstruction) — checked carefully before
concluding that and building it from scratch, the same way the "no adaptation" baseline training
loop had to be built from scratch in Milestone 4.

Verified with a full trial run on real data (CPU, one epoch) before trusting it, same as every
previous integration step — ran cleanly with no errors and sane numbers. Then handed off row
**C2** (DGCNN + adversarial adaptation, the "anchor" row every other adversarial row in the table
will copy) to the cluster GPU to train for real (job 308389). Result not back yet.

---

## Current Status: the 24-Row Strategy Table

The full experiment plan is a 24-row table (5 blocks: three model architectures each tested
under several domain-adaptation strategies, plus a fusion block and an optional deployment
block). Full detail lives in `docs/Domain_Invariance_Strategy_Table.docx`; this is just the
at-a-glance status.

| Block | Row | What it is | Status |
|---|---|---|---|
| A (PointNet++) | A1 | No adaptation (baseline) | **Done** — full result committed |
| A (PointNet++) | A2 | Adversarial (anchor method) | Not started |
| A (PointNet++) | A3 | Self-supervised | Not started |
| A (PointNet++) | A4 | Adversarial, cropping/dropout augmentation only | Not started |
| A (PointNet++) | A5 | Oracle (upper-bound reference) | Not started |
| B (KPConv) | B1 | No adaptation (baseline) | Not started |
| B (KPConv) | B2 | Adversarial (anchor method) | Not started |
| B (KPConv) | B3 | Discrepancy-based adaptation | Not started |
| B (KPConv) | B4 | Deliberately unadapted, cropping/dropout only | Not started |
| B (KPConv) | B5 | Oracle (upper-bound reference) | Not started |
| C (DGCNN) | C1 | No adaptation (baseline) | **Done** — full result committed |
| C (DGCNN) | C2 | Adversarial (anchor method) | Training on cluster (job 308389) |
| C (DGCNN) | C3 | Self-supervised | Not started |
| C (DGCNN) | C4 | Adversarial, jitter-noise augmentation only | Not started |
| C (DGCNN) | C5 | Oracle (upper-bound reference) | Not started |
| D (fusion) | D1–D6 | Combining the best backbone/strategy with growth-curve features | Not started |
| E (deployment, optional) | E1–E3 | Model compression / distillation | Not started |

**In one sentence:** two rows (A1, C1) are fully done, giving the first real cross-backbone
comparison, and the other 22 rows are not started yet.

---

*Update this file with a new dated section after each real milestone — not every small step.*
