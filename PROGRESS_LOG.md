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
will copy) to the cluster GPU to train for real (job 308389).

## 10. Row C2 Trained — Does Adversarial Adaptation Actually Help? — 2026-09-11

Job 308389 finished. The headline number looks great at first glance: the model checkpoint the
project's (correct, hands-off-of-target-labels) selection process picked scores 0.94 accuracy on
the unseen target dataset, way up from C1's 0.75. Taken at face value, that would mean
adversarial adaptation fixed the "quiet erosion" problem from Milestones 6 and 8.

**It doesn't hold up under closer inspection, and reporting it as a win would have been
misleading.** Looking at target accuracy across the *entire* 100-epoch run (not just the one
selected checkpoint) tells a different story: instead of stabilizing, target accuracy swings
wildly throughout training — between roughly 0.37 and 0.95 — and in the second half of training
it averages only 0.46, actually *worse* than C1's no-adaptation baseline over the same stretch
(0.77). That instability starts almost exactly when the adversarial training pressure reaches
full strength partway through the run. The one great-looking number the selection process landed
on turns out to be a lucky snapshot from a highly volatile process, not evidence of a genuinely
better or more stable model — a statistical check confirmed the model-selection signal (which
only ever looks at source-side performance, since target labels aren't allowed) carries almost no
information about where target accuracy actually is at any given point in this run.

**So: no, this first attempt at adversarial domain adaptation does not close or reduce the
target-accuracy drift.** If anything, it trades a slow, predictable decline (C1) for a faster,
much more volatile one that's worse on average later in training. This is a known, well-studied
failure mode of this class of adversarial technique when used without extra stabilization tricks
— not a bug in the implementation (which was checked carefully before the real run, same as every
other integration step). It's a genuine, useful negative finding, and one worth carrying into how
the next adversarial rows (A2 on PointNet++, C4 with different augmentation) get interpreted.

## 11. Row C3 Started — Self-Supervised Adaptation, a Third Way to Try Closing the Gap — 2026-09-11

Started the third domain-adaptation approach: instead of a discriminator fighting the network
(Milestone 10) or just hoping (Milestones 6/8), this one gives the network a self-supervised
puzzle — hide part of a plant's point cloud and make the network reconstruct it — run on *both*
datasets at once, so the "get good at reconstructing plant geometry" pressure doesn't care which
dataset a plant came from. Unlike the adversarial method, the machinery for this technique
already existed ready-made in the base reference codebase (it's literally that codebase's own
main contribution), so this was more a matter of correctly wiring it into our own
classification+segmentation setup than building something new from scratch.

This row also happens to be the one most directly comparable to a well-known external published
benchmark, which earlier rows didn't have — a useful outside sanity check once results are in.

Verified with the same CPU trial-run process as every previous row before trusting it (found and
fixed one unrelated environment hiccup along the way — a fresh work session hadn't activated the
right software environment). Then handed off row **C3** to the cluster GPU (job 308449) — that
first attempt ran out of GPU memory partway through the very first training step (the technique's
underlying reconstruction-quality check needs a lot of memory at our point-cloud size, more than
the published benchmark it was originally built for uses). Fixed by breaking that one step into
smaller pieces processed one at a time instead of all at once — same math, much less memory at
any given moment — verified with another CPU trial run, then resubmitted (job 308459).

## 12. Row C3 Finished — Self-Supervised Adaptation Doesn't Help Here, But Doesn't Destabilize Either — 2026-09-11

Job 308459 finished. Applying the same "don't trust the headline number alone" lesson learned
from Milestone 10, the full picture across the whole 100-epoch run was checked before drawing any
conclusion, not just the one checkpoint the project's selection process picked.

**The self-supervised approach did not close the gap — if anything it's slightly worse than doing
nothing.** The selected checkpoint scores 0.57 target accuracy, lower than C1's plain "no
adaptation" baseline (0.75), and looking at the whole run instead of one snapshot confirms it's
not a fluke: averaged across every epoch, self-supervised adaptation (0.70) still comes in below
the no-adaptation baseline (0.79). This lines up with something found while double-checking the
published external benchmark for this exact technique: the original paper's own follow-up
experiments found that running this self-supervised trick on *both* datasets at once (which is
what our project's plan specifies) actually works worse than running it on the unlabeled target
dataset only. That's a real, documented property of this specific configuration — not a bug — and
it's now flagged clearly before anyone mistakes a future underwhelming result for a mistake.

**The one genuinely good news finding: unlike the adversarial approach from Milestone 10, this
one does NOT destabilize training.** Its epoch-to-epoch accuracy swings are just as mild as the
no-adaptation baseline's, nothing like the wild swings the adversarial approach showed. So the two
adaptation methods tried so far fail for two completely different reasons — one destabilizes
training and gets worse the longer it runs; the other stays stable but just doesn't transfer the
self-supervised signal usefully to the target dataset under this exact setup. Distinguishing
those two failure modes clearly is itself useful groundwork for the fusion/physics-informed work
planned later in the project.

## 13. A Step Back, a Real Bug Found, and a Fix — Before Committing More GPU Time to Row C4 — 2026-09-12

Before starting the next adversarial row (C4), stopped to ask a harder question: neither of the
two adaptation methods tried so far (Milestones 10 and 12) actually beat the plain "no
adaptation" baseline's overall behavior. Was that a real, expected finding about this data, or
was something fixable going on under the hood? Rather than guessing, this was checked properly —
re-reading the actual training logs and the actual class breakdown of both datasets.

**Found something concrete and specific.** The two datasets have *opposite* majority species —
Crops3D (source) is about three-quarters maize, Pheno4D (target) is about two-thirds tomato. The
adversarial approach from Milestone 10 was, nearly a quarter of the time, collapsing to
predicting maize on every single target plant — the source dataset's majority class winning out,
not genuine learning. Traced this to a specific, fixable design choice: one part of the
adversarial training recipe (a term that pushes the model toward making confident, decisive
predictions) was applied at full strength from the very first training step, before the model
had learned anything reliable yet — which let it lock in on the wrong, source-biased answer
early and stay there. The code's own documentation had actually already warned about exactly
this risk when it was first written, but the fix for that risk wasn't actually implemented at
the time — a real inconsistency between the stated reasoning and the code, now corrected.

**The fix worked, measurably.** Re-ran row C2 with that one term now phased in gradually instead
of applied at full strength immediately. Results: the wild epoch-to-epoch swings roughly halved,
and the "collapses to predicting one class for everyone" problem dropped from about a quarter of
all epochs to about 1 in 20. That's strong, concrete confirmation the diagnosis was right, not
just a plausible-sounding story.

**But it's not a silver bullet.** Even fixed and much more stable, row C2's adversarial approach
still doesn't beat the plain "no adaptation" baseline (Milestone 4) when judged across its whole
training run, not just one snapshot. So the fix cleaned up a real, specific bug — but the
underlying, larger finding (adversarial adaptation doesn't obviously help on this particular
data, likely for a mix of reasons: opposite-majority class balance between the two datasets, a
genuinely small dataset, and Pheno4D spanning growth stages Crops3D never does at all) still
stands. Both the original (buggy) and corrected results were kept side by side in the technical
notes, not overwritten, since the diagnosis process itself — how a subtle instability was traced
to its actual cause — is worth keeping as a record, separate from which numbers are now "official."

Row C4 (the next adversarial row, testing whether the same method holds up when only one specific
kind of data augmentation — simulated sensor noise — is used instead of the full mix) now starts
from this corrected, more trustworthy baseline.

---

## 14. Row C4 Finished — Noise-Only Augmentation Doesn't Rescue Adversarial Adaptation Either — 2026-09-12

Row C4 asked a narrow, direct question: does the adversarial adaptation method's disappointing
result in Milestone 13 (fixed but still underperforming plain no-adaptation) have anything to do
with the specific mix of augmentations C2 used (rotation, scaling, flipping, cropping, dropout,
and jitter, all together)? C4 reruns the exact same, now-corrected adversarial machinery, but
with every augmentation removed except simulated sensor noise (jitter) — a direct test of whether
narrowing the augmentation to the one kind DGCNN is supposedly most vulnerable to changes the
picture.

**It doesn't, in any way that matters.** Read across the full 100-epoch training run (not just
one snapshot — the same careful reading applied to every adversarial row so far), C4's average
target-domain accuracy (0.613) lands close to C2's corrected result (0.633) and both sit well
below the plain no-adaptation baseline's average (0.791). The shape of the decline over training
is also the same in both: an early rise, then a steady fall across the back half of training —
the same pattern, just with a different augmentation mix underneath it. Whether the model's
predictions on unseen target data track the training signal it's allowed to watch (accuracy on
the labeled source data) is also equally uninformative in both cases — essentially no
correlation either way.

**One place the augmentation choice did matter a lot: how well the model segments plant organs.**
C4's segmentation quality (mIoU) came in noticeably higher than either C1 or C2 for both Tomato
and Maize. The most likely explanation isn't anything about noise-robustness — it's that the full
augmentation mix used elsewhere includes random cropping and dropout, which physically remove
points from the cloud and make the per-point segmentation task harder regardless of domain
adaptation; jitter-only never removes points. This is a plausible explanation, not a confirmed
one — no experiment isolates cropping/dropout removal specifically as the cause.

**Bottom line:** the disappointing adversarial-adaptation result from Milestone 13 isn't an
artifact of which augmentations were used — it shows up again, in the same shape, under a
completely different augmentation mix. This strengthens the case that something about the
adversarial method itself, interacting with this particular pair of datasets (small sample
sizes, opposite class majorities between source and target, and a target dataset that spans
plant growth stages the source dataset never does), is the real driver — not a fixable detail of
augmentation choice. Full trajectory tables, quartile breakdowns, and the segmentation-mIoU
comparison are in `step_notes/C4_DGCNN_DA_A_LN.md`.

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
| C (DGCNN) | C2 | Adversarial (anchor method) | **Done** — a real bug found and fixed (Milestone 13); still doesn't beat no-adaptation overall, but far more stable now |
| C (DGCNN) | C3 | Self-supervised | **Done** — trained, doesn't help target accuracy but training stayed stable (see Milestone 12) |
| C (DGCNN) | C4 | Adversarial, jitter-noise augmentation only | **Done** — same underperformance vs. no-adaptation as C2, confirming it's not an augmentation-mix artifact (Milestone 14) |
| C (DGCNN) | C5 | Oracle (upper-bound reference) | Not started |
| D (fusion) | D1–D6 | Combining the best backbone/strategy with growth-curve features | Not started |
| E (deployment, optional) | E1–E3 | Model compression / distillation | Not started |

**In one sentence:** five rows (A1, C1, C2, C3, C4) are fully done — giving the first real
cross-backbone comparison and a consistent picture across both adaptation methods and two
different augmentation mixes tried so far (adversarial: underperforms no-adaptation regardless of
augmentation mix; self-supervised: stable but also worse) — and the other 19 rows are not started
yet.

---

*Update this file with a new dated section after each real milestone — not every small step.*
