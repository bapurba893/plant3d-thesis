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

## 15. Row C5 Finished — The Oracle Ceiling, and How Much Headroom Was Really There — 2026-09-12

Row C5 closes out Block C by asking the question everything before it was building toward: if a
model got to train directly on real, labeled examples from the target sensor/greenhouse setup
(Pheno4D) instead of only ever seeing the source dataset (Crops3D), how well could it possibly
do? This isn't a method anyone would actually deploy — getting labeled data from every new sensor
defeats the whole purpose of domain adaptation — it's a ceiling, a way of finding out how much
room for improvement genuinely exists before concluding that a domain-adaptation method's
disappointing result (Milestones 10, 12, 14) means "nothing more was achievable anyway."

The training data itself needed care: Pheno4D's labeled subset is small (90 labeled scans across
only 10 plants) and multi-temporal, so the split into a training set and a validation set was
done by plant, not by individual scan, to avoid near-duplicate scans of the same plant leaking
across the split. The held-out test set used for the final, most important number is the exact
same one every other row in Block C (C1 through C4) reports against — so the number that comes
out is directly comparable, not a different yardstick.

**The result: a near-perfect ceiling, and a real, substantial gap above the no-adaptation
baseline.** Read across the model's full 100-epoch training run (the same careful,
whole-trajectory reading applied to every row so far, not just a single snapshot), the Oracle's
average target-domain accuracy came in at 0.925 — compared to 0.791 for the plain "no
adaptation" baseline (Milestone 4), a 13-point gap. Read from the single best checkpoint each
protocol selects, the gap is even larger: a perfect 1.0000 (63 out of 63 correct, both plant
species) for the Oracle versus 0.746 for no-adaptation — a 25-point gap.

**This changes how to read every adversarial and self-supervised result so far.** All three
adaptation attempts tried in this block (adversarial with full augmentation, self-supervised
reconstruction, adversarial with noise-only augmentation) landed *below* the no-adaptation
baseline, not just short of the Oracle ceiling. Because the ceiling sits well above the
baseline, that's not "the task was already maxed out, nothing to gain" — there was real,
double-digit-point improvement sitting on the table, and none of the three methods tried so far
captured any of it. The underlying model architecture (DGCNN) is clearly capable of learning
this task well when given real target-domain labels, even from a small amount of them — so the
recurring problem documented in Milestones 10-14 is a *transfer* problem specific to how these
domain-adaptation methods interact with this dataset, not a sign that the task itself, or the
model's capacity to learn it, is the bottleneck.

One more contrast worth naming: every no-adaptation/adversarial/self-supervised run so far showed
target accuracy *declining* over the second half of training, with the training signal available
for model selection (accuracy on labeled source data) telling a misleading or unhelpful story
about what was actually happening on target data. The Oracle run shows the opposite: accuracy
climbs steadily the longer it trains, and the val-loss signal used to pick the best checkpoint
tracks target performance closely and reliably. That's the ordinary, healthy shape supervised
learning is supposed to have — a shape none of the adaptation methods have shown yet, because
none of them have ever had a real target label to learn from.

Full trajectory tables, the complete C1-C5 comparison, and caveats (small sample sizes on both
the Oracle's own training data and the held-out test set) are in `step_notes/C5_DGCNN_DA_O.md`.

---

## 16. Row A2 Finished — Same Pattern, Smaller Effect, on a Different Backbone — 2026-09-12

With Block C's story clear (no-adaptation beats every adaptation method tried so far, on
DGCNN), the natural next question was whether that's a fact about DGCNN specifically, or about
this pair of datasets more generally. Row A2 tests this directly: it takes the exact same
adversarial adaptation machinery used for row C2, changes nothing about the method, and swaps
in the PointNet++ backbone instead — comparing against A1 (PointNet++'s own no-adaptation
baseline) the same way C2 was compared against C1.

**The same basic pattern shows up again, but much less dramatically.** Read across the full
100-epoch training run, adversarial adaptation's average target accuracy (0.554) came in below
plain no-adaptation's (0.599) — the same direction as the DGCNN result — but the gap is much
smaller (about 4-5 points, versus roughly 16 points on DGCNN) and, read from the single best
checkpoint each protocol selects, the two are close enough to call a tie (a difference of under
2 points).

**The reason the effect looks smaller here is itself informative.** PointNet++'s own
no-adaptation baseline (row A1) turns out to be considerably less stable than DGCNN's — even
with no adversarial training involved at all, it collapses to predicting a single plant species
for the whole target test set in 17 out of 100 training epochs, compared to DGCNN's 3 out of
100. Against a baseline already that noisy, adding adversarial adaptation's further ~4-5 point
average drop is a much smaller, harder-to-be-sure-about effect than the same method's clear,
sharp degradation of DGCNN's otherwise very stable baseline.

**Bottom line:** this doesn't fully settle whether the underperformance is "about the dataset"
or "about the backbone" — it's some of both. The same basic shape (adaptation doesn't help,
and the loss signal used to pick a checkpoint becomes less trustworthy once adaptation is
turned on) shows up on both backbones, supporting the idea that something about this specific
pair of datasets (opposite class balance, small sample sizes) is a real, shared headwind. But
DGCNN's result is the clean, decisive version of that story specifically because DGCNN's
baseline had so little pre-existing noise to blur the comparison — PointNet++'s baseline was
already unstable on its own, so the additional damage from adaptation is real but harder to
separate from noise that was already there. Full trajectory tables and the four-way comparison
(A1, A2, C1, C2) are in `step_notes/A2_PointNet2_DA_A.md`.

---

## 17. Row A3 Finished — The First Adaptation Method That Actually Wins, But Not for Free — 2026-09-12

Every domain-adaptation attempt so far (adversarial on both backbones, self-supervised on
DGCNN) has underperformed its own backbone's plain no-adaptation baseline. Row A3 asks whether
that holds for self-supervised adaptation on PointNet++ too — the DGCNN version of this method
(row C3) was the least-bad result seen so far (only about 9 points below no-adaptation, versus
roughly 16-18 points for the adversarial attempts), so the question was whether that "least bad"
pattern would repeat on a different backbone.

**It didn't just repeat — it reversed, dramatically.** Read across the full 100-epoch training
run, this method's average target-domain accuracy (0.800) came in about 20 points *above* the
no-adaptation baseline (0.599) — the first clear win by any adaptation method attempted in this
project so far. Read from the single best checkpoint the protocol selects, the gain is even
larger: about 43 points (0.889 versus 0.460). The improvement also came with a big stability
gain: the no-adaptation baseline collapsed to guessing one plant species for the whole target
test set in 17 of 100 training epochs; this run did that only twice.

Given how different this is from every prior result, the numbers were checked carefully before
being trusted: no target labels were used anywhere in training (confirmed the same
never-touch-target-labels code path already used safely in the DGCNN version of this method),
model selection never touched target data, and the improvement holds up across nearly the
entire training run rather than being a lucky single checkpoint.

**But it's not a clean win — there's a real cost, on a different metric.** Reading the organ
segmentation quality on the labeled source data (Crops3D) that this row also tracks: Tomato
segmentation quality collapsed early in training and stayed collapsed for the rest of the run,
while Maize segmentation stayed healthy and even improved. The likely reason (not confirmed by
a dedicated experiment): Tomato is already the smaller, harder-to-segment species in the
training data, and the extra self-supervised training signal seems to have crowded out Tomato's
already-fragile segmentation quality in favor of the classification task, which improved
dramatically.

**What this means for the bigger picture:** the adversarial method's underperformance (rows C2,
C4, A2) looks like a property of the dataset that shows up on both backbones, just more clearly
on one than the other. The self-supervised method (rows C3 now vs. A3) is different — it
actively hurts one backbone and actively helps the other, a much stronger kind of
architecture-dependence than anything seen before it. Whichever backbone eventually gets carried
forward into later stages of this project, this result is a reminder that a domain-adaptation
method's effect — not just how strong it is, but which direction it points — can flip entirely
between architectures, and needs to be checked on each one rather than assumed. Full trajectory
tables, the segmentation-tradeoff detail, and the reasoning behind the likely mechanism are in
`step_notes/A3_PointNet2_DA_S.md`.

---

## 18. Third Backbone Integrated: KPConv, and Block B Started — 2026-09-12

With Block A (PointNet++) and Block C (DGCNN) both well underway, started integrating the third
and last model architecture: KPConv, a convolutional approach whose whole selling point is being
less sensitive to how densely packed the points in a scan are — directly relevant to the sensor
gap between the two datasets (structured-light/RGB-D vs. laser scanning naturally produce
different point densities).

The setup notes had warned this one would need a compiled GPU extension and be the most
troublesome of the three to integrate. Turned out to be half right: it does need compiled code,
but it's compiled *CPU* code (for a preprocessing step, not the actual neural network layer),
confirmed by successfully building and running it on the cluster's login node, which has no GPU
at all. The real friction was different: the external code was written around 2020 and needed a
handful of small, well-understood compatibility fixes to work with today's numpy/Python tooling —
the same kind of minor patching already needed once before for the DGCNN reference codebase.

Built the model and training script the same way as the other two backbones (mirroring their
"no adaptation" baseline setup), verified it with a one-epoch trial run on real data before
trusting it — same process every previous integration went through — and handed off row **B1**
(KPConv, no adaptation, Block B's own starting baseline) to the cluster GPU to train for real
(job 308845). One bit of good luck along the way: a local connection hiccup interrupted the
trial run partway through, but because it was running directly on the cluster (not the laptop),
it kept going unattended and finished cleanly on its own — no work lost.

---

## Current Status: the 24-Row Strategy Table

The full experiment plan is a 24-row table (5 blocks: three model architectures each tested
under several domain-adaptation strategies, plus a fusion block and an optional deployment
block). Full detail lives in `docs/Domain_Invariance_Strategy_Table.docx`; this is just the
at-a-glance status.

| Block | Row | What it is | Status |
|---|---|---|---|
| A (PointNet++) | A1 | No adaptation (baseline) | **Done** — full result committed |
| A (PointNet++) | A2 | Adversarial (anchor method) | **Done** — same underperformance-vs-DA-0 direction as C2, but much smaller/less clear-cut, because A1's own baseline is already unstable (Milestone 16) |
| A (PointNet++) | A3 | Self-supervised | **Done** — first adaptation method to clearly beat its own no-adaptation baseline (+20 pts full-run mean), but at a real cost to Tomato segmentation quality (Milestone 17) |
| A (PointNet++) | A4 | Adversarial, cropping/dropout augmentation only | Not started |
| A (PointNet++) | A5 | Oracle (upper-bound reference) | Not started |
| B (KPConv) | B1 | No adaptation (baseline) | Training on cluster (job 308845), not yet finished |
| B (KPConv) | B2 | Adversarial (anchor method) | Not started |
| B (KPConv) | B3 | Discrepancy-based adaptation | Not started |
| B (KPConv) | B4 | Deliberately unadapted, cropping/dropout only | Not started |
| B (KPConv) | B5 | Oracle (upper-bound reference) | Not started |
| C (DGCNN) | C1 | No adaptation (baseline) | **Done** — full result committed |
| C (DGCNN) | C2 | Adversarial (anchor method) | **Done** — a real bug found and fixed (Milestone 13); still doesn't beat no-adaptation overall, but far more stable now |
| C (DGCNN) | C3 | Self-supervised | **Done** — trained, doesn't help target accuracy but training stayed stable (see Milestone 12) |
| C (DGCNN) | C4 | Adversarial, jitter-noise augmentation only | **Done** — same underperformance vs. no-adaptation as C2, confirming it's not an augmentation-mix artifact (Milestone 14) |
| C (DGCNN) | C5 | Oracle (upper-bound reference) | **Done** — Block C complete. Near-perfect ceiling (0.925 full-run mean / 1.0000 selected) confirms real, ~13-25 point headroom above the no-adaptation baseline (Milestone 15) |
| D (fusion) | D1–D6 | Combining the best backbone/strategy with growth-curve features | Not started |
| E (deployment, optional) | E1–E3 | Model compression / distillation | Not started |

**In one sentence:** Block C (all five DGCNN rows: C1-C5) is fully done, plus A1/A2/A3 on
PointNet++ — giving a real cross-backbone comparison showing adversarial adaptation's
underperformance is a shared (dataset-driven) pattern of differing magnitude, self-supervised
adaptation's effect is architecture-dependent enough to actually flip direction (hurts DGCNN,
helps PointNet++ with a segmentation tradeoff), and a confirmed Oracle ceiling showing real
headroom exists — the third backbone (KPConv, Block B) is now integrated and its own baseline
(B1) is training on the cluster, with the other 16 rows not started yet.

---

*Update this file with a new dated section after each real milestone — not every small step.*
