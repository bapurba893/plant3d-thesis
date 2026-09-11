# Row C2: DGCNN, Adversarial Domain Adaptation (DA-A), All Augmentations

## What this is, in plain language

C2 is the "anchor" adversarial row on the DGCNN backbone. Instead of training only on the
labeled source dataset (Crops3D) and hoping it happens to generalize (that's what C1/DA-0 did),
C2 actively forces the network to build features that look the same whether they came from
Crops3D or from unlabeled Pheno4D. It does this with a classic trick (DANN): a small second
network (a "domain discriminator") tries to guess which dataset a plant's features came from,
while the main network is trained to fool it. A "Gradient Reversal Layer" (GRL) is what makes
this a single, ordinary-looking training loop instead of two competing optimizers — it just
flips the sign of the discriminator's gradient on its way back into the main network, so one
`loss.backward()` call does both the "discriminator learns to detect domain" step and the "main
network learns to hide domain" step at once. On top of that, C2 also nudges the network to make
confident (not wishy-washy) predictions on target data it's never labeled ("entropy
minimization").

This is the row every other DA-A row in the table (A2, A4, B2, C4) copies exactly, so that any
difference between backbones is attributable to the architecture, not to the adaptation method
itself.

## Status: in progress (started 2026-09-11)

## Design decisions confirmed before writing code

Per CLAUDE.md's "Loss architecture" two-regime rule, `L_dom` and `L_ent` were built to be
**fixed/scheduled, never learned** — they stay completely outside the
`KendallUncertaintyWeighting` module that already combines `L_cls`/`L_seg` (unchanged from C1).
The reasoning CLAUDE.md gives (a learnable weight would let the network inflate its uncertainty
to silently disable adaptation) is why: total loss = `Kendall(L_cls, L_seg) + L_dom + λ_ent·L_ent`,
added on top of the Kendall term, not passed into it.

## What was done and how

1. **Checked the reference repo first.** Grepped all of `DefRec_and_PCM` for
   adversarial/domain_cls/dann/grad_rev/GRL/discriminator — no matches. Read `PointDA/trainer.py`
   in full: it implements DefRec (self-supervised deformation reconstruction) + PCM (point cloud
   mixup), not DANN — there is no domain-classification branch anywhere in the repo. So there was
   nothing to "adapt" for the adversarial machinery itself; only the DGCNN backbone (already
   wrapped by `adapters/models.py::DGCNN_ClsSeg`, reused unmodified) comes from the reference
   repo. This mirrors exactly how C1 handled the DA-0 gap (see that script's own docstring).
2. **Found the exact `λ_p` formula** by extracting the strategy table docx's raw text
   (`docs/Domain_Invariance_Strategy_Table.docx` → `word/document.xml`, stripped of XML tags) and
   grepping it, rather than guessing: `λ_p = 2/(1+e^(−10p)) − 1`, described as covering
   "Adversarial, physics, and distillation terms" that are "never assigned a learned σ" — this is
   the standard Ganin & Lempitsky (2016) DANN schedule, `p` = training progress in `[0,1]`.
3. **Two inference calls, both flagged explicitly** (the docx doesn't fully specify these — see
   `adapters/dann.py`'s docstring for the full reasoning, not repeated here):
   - `λ_p` is applied as the Gradient Reversal Layer's `alpha` only (ramping how hard the
     reversed gradient pushes the backbone), with `L_dom`'s own coefficient left at 1 in the
     total loss. Applying `λ_p` a second time as an outer multiplier would scale the backbone's
     adversarial gradient by `λ_p²` while only scaling the discriminator's gradient by `λ_p¹` —
     an asymmetric double-count. Single-application-via-GRL-alpha is standard practice in
     essentially every reference DANN implementation.
   - `L_ent`'s weight (`λ_ent`) has no formula anywhere in the docx (the sentence naming `λ_p`
     doesn't mention entropy minimization specifically). Used a fixed, CLI-configurable constant
     (default 0.1) instead of ramping it with `λ_p`, since ramping entropy minimization up
     alongside a still-unreliable discriminator early in training risks reinforcing
     confidently-wrong target pseudo-predictions.
4. **Built `adapters/dann.py`** — new, backbone-agnostic module (not folded into
   `losses.py`/`models.py`) holding the Gradient Reversal Layer (`_GradReverse` autograd
   Function + `grad_reverse` helper), `lambda_p_schedule`, `DomainDiscriminator` (binary MLP:
   1024→256→128→2, BatchNorm+LeakyReLU+Dropout), and `entropy_loss`. Kept separate and
   backbone-agnostic on purpose: CLAUDE.md says DA-A "appears identically across all 3 backbones
   so cross-backbone differences are attributable to architecture, not method" — this file will
   be reused byte-for-byte by A2/A4 (PointNet++) and B2 (KPConv), and C4 (DGCNN/L-N) later.
5. **Exposed the pooled global feature from `DGCNN_ClsSeg.forward`** (`adapters/models.py`):
   added `logits["feat"] = x5_pooled` (the same 1024-dim pre-classifier vector `self.C` already
   consumes) as an additive key — C1's evaluation code, which only reads `logits["cls"]`/
   `["seg_feat"]`, is unaffected. This is the domain discriminator's input.
6. **Wrote `adapters/train_c2_dgcnn_da_a.py`**, built directly on top of `train_c1_dgcnn_da0.py`
   (same backbone, same `L_cls`/`L_seg` setup, same source data/splits/protocol) with exactly the
   adversarial machinery added on top, per CLAUDE.md's row-ordering note ("isolated,
   one-variable-at-a-time increment on top of C1 rather than a rewrite"). New pieces:
   - Loads `data/pheno4d_adaptation_pool.csv` (160 plants) as a second, target-side data loader
     — **points only**, species labels are read (required by `PlantSpeciesDataset`'s return
     signature) but never touched in the loss. This is separate from
     `data/pheno4d_heldout_eval.csv` (63 plants), still used purely as a hands-off
     epoch-by-epoch domain-gap indicator exactly as in C1/A1 — never trained on, never used for
     model selection.
   - Epoch length is source-loader-driven (32 batches at batch size 8 in the smoke test, will be
     8 batches at batch size 32 for the real run); the target adaptation loader is `cycle()`d
     against it, matching `PointDA/trainer.py`'s own `zip(src_loader, trgt_loader)` pattern for
     pairing unevenly-sized domains.
   - Single `Adam` optimizer over backbone + discriminator + Kendall log-vars — the GRL, not a
     separate optimizer/min-max alternation, is what turns one `backward()` call into both "the
     discriminator learns to detect domain" and "the backbone learns to hide it," standard DANN
     practice.
   - Model selection: identical protocol to C1/A1 — best checkpoint by lowest **source** val
     total loss (`Kendall(cls,seg)` on Crops3D val only). `L_dom`/`L_ent` are deliberately
     excluded from the selection criterion (documented in the script's own docstring): they
     involve target features/predictions and are adversarial/entropy terms, not accuracy
     measures, so a lower `L_dom` doesn't mean a better model — and including them would either
     need target labels (defeating the point) or select on an unreliable signal. Also matches
     `PointDA/trainer.py`'s own stated convention ("save model according to best source model,
     since we don't have target labels").
7. **CPU smoke test** (2026-09-11): 1 epoch, batch size 8, real Crops3D/Pheno4D data, `--gpu -1`.
   Ran end-to-end in ~12 minutes with no shape/gradient errors. All losses/metrics came back
   non-zero and in sane ranges for a single epoch (`cls loss 0.47`, `seg loss 2.24`, `dom loss
   0.73` — near `ln(2)≈0.69` as expected early on when the discriminator is still weak, `ent loss
   0.36`). Confirmed `λ_p` schedule uses `args.epochs` (not the smoke test's local epoch count)
   for its denominator, so it ramps gradually across the full 100-epoch real run rather than
   hitting 1.0 within the smoke test's single epoch by construction — the smoke test showing
   `λ_p: 0.9999` at end of epoch 0/1 is the *expected* smoke-test artifact of `total_steps =
   1*32`, not a bug; the real run's `total_steps = 100*8` ramps it properly. Deleted the smoke
   test's `results/_smoketest_c2/` output afterward (scratch, not part of the real result).
8. **Added `jobs/c2_dgcnn_da_a.sbatch`** — same cluster settings as C1's job (dgx
   partition/qos, 1 GPU, 4h wall time), 100 epochs / batch 32 / lr 1e-3 / wd 5e-5 / dropout 0.5
   (identical to C1, isolating the DA-A variable), plus `--disc_dropout 0.3 --gamma 10.0
   --lambda_ent 0.1`.

## Technical specifics

- **New files:** `adapters/dann.py`, `adapters/train_c2_dgcnn_da_a.py`, `jobs/c2_dgcnn_da_a.sbatch`
- **Modified:** `adapters/models.py` (`DGCNN_ClsSeg.forward` now also returns `logits["feat"]`)
- **Not modified:** `adapters/losses.py` (`KendallUncertaintyWeighting` reused as-is, still only
  ever sees `{"cls", "seg"}` — never `L_dom`/`L_ent`)

## What's next

Submit `jobs/c2_dgcnn_da_a.sbatch` to the cluster (100 epochs, ~4h budgeted, same wall-clock
ballpark as C1/A1). Once it lands, compare against C1 (DA-0) on the same backbone: does the
adversarial anchor actually close the target-classification gap C1/A1 both showed drifting
downward over training? That comparison is the whole point of the anchor row existing.

---
**2026-09-11:** Submitted `jobs/c2_dgcnn_da_a.sbatch` to the `dgx` partition — job **308389**.
Queue was empty beforehand (no conflicting jobs). Awaiting completion.
