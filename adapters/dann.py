"""
dann.py

Gradient Reversal Layer + domain discriminator + entropy minimization for
DA-A (adversarial) rows -- A2/A4, B2, C2/C4 in the strategy table. Kept in
its own module (not folded into losses.py/models.py) so it is reused
byte-for-byte across all three backbones: CLAUDE.md's DA-A description
says this method "appears identically across all 3 backbones so
cross-backbone differences are attributable to architecture, not method."

Not adapted from DefRec_and_PCM: that repo implements DefRec (self-
supervised deformation reconstruction) + PCM (point cloud mixup), not
DANN. Confirmed by grepping its full source tree for
adversarial/domain_cls/dann/grad_rev/GRL/discriminator -- no matches, and
PointDA/trainer.py's training loop (the repo's only trainer) has no
domain-classification branch at all. So CLAUDE.md's "Do not reimplement
DANN... adapt this repo" is satisfied the same way C1 handled the DA-0 gap
(see train_c1_dgcnn_da0.py's docstring): the repo's DGCNN backbone is
reused unmodified (via adapters/models.py), and the adversarial machinery
that doesn't exist upstream is implemented here from scratch, standard
DANN (Ganin & Lempitsky, 2016).

Two-regime loss weighting (CLAUDE.md "Loss architecture"): L_dom and L_ent
are fixed/scheduled, NEVER routed through KendallUncertaintyWeighting --
a learned weight on the adversarial term would let the network inflate
its uncertainty to silently disable adaptation. Training scripts must
combine them as `kendall({"cls":..., "seg":...}) + l_dom + lambda_ent *
l_ent`, added to (not passed into) the Kendall module.

lambda_p(p) = 2/(1+exp(-10p)) - 1 is the strategy table docx's own formula
(docs/Domain_Invariance_Strategy_Table.docx, "Level 2" section: "Adversarial,
physics, and distillation terms use fixed/scheduled coefficients (lambda_p
= 2/(1+e^(-10p)) - 1) and are never assigned a learned sigma"), identical
to the original DANN paper's schedule, p in [0,1] = training progress.

INFERENCE, not a confirmed spec value (flagged per CLAUDE.md "Flag
inferences vs. confirmed facts explicitly" -- the docx gives the lambda_p
formula but does not spell out whether it is applied as the GRL's gradient-
scaling alpha, as an outer coefficient on L_dom in the total-loss sum, or
both):
  - This implementation applies lambda_p as the GRL alpha only (ramping how
    hard the reversed gradient pushes the backbone toward domain-invariant
    features) and leaves L_dom's own coefficient at 1 in the total loss.
    Applying lambda_p a second time as an outer multiplier on L_dom would
    scale the backbone's adversarial gradient by lambda_p^2 (once via GRL,
    once via the outer term) while only scaling the discriminator's own
    gradient by lambda_p^1 -- an asymmetric double-count. Single-
    application-via-GRL-alpha is what essentially every standard DANN
    reference implementation does (e.g. Ganin's original Caffe/Theano code,
    fungtion/DANN), so it was chosen here over the literal-double-
    application reading.
  - L_ent's weight (lambda_ent) has no formula anywhere in the docx --
    the "adversarial, physics, and distillation terms" sentence covering
    lambda_p does not name entropy minimization specifically.

REVISED 2026-09-11, after C2's first real run: lambda_ent is now RAMPED
alongside lambda_p (lambda_ent_schedule below), not held at a fixed
constant. The original version of this file used a fixed, non-ramped
lambda_ent, reasoning that ramping it up early risked "reinforcing
confidently-wrong target pseudo-predictions." That reasoning was correct
but the implementation didn't act on it -- a fixed lambda_ent from epoch 0
applies exactly the risk it warned about, at full strength, before the
shared features have any adversarial pressure behind them yet. C2's actual
run showed this happening, verified from the log, not assumed: source
(Crops3D) and target (Pheno4D) have OPPOSITE class majorities (source is
73% Maize, target is 62-63% Tomato -- data/crops3d_train.csv vs.
data/pheno4d_adaptation_pool.csv/pheno4d_heldout_eval.csv), and 23 of C2's
100 epochs showed target predictions collapsed to source's majority class
(avg_acc ~= 0.5000, acc ~= 0.3651 = 23/63, i.e. predicting Maize on every
target sample) while the entropy loss fell from 0.32 to 0.05 (max possible
for 2 classes is ln(2)=0.693) -- entropy minimization was successfully
driving very confident predictions, just frequently confident in the
source-biased direction, especially early before the discriminator had
learned anything real to oppose. Full diagnosis, including how this was
found, in step_notes/C2_DGCNN_DA_A.md (kept as a documented investigation,
not deleted, even though the numbers it explains are superseded by the
rerun with this fix).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Identity in the forward pass; scales the backward gradient by
    -alpha. alpha is expected to be lambda_p(p) (see below), ramped from 0
    to 1 over training so early gradients don't destabilize the backbone
    before the discriminator has learned anything useful to oppose."""
    return _GradReverse.apply(x, alpha)


def lambda_p_schedule(p: float, gamma: float = 10.0) -> float:
    """p in [0,1] = training progress (epoch*n_batches+batch)/(epochs*n_batches).
    lambda_p = 2/(1+exp(-gamma*p)) - 1, per the strategy table docx."""
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def lambda_ent_schedule(p: float, lambda_ent_max: float, gamma: float = 10.0) -> float:
    """Ramped entropy-minimization weight: lambda_ent_max * lambda_p(p) --
    rides the same DANN ramp as the GRL alpha so entropy minimization
    doesn't apply full-strength pressure on unlabeled target predictions
    before the shared features have had any adversarial pressure behind
    them (see module docstring, "REVISED 2026-09-11" -- this replaced an
    earlier fixed-constant lambda_ent after C2's first run showed target
    predictions repeatedly collapsing to source's majority class while
    entropy minimization drove them to high confidence early)."""
    return lambda_ent_max * lambda_p_schedule(p, gamma)


class DomainDiscriminator(nn.Module):
    """Binary domain classifier (source=0, target=1) over a pooled global
    feature vector. The caller applies grad_reverse to the feature before
    passing it in here, so this module itself is a plain MLP trained by
    ordinary (non-reversed) gradient descent to actually detect domain --
    it is the backbone's gradient, not the discriminator's, that gets
    flipped."""

    def __init__(self, in_dim: int = 1024, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Mean per-sample entropy of softmax(logits), for minimization on
    unlabeled target classification predictions (CLAUDE.md: "entropy
    minimization on unlabeled target predictions"). Ground-truth target
    labels are never used -- only the model's own predicted distribution."""
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=1).mean()
