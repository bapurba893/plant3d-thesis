"""
coral.py

Deep CORAL (Sun & Saenko, 2016) feature-covariance alignment for DA-D (discrepancy-based) rows
-- currently only B3 in the strategy table (CLAUDE.md: "Block B (KPConv): ... B3 DA-D/ALL").
Kept in its own module, the same convention `dann.py` already established for DA-A, in case a
DA-D row is ever added to Block A/C the way B3b (DA-S) was added to Block B.

Not adapted from DefRec_and_PCM: that repo implements DefRec (self-supervised) + PCM (mixup)
only -- confirmed by grepping its full source tree for coral/mmd/discrepancy/covariance, no
matches (same check already done for `dann.py`, repeated here rather than assumed still true).
So this is implemented from scratch from the published CORAL formula, the same category as
`dann.py`'s from-scratch DANN implementation.

**Why CORAL over MMD**: CLAUDE.md's DA-D definition names either as an option ("Deep CORAL
... or MMD ..."), with no mandate for one over the other in the strategy table docx's summary.
Chose CORAL: no kernel-bandwidth hyperparameter to select or tune (MMD's Gaussian-kernel
bandwidth is a real, non-trivial extra choice with no strategy-table guidance either), a single
well-defined covariance-alignment loss, and it matches CLAUDE.md's own framing of DA-D as
"cheaper and more stable" than DA-A more directly (no kernel computation over all pairs of
source/target samples). Flagged as an explicit choice, not read from a literal spec value --
same discipline as `dann.py`'s own flagged inferences (lambda_p application point, lambda_ent
weight).

**Kendall routing**: per CLAUDE.md's loss architecture, "cooperative DA losses (L_CORAL/L_MMD/
L_defrec)" are explicitly named as LEARNED (Kendall-weighted) cooperative losses -- unlike
DA-A's L_dom/L_ent, which are fixed/scheduled and must never be passed into
`KendallUncertaintyWeighting`. So `L_coral` gets a `"regression"`-type Kendall term (same
category as `L_defrec` in A3/C3/B3b), combined via `kendall({"cls":..., "seg":..., "coral":...})`
in the training script -- NOT summed in with a fixed coefficient the way DA-A's `l_dom` is.
"""

import torch


def coral_loss(source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss: squared Frobenius norm between source and target feature covariance
    matrices, normalized by 4*d^2 (Sun & Saenko, 2016, Eq. 2) so the loss scale doesn't grow
    with feature width `d`.

    source_feat, target_feat: (B, d) pooled global features (this project's KPConv/DGCNN/
    PointNet2 backbones all expose this as `logits["feat"]`) -- no gradient-reversal, no
    discriminator, no domain labels; the two batches' covariance structure is aligned directly.
    Requires B >= 2 for both inputs (unbiased covariance needs at least 2 samples) -- always
    true here since every row's batch_size is >= 4.
    """
    d = source_feat.size(1)
    src_centered = source_feat - source_feat.mean(dim=0, keepdim=True)
    trg_centered = target_feat - target_feat.mean(dim=0, keepdim=True)
    cov_src = (src_centered.t() @ src_centered) / (source_feat.size(0) - 1)
    cov_trg = (trg_centered.t() @ trg_centered) / (target_feat.size(0) - 1)
    return (cov_src - cov_trg).pow(2).sum() / (4.0 * d * d)
