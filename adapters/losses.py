"""
losses.py

Loss components for L_cls/L_seg per the strategy table's "Hybrid Loss
Definitions" (docs/Domain_Invariance_Strategy_Table.docx):

  L_seg = L_wCE + lambda_lov * L_Lovasz     (w_c = (f_c+eps)^-1, lambda_lov = 1.0)
  L_cls = L_CE

Cross-task combination (Level 2): cooperative task losses (L_cls, L_seg,
L_trait, L_growth, cooperative DA losses) use learned Kendall uncertainty
weighting, NOT a plain sum -- see CLAUDE.md "Loss architecture". Per the
docx: classification-type task j contributes exp(-s_j)*L_j + s_j/2;
regression-type task i contributes exp(-s_i)/2*L_i + s_i/2. L_dom/L_ent/
L_phys/L_mono/L_KD-* are NEVER included here -- they use fixed/scheduled
coefficients instead (see CLAUDE.md).

None of this exists in the DefRec_and_PCM reference repo (its own
PointSegDA/trainer.py uses a plain, unweighted nn.CrossEntropyLoss with no
Lovasz term) -- these are the strategy table's own loss terms, implemented
from the published Lovasz-Softmax formula (Berman et al., CVPR 2018), not
adapted from any existing repo code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights_from_counts(counts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """w_c = (f_c + eps)^-1, normalized to sum to num_classes (keeps LR/wd
    scale meaningful) -- the exact pattern already used for C1's L_cls
    weighting, reused here for L_seg's per-class weights."""
    freq = counts.float() / counts.sum().clamp(min=1)
    w = 1.0 / (freq + eps)
    return w / w.sum() * len(counts)


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension w.r.t. sorted errors, for one class."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-8)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    probas: (P, C) softmax probabilities, flattened over batch and points.
    labels: (P,) int64 ground-truth class ids.
    Averages the per-class Lovasz hinge over classes actually present in
    `labels` (the standard 'present' variant -- a class with zero ground
    truth points in this batch contributes no usable gradient signal).
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.0  # no class present -- zero loss, keeps grad graph valid
    return torch.stack(losses).mean()


def seg_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor,
             lambda_lov: float = 1.0) -> torch.Tensor:
    """
    L_seg = L_wCE + lambda_lov * L_Lovasz for one species' segmentation
    head. logits: (B, N, num_classes); labels: (B, N) int64.
    """
    logits_flat = logits.reshape(-1, logits.size(-1))
    labels_flat = labels.reshape(-1)
    l_wce = F.cross_entropy(logits_flat, labels_flat, weight=class_weights)
    probas = F.softmax(logits_flat, dim=1)
    l_lov = lovasz_softmax_flat(probas, labels_flat)
    return l_wce + lambda_lov * l_lov


class KendallUncertaintyWeighting(nn.Module):
    """
    Learned homoscedastic-uncertainty combination of cooperative task
    losses (Kendall & Gal, 2018), per the strategy table's Level-2
    cross-task combination rule. Holds one learnable log-variance
    parameter s_j per named task; task_types says whether each is
    'classification' (exp(-s_j)*L_j + s_j/2) or 'regression'
    (exp(-s_i)/2*L_i + s_i/2). Only cooperative task losses go through
    this -- L_dom/L_ent/L_phys/L_mono/L_KD-* use fixed/scheduled weights
    instead and must never be passed in here.
    """

    def __init__(self, task_types: dict):
        super().__init__()
        self.task_types = dict(task_types)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(())) for name in task_types
        })

    def weighted_term(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        """Same per-task weighting formula as forward(), for a single named
        task. Lets a caller backprop through one task's graph at a time
        (e.g. summing several sub-batch chunks of one task's loss, each
        immediately .backward()'d, to bound peak memory for a task whose
        forward pass is far more memory-hungry than the others -- see
        train_c3_dgcnn_da_s.py's DefRec chunking) while still accumulating
        gradients into the same shared s_j parameter and backbone weights
        that a single forward()-based call would produce, since each
        term's contribution to the total is linear in that term's loss."""
        s = self.log_vars[name]
        if self.task_types[name] == "regression":
            return torch.exp(-s) / 2.0 * loss + s / 2.0
        return torch.exp(-s) * loss + s / 2.0

    def forward(self, losses: dict) -> torch.Tensor:
        total = 0.0
        for name, loss in losses.items():
            total = total + self.weighted_term(name, loss)
        return total
