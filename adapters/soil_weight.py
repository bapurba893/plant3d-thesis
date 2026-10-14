"""
soil_weight.py

Learned soil/no-soil weight on the per-point segmentation cross-entropy, for
the Block PP soil-weight experiment (adapters/train_c1_dgcnn_da0_softw.py).
Requested by the supervisor: the soil weight is LEARNED (trainable), not fixed
from class counts, and can depend on per-point features.

Scope (where the weight applies):
  - Crops3D Maize samples only, on points whose label is soil (id 0) or pot
    (id 5) -- ids from step_notes/PP_0_Prerequisite_Checks.md check 1.
    Tomato has no soil, and Pheno4D (target) is unlabelled in this baseline.
  - Only the L_wCE half of L_seg. The Lovasz term is a set-level IoU
    surrogate with no per-point weight, so it is left unweighted.
  - The existing inverse-frequency class weights stay; the soil weight
    multiplies on top:
        L_wCE = sum_i s_i * cw[y_i] * ce_i / sum_i cw[y_i]
    which equals the original F.cross_entropy(weight=cw) when every s_i = 1.

Modes:
  - learned_scalar:   one trainable raw scalar a, soil weight w = B(a).
  - learned_features: 2-layer MLP (3 -> 16 -> 1), soil weight w_i = B(f(x_i)).
  where B(r) = 0.2 + 4.8 * sigmoid(r) bounds the weight to [0.2, 5]
  (user decision 2026-10-14: the bound stops the optimiser pushing the plant
  weights to ~0 by favouring the easy soil points).

Collapse protection (normalisation, user decision 2026-10-14): before
normalising, soil/pot points carry w_i and all other maize points carry 1;
the whole vector is then divided by its mean over ALL labelled maize points
in the batch, so the mean weight over maize points is exactly 1. The weight
can therefore only shift emphasis between soil and plant points, never scale
the loss down overall. (Normalising over soil points only was rejected: it
makes learned_scalar identically 1.)

Per-point features for learned_features (computed from the input cloud alone,
no labels, no gradient):
  1. end-height: h = (z - z_min) / (z_max - z_min), feature = 2 * min(h, 1-h),
     in [0, 1]. NOT plain "height above the lowest point", because the ALL
     augmentation's cubic_symmetry flips the z sign in half the training
     clouds, which would put the soil at h ~ 1 in those clouds. The weight is
     only ever evaluated on soil/pot points, which sit at one vertical end of
     the cloud, so for them this feature equals the height above the lowest
     point (up to x2) whether the cloud is flipped or not.
  2. radial distance from the vertical axis through the cloud's xy centroid,
     divided by the cloud's maximum radial distance, in [0, 1]. Invariant to
     the z-rotation and x/y flips in the augmentation.
  3. local density: log(d_i / median_j d_j), where d_i is the mean distance
     from point i to its 16 nearest neighbours in the same cloud; clamped to
     [-3, 3]. Scale-free, so it is unaffected by the random scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SOIL_IDS_MAIZE = (0, 5)  # Crops3D maize: 0 = soil, 5 = pot
W_MIN, W_MAX = 0.2, 5.0
KNN_K = 16
FEATURE_NAMES = ("end_height", "radial_dist", "log_rel_knn_dist")
SOIL_WEIGHT_MODES = ("none", "learned_scalar", "learned_features")


def _bounded(raw: torch.Tensor) -> torch.Tensor:
    return W_MIN + (W_MAX - W_MIN) * torch.sigmoid(raw)


# raw value for which the bounded weight is exactly 1 (start at "no reweighting")
_RAW_AT_ONE = float(torch.logit(torch.tensor((1.0 - W_MIN) / (W_MAX - W_MIN))))


@torch.no_grad()
def point_features(pts: torch.Tensor) -> torch.Tensor:
    """pts: (B, 3, N) as fed to the model. Returns (B, N, 3) features."""
    xyz = pts.permute(0, 2, 1)  # (B, N, 3)
    z = xyz[..., 2]
    z_min = z.min(dim=1, keepdim=True)[0]
    z_max = z.max(dim=1, keepdim=True)[0]
    h = (z - z_min) / (z_max - z_min).clamp(min=1e-8)
    end_height = 2.0 * torch.minimum(h, 1.0 - h)

    xy = xyz[..., :2]
    r = (xy - xy.mean(dim=1, keepdim=True)).norm(dim=2)
    radial = r / r.max(dim=1, keepdim=True)[0].clamp(min=1e-8)

    dens = []
    for b in range(xyz.size(0)):  # one cloud at a time: bounded memory (N x N)
        d = torch.cdist(xyz[b], xyz[b])
        knn = d.topk(KNN_K + 1, dim=1, largest=False)[0][:, 1:]  # drop self
        md = knn.mean(dim=1)
        dens.append(torch.log(md.clamp(min=1e-8) / md.median().clamp(min=1e-8)).clamp(-3.0, 3.0))
    log_rel_knn = torch.stack(dens)

    return torch.stack([end_height, radial, log_rel_knn], dim=2)


class SoilWeight(nn.Module):
    """Raw (pre-normalisation) soil weight in [W_MIN, W_MAX], initialised to 1."""

    def __init__(self, mode: str, hidden: int = 16):
        super().__init__()
        if mode not in ("learned_scalar", "learned_features"):
            raise ValueError(f"SoilWeight mode must be learned_*, got {mode!r}")
        self.mode = mode
        if mode == "learned_scalar":
            self.raw = nn.Parameter(torch.tensor(_RAW_AT_ONE))
        else:
            self.mlp = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))
            nn.init.zeros_(self.mlp[2].weight)  # start at w = 1 for every point
            nn.init.constant_(self.mlp[2].bias, _RAW_AT_ONE)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: (B, 3, N). Returns raw soil weight per point, (B, N)."""
        if self.mode == "learned_scalar":
            return _bounded(self.raw).expand(pts.size(0), pts.size(2))
        feats = point_features(pts)
        return _bounded(self.mlp(feats).squeeze(-1))


def normalized_point_weights(raw_w: torch.Tensor, labels: torch.Tensor):
    """raw_w, labels: (B, N) for the batch's maize samples. Soil/pot points
    carry raw_w, all others 1; divided by the mean over every point, so the
    result has mean exactly 1. Returns (weights, soil_mask)."""
    soil = torch.zeros_like(labels, dtype=torch.bool)
    for sid in SOIL_IDS_MAIZE:
        soil |= labels == sid
    w = torch.where(soil, raw_w, torch.ones_like(raw_w))
    return w / w.mean(), soil


def weighted_wce(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor,
                 point_w: torch.Tensor) -> torch.Tensor:
    """Per-point-weighted version of F.cross_entropy(weight=class_weights):
    sum(point_w * cw[y] * ce) / sum(cw[y]). logits (B, N, C), labels/point_w (B, N)."""
    logits_flat = logits.reshape(-1, logits.size(-1))
    labels_flat = labels.reshape(-1)
    ce = F.cross_entropy(logits_flat, labels_flat, reduction="none")
    cw = class_weights[labels_flat]
    return (point_w.reshape(-1) * cw * ce).sum() / cw.sum()


class SoilWeightStats:
    """Per-epoch running statistics of the soil and plant weights, for logging."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = {k: {"sum": 0.0, "sq": 0.0, "n": 0, "min": float("inf"), "max": float("-inf")}
                    for k in ("soil_raw", "soil", "plant")}
        self.at_lo, self.at_hi = 0, 0

    @torch.no_grad()
    def update(self, raw_w, weights, soil):
        for key, vals in (("soil_raw", raw_w[soil]), ("soil", weights[soil]),
                          ("plant", weights[~soil])):
            if vals.numel() == 0:
                continue
            vals = vals.double()
            a = self.acc[key]
            a["sum"] += vals.sum().item()
            a["sq"] += (vals ** 2).sum().item()
            a["n"] += vals.numel()
            a["min"] = min(a["min"], vals.min().item())
            a["max"] = max(a["max"], vals.max().item())
        soil_raw = raw_w[soil]
        tol = 0.01 * (W_MAX - W_MIN)
        self.at_lo += int((soil_raw <= W_MIN + tol).sum().item())
        self.at_hi += int((soil_raw >= W_MAX - tol).sum().item())

    def summary(self) -> str:
        parts = []
        for key in ("soil_raw", "soil", "plant"):
            a = self.acc[key]
            if a["n"] == 0:
                parts.append(f"{key}: n=0")
                continue
            mean = a["sum"] / a["n"]
            std = max(a["sq"] / a["n"] - mean ** 2, 0.0) ** 0.5
            parts.append(f"{key}: mean={mean:.4f} std={std:.4f} min={a['min']:.4f} "
                         f"max={a['max']:.4f}")
        n_soil = self.acc["soil_raw"]["n"]
        if n_soil:
            parts.append(f"soil_raw at bounds: lo={self.at_lo / n_soil:.4f} "
                         f"hi={self.at_hi / n_soil:.4f}")
        return " | ".join(parts)
