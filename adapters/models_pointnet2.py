"""
models_pointnet2.py

Joint classification + per-species segmentation PointNet++ (single-scale
grouping, "SSG"), composed from the vendored yanx27/Pointnet_Pointnet2_pytorch
reference repo's set-abstraction / feature-propagation building blocks --
per CLAUDE.md ("adapt this repo", never reimplement the reference
architectures) -- plus the same PointSegDA.Models.segmentation head shape
already reused for adapters/models.py::DGCNN_ClsSeg, so both backbones'
segmentation heads and the surrounding training-loop pattern
(compute_seg_loss / seg_logits_for_species / etc.) stay identical.

Vendored from https://github.com/yanx27/Pointnet_Pointnet2_pytorch, commit
eb64fe0b4c24055559cea26299cb485dcb43d8dd, cloned as a sibling repo the same
way as DefRec_and_PCM (../Pointnet_Pointnet2_pytorch, i.e. next to
plant3d-thesis/ under /home/pearl/25m0301/ on the cluster) -- unmodified
upstream, imported via PYTHONPATH/sys.path, never version-controlled as
part of this repo. Compatibility check (2026-09-11, mirroring how
DefRec_and_PCM was checked before use): it is pure PyTorch -- no compiled
CUDA extension needed (unlike KPConv) -- confirmed by inspecting
models/pointnet2_utils.py, whose farthest-point-sampling / ball-query /
feature-propagation ops are plain torch tensor code, not C++/CUDA kernels.
Ran PointNetSetAbstraction/PointNetFeaturePropagation on CPU against our
exact data shape (B, 3, 4096) under torch 2.6.0+cu124 + numpy 2.2.6 (our
`plant3d` conda env) with no errors. Their own driver scripts
(train_partseg.py / train_semseg.py / test_*.py) use the removed `np.float`
alias, but we don't import those -- we only import pointnet2_utils.py's two
building-block classes directly, which have no numpy dependency at all.
`models/` has no `__init__.py` in the upstream repo; test_semseg.py/
part_seg.py import it as `from models.pointnet2_utils import ...` (repo
root on sys.path, resolved as a namespace package), but that collides with
this repo's own `adapters/models.py` (DGCNN_ClsSeg) -- both scripts add
`adapters/` to sys.path, so a bare `models` import resolves to our flat
module instead, and `models.pointnet2_utils` then fails with "'models' is
not a package" (hit and confirmed while integrating, not just anticipated).
Sidestepped by adding the vendored repo's `models/` subdirectory itself to
sys.path and importing the uniquely-named `pointnet2_utils` submodule
directly -- the same style pointnet2_cls_ssg.py/cls_msg.py already use
internally (`from pointnet2_utils import ...`).

Architecture: mirrors pointnet2_cls_ssg.py's 3-level set-abstraction encoder
(no input normals -- our cached points are xyz-only, unlike ModelNet40's
normal-resampled variant) for the pooled global feature used by "cls", plus
a 3-level feature-propagation decoder mirroring pointnet2_sem_seg.py's FP
stack (including its convention of feeding None for the finest level's
original per-point features, since we have none beyond xyz) to recover
full-resolution per-point features for "seg_feat". This is deliberately
NOT pointnet2_part_seg_ssg.py's architecture, which conditions its FP
decoder on a one-hot ShapeNet category label -- we have no equivalent input
(species is our "cls" target, not something to feed back in as an input).

DefRec/DANN hooks are intentionally not wired here yet -- row A1 is DA-0
(source-only, no adaptation), the same scope adapters/models.py::
DGCNN_ClsSeg shipped with for row C1. They'll be added when rows
A2/A3/A4 need adversarial/self-supervised training, same as the DGCNN track.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
_POINTNET2_ROOT = os.environ.get(
    "POINTNET2_ROOT", str(_REPO_ROOT.parent / "Pointnet_Pointnet2_pytorch"))
_POINTNET2_MODELS_DIR = str(Path(_POINTNET2_ROOT) / "models")
if _POINTNET2_MODELS_DIR not in sys.path:
    sys.path.insert(0, _POINTNET2_MODELS_DIR)

from pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation  # noqa: E402
from PointSegDA.Models import segmentation as SegmentationHead  # noqa: E402


class PointNet2_ClsSeg(nn.Module):
    def __init__(self, args, num_class: int, seg_num_classes: dict):
        """seg_num_classes: {species_name: num_organ_classes}, e.g.
        {"Tomato": 3, "Maize": 6}. `args` only needs `.dropout` -- matches
        DGCNN_ClsSeg's constructor signature so both backbones plug into the
        same training-script pattern (see adapters/train_c1_dgcnn_da0.py)."""
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(args.dropout)
        self.fc3 = nn.Linear(256, num_class)

        seg_input_size = 128 + 1024  # fp1 per-point width + pooled global feature
        self.seg_heads = nn.ModuleDict({
            species: SegmentationHead(args, input_size=seg_input_size, num_classes=n)
            for species, n in seg_num_classes.items()
        })

    def forward(self, x, activate_DefRec: bool = False):
        """x: (B, 3, N) xyz. Returns {"cls": (B, num_class) raw logits,
        "feat": (B, 1024) pooled global feature (pre-classifier -- domain
        discriminator input for row A2, added when that row was built; C1/A1
        callers that only read logits["cls"]/["seg_feat"] are unaffected),
        "seg_feat": (B, 1152, N) per-point features} -- matches
        DGCNN_ClsSeg.forward's interface exactly (raw logits, not
        log-softmax, since the training scripts feed this straight into
        nn.CrossEntropyLoss / the L_seg helpers). activate_DefRec is
        accepted for call-site compatibility but not implemented -- see
        module docstring; row A1 never sets it True."""
        if activate_DefRec:
            raise NotImplementedError(
                "DefRec is not wired for PointNet2_ClsSeg yet -- row A1 is "
                "DA-0, which never activates it (see module docstring).")
        batch_size, _, num_points = x.shape

        l1_xyz, l1_points = self.sa1(x, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(x, l1_xyz, None, l1_points)  # (B, 128, N)

        global_feat = l3_points.view(batch_size, -1)  # (B, 1024)

        logits = {}
        c = self.drop1(F.relu(self.bn1(self.fc1(global_feat))))
        c = self.drop2(F.relu(self.bn2(self.fc2(c))))
        logits["cls"] = self.fc3(c)
        logits["feat"] = global_feat  # pooled global feature (B, 1024) -- domain discriminator input, see adapters/dann.py (row A2)
        logits["seg_feat"] = torch.cat(
            (l0_points, global_feat.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
        return logits

    def seg_logits_for_species(self, seg_feat: torch.Tensor, species: str) -> torch.Tensor:
        """Same contract as DGCNN_ClsSeg.seg_logits_for_species: seg_feat is
        (B, C, N) already selected for one species, returns
        (B, N, num_classes_for_species)."""
        return self.seg_heads[species](seg_feat)
