"""
models.py

Joint classification + per-species segmentation DGCNN, composed from
DefRec_and_PCM's existing building blocks rather than reimplemented from
scratch (per CLAUDE.md: "adapt this repo", never reimplement the reference
architectures):
  - backbone + classifier head: PointDA.Models.DGCNN (already used for
    row C1's classification-only training).
  - segmentation head shape: PointSegDA.Models.segmentation (the body-part
    segmentation benchmark's own head -- plain Conv1d stack over an
    arbitrary input feature width, so it plugs onto PointDA's DGCNN
    feature dimensions unchanged).

PointDA.Models.DGCNN.forward() computes per-point multi-scale features
(x_cat) internally but only exposes the pooled global vector via
logits["cls"] -- there's no hook to also get x_cat out. We subclass it and
re-run its forward pass by calling its own submodules (self.conv1..conv5,
self.input_transform_net, self.C) in the same sequence as the parent
class -- no layer is reimplemented, only re-sequenced so the intermediate
per-point features can also feed a new segmentation head. This is the
smallest change that doesn't touch the DefRec_and_PCM sibling repo (which
must stay an unmodified upstream dependency).

Per-species segmentation heads: Crops3D's organ-label class count differs
by species (Tomato num_classes=3, Maize num_classes=6 -- see CLAUDE.md
"Per-point Crops3D organ segmentation labels backfilled") and there's no
evidence the numeric ids share meaning across species, so a single shared
seg head would either waste capacity (padded to 6) or force an unjustified
shared label space. One nn.ModuleDict head per species, keyed by species
name, sharing the same DGCNN backbone/classifier.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from PointDA.Models import DGCNN, get_graph_feature  # noqa: E402
from PointSegDA.Models import segmentation as SegmentationHead  # noqa: E402


class DGCNN_ClsSeg(DGCNN):
    def __init__(self, args, num_class: int, seg_num_classes: dict):
        """seg_num_classes: {species_name: num_organ_classes}, e.g.
        {"Tomato": 3, "Maize": 6}."""
        super().__init__(args, num_class=num_class)
        num_f_prev = 64 + 64 + 128 + 256  # matches DGCNN.__init__'s num_f_prev
        seg_input_size = num_f_prev + 1024
        self.seg_heads = nn.ModuleDict({
            species: SegmentationHead(args, input_size=seg_input_size, num_classes=n)
            for species, n in seg_num_classes.items()
        })

    def forward(self, x, activate_DefRec: bool = False):
        """Same backbone forward as DGCNN.forward, but also returns the
        per-point feature map (logits["seg_feat"]) so the caller can route
        each sample through its species' segmentation head -- see
        seg_logits_for_species below -- and the pooled global feature
        (logits["feat"], (B, 1024), pre-classifier) for the DA-A rows'
        domain discriminator (adapters/dann.py). Both are additive keys;
        callers that only read logits["cls"]/["seg_feat"] (C1/A1) are
        unaffected."""
        batch_size = x.size(0)
        num_points = x.size(2)
        logits = {}

        x0 = get_graph_feature(x, self.args, k=self.k)
        transformd_x0 = self.input_transform_net(x0)
        x = torch.matmul(transformd_x0, x)

        x = get_graph_feature(x, self.args, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, self.args, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, self.args, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, self.args, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = F.leaky_relu(self.bn5(self.conv5(x_cat)), negative_slope=0.2)
        x5_pooled = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)

        logits["cls"] = self.C(x5_pooled)
        logits["feat"] = x5_pooled  # pooled global feature (B, 1024) -- domain discriminator input, see adapters/dann.py
        logits["seg_feat"] = torch.cat(
            (x_cat, x5_pooled.unsqueeze(2).repeat(1, 1, num_points)), dim=1)

        if activate_DefRec:
            DefRec_input = torch.cat(
                (x_cat, x5_pooled.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            logits["DefRec"] = self.DefRec(DefRec_input)

        return logits

    def seg_logits_for_species(self, seg_feat: torch.Tensor, species: str) -> torch.Tensor:
        """seg_feat: (B, C, N) subset already selected for one species.
        Returns (B, N, num_classes_for_species)."""
        return self.seg_heads[species](seg_feat)
