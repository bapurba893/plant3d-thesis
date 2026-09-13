"""
models_kpconv.py

Joint classification + per-species segmentation KPConv, composed from the
vendored HuguesTHOMAS/KPConv-PyTorch reference repo's block-construction
logic (models/architectures.py::KPFCNN.__init__/.forward) -- per
CLAUDE.md ("adapt this repo", never reimplement the reference
architectures) -- plus the same PointSegDA.Models.segmentation head shape
already reused for DGCNN_ClsSeg/PointNet2_ClsSeg.

Vendored from https://github.com/HuguesTHOMAS/KPConv-PyTorch, cloned as a
sibling repo the same way as DefRec_and_PCM/Pointnet_Pointnet2_pytorch
(../KPConv-PyTorch, next to plant3d-thesis/ under /home/pearl/25m0301/ on
the cluster), imported via KPCONV_ROOT/sys.path -- see
step_notes/B1_KPConv_DA0.md for the exact commit pinned and the
compilation compatibility fixes this took (numpy.distutils removal +
PyArray_DATA/_NDIM/_DIM signature changes -- both real, necessary patches
to the vendored repo's cpp_wrappers/, not project-specific code, same
category as the np.int patch CLAUDE.md documents for DefRec_and_PCM).

CLAUDE.md describes KPConv as needing "a compiled CUDA extension" --
confirmed while integrating (2026-09-12) that this is not quite right:
the compiled pieces (cpp_wrappers/cpp_subsampling, cpp_wrappers/
cpp_neighbors) are plain CPU C++ extensions (grid subsampling + radius
neighbor search preprocessing), not CUDA -- they compiled and ran
correctly on the cluster's LOGIN node (no GPU/CUDA toolkit needed at all
for these two ops). The actual KPConv convolution itself is plain
PyTorch (models/blocks.py::KPConv), so it runs on GPU automatically like
any other nn.Module -- there was no CUDA-specific compilation step
anywhere in this integration.

Neither KPCNN (classification-only) nor KPFCNN (segmentation-only) in
the reference repo does joint cls+seg, so -- same as DGCNN_ClsSeg's
"re-run the base class's submodules in sequence rather than subclass
blindly" approach -- KPConv_ClsSeg's __init__/forward mirror KPFCNN's own
encoder-loop/decoder-loop construction line-for-line (same block_decider
calls, same skip-connection bookkeeping), with one addition: a
GlobalAverageBlock() applied to the deepest encoder-only feature (the
bottleneck, before any decoder/upsample blocks) feeds a classification
head, mirroring exactly how KPCNN's own architecture stops at the
bottleneck and applies 'global_average' -- just done as part of the same
forward pass that also runs KPFCNN's decoder for segmentation, rather
than as two separate networks.

Per-point segmentation output reshaping: KPConv's per-point features live
in a "stacked" (total_points_in_batch, C) layout (see
adapters/kpconv_collate.py's docstring), not the (B, C, N) layout
DGCNN_ClsSeg/PointNet2_ClsSeg's seg heads expect. Since every sample in
this project is already a FIXED N=4096 point cloud (no cropping/variable
length -- unlike this reference repo's usual scene-segmentation crops),
`batch.lengths[0]` is always `[N]*B`, so the finest-layer decoder output
is guaranteed to be B contiguous chunks of exactly N rows each, in batch
order. Reshaping (B*N, C) -> (B, N, C) -> permute -> (B, C, N) is
therefore an exact, order-preserving operation (not an approximation),
letting `logits["seg_feat"]` plug into the EXACT SAME
`compute_seg_loss`/`seg_logits_for_species`/`seg_eval_metrics` helpers
every other row's training script already uses, unmodified.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KPCONV_ROOT = os.environ.get("KPCONV_ROOT", str(_REPO_ROOT.parent / "KPConv-PyTorch"))
if _KPCONV_ROOT not in sys.path:
    sys.path.insert(0, _KPCONV_ROOT)

# `KPCONV_ROOT/models` has no __init__.py (a namespace package upstream),
# and `import models.blocks` collides with this repo's own flat
# `adapters/models.py` (DGCNN_ClsSeg) once both are on sys.path -- the
# same collision hit integrating PointNet2 (see adapters/models_pointnet2.py's
# docstring). Same fix: add the vendored repo's `models/` subdirectory
# itself to sys.path and import `blocks` by its unique name directly,
# bypassing the `models` package prefix entirely. `blocks.py` has no
# internal `models.*`/`datasets.*`/`utils.*` imports of its own (checked
# before relying on this), so importing it standalone is safe.
_KPCONV_MODELS_DIR = str(Path(_KPCONV_ROOT) / "models")
if _KPCONV_MODELS_DIR not in sys.path:
    sys.path.insert(0, _KPCONV_MODELS_DIR)

from blocks import block_decider, global_average  # noqa: E402
from utils.config import Config  # noqa: E402
from PointSegDA.Models import segmentation as SegmentationHead  # noqa: E402
from PointDA.Models import RegionReconstruction  # noqa: E402


class PlantKPConvConfig(Config):
    """KPConv hyperparameters for this project's data: fixed N=4096 points,
    normalized to a unit sphere (scripts/augmentations.py::normalize --
    center at origin, scale so max radius = 1.0, giving diameter ~2.0).

    first_subsampling_dl=0.08 -> layer-0 conv radius = 0.08*2.5 = 0.2,
    layer-1 = 0.4 -- chosen to match PointNet2_ClsSeg's own first two SA
    radii (0.2, 0.4 in adapters/models_pointnet2.py) on this SAME
    normalized data, as a cross-check that these are sane absolute values
    for our point spacing (avg NN spacing on a unit sphere at N=4096 is
    ~0.1) rather than an arbitrarily chosen number -- not independently,
    rigorously tuned for KPConv specifically, but grounded in a working
    reference point from this project's own PointNet2 integration.
    in_features_dim=1: no color/normal, just a constant "ones" feature
    per point (the same convention datasets/ModelNet40.py uses for
    color-less inputs) -- KPConv relies on kernel-point geometry, not
    input features, to build its representation.
    first_features_dim=64 (bottleneck ends at 256 after 2 stridings) --
    a resource-conscious default for this first baseline row, not
    exhaustively tuned; comparable in spirit to PointNet2's SA channel
    growth (128->256->1024) but smaller, since our task (2-way species
    cls + small per-species organ seg) needs less capacity than this
    repo's original room-scale multi-class benchmarks.
    architecture: 3 layers (2 'resnetb_strided' poolings), matching
    PointNet2_ClsSeg's 3-level SA/FP depth for a comparable-depth
    cross-backbone architecture. No deformable blocks -- vanilla (rigid)
    KPConv is enough to test this backbone family per CLAUDE.md's
    Backbones section ("KPConv... claimed strength: density-robust due
    to grid-subsampled neighborhoods"); deformable KPConv is a separate,
    optional refinement this row doesn't need.
    """
    dataset = "plant3d"
    num_classes = 2  # overwritten per-use (species cls); kept for Config completeness
    in_points_dim = 3
    in_features_dim = 1
    first_subsampling_dl = 0.08
    conv_radius = 2.5
    deform_radius = 5.0
    KP_extent = 1.0
    KP_influence = "linear"
    aggregation_mode = "sum"
    fixed_kernel_points = "center"
    num_kernel_points = 15
    first_features_dim = 64
    use_batch_norm = True
    batch_norm_momentum = 0.02
    modulated = False
    dropout = 0.5  # overridden per-run from --dropout in the training script, same as the other backbones
    architecture = [
        "simple",
        "resnetb",
        "resnetb_strided",
        "resnetb",
        "resnetb_strided",
        "resnetb",
        "nearest_upsample",
        "unary",
        "nearest_upsample",
        "unary",
    ]


class KPConv_ClsSeg(nn.Module):
    def __init__(self, config, num_class: int, seg_num_classes: dict):
        """seg_num_classes: {species_name: num_organ_classes}, e.g.
        {"Tomato": 3, "Maize": 6}. Mirrors KPFCNN.__init__'s encoder/
        decoder block construction (see module docstring) plus a
        classification head off the bottleneck's globally-pooled feature."""
        super().__init__()
        self.config = config

        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim

        self.encoder_blocks = nn.ModuleList()
        self.encoder_skips = []
        self.encoder_skip_dims = []

        for block_i, block in enumerate(config.architecture):
            if any(tmp in block for tmp in ["pool", "strided", "upsample", "global"]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
            if "upsample" in block:
                break
            self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))
            in_dim = out_dim // 2 if "simple" in block else out_dim
            if "pool" in block or "strided" in block:
                layer += 1
                r *= 2
                out_dim *= 2

        bottleneck_dim = in_dim  # feature width entering the (absent) first decoder block

        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []
        start_i = next(i for i, b in enumerate(config.architecture) if "upsample" in b)
        for block_i, block in enumerate(config.architecture[start_i:]):
            if block_i > 0 and "upsample" in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)
            self.decoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))
            in_dim = out_dim
            if "upsample" in block:
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        seg_feat_dim = out_dim  # final decoder output width, at full (layer-0) resolution

        # Classification head, off the bottleneck's pooled global feature
        # -- mirrors KPCNN's 'global_average' + head_mlp/head_softmax, just
        # composed alongside the segmentation decoder in one forward pass.
        self.cls_fc1 = nn.Linear(bottleneck_dim, 512)
        self.cls_bn1 = nn.BatchNorm1d(512)
        self.cls_drop1 = nn.Dropout(config.dropout)
        self.cls_fc2 = nn.Linear(512, 256)
        self.cls_bn2 = nn.BatchNorm1d(256)
        self.cls_drop2 = nn.Dropout(config.dropout)
        self.cls_fc3 = nn.Linear(256, num_class)

        self.seg_heads = nn.ModuleDict({
            species: SegmentationHead(config, input_size=seg_feat_dim, num_classes=n)
            for species, n in seg_num_classes.items()
        })

        # DefRec (DA-S, row B3b) reconstruction head -- same generic per-point Conv1d stack
        # DGCNN_ClsSeg/PointNet2_ClsSeg both already use via PointDA.Models.RegionReconstruction,
        # fed the same already-computed seg_feat tensor (already exactly DefRec's expected
        # (B, C, N) input shape), mirroring PointNet2_ClsSeg's composition exactly (see
        # adapters/train_b3b_kpconv_da_s.py's docstring for why KPConv additionally needs a
        # batch-rebuild step DGCNN/PointNet2 don't -- this head itself is identical either way).
        # RegionReconstruction only reads `config.dropout`, already present on PlantKPConvConfig.
        self.DefRec = RegionReconstruction(config, seg_feat_dim)

    def forward(self, batch, activate_DefRec: bool = False):
        """batch: adapters.kpconv_collate.KPConvBatch. Returns
        {"cls": (B, num_class) raw logits, "feat": (B, bottleneck_dim)
        pooled global feature (pre-classifier, for future DA-A rows on
        this backbone, mirroring logits["feat"] on the other two
        backbones), "seg_feat": (B, seg_feat_dim, N) per-point features,
        reshaped from KPConv's native stacked (total_points, C) layout --
        see module docstring for why this reshape is exact, not
        approximate, given our fixed N per sample, and, when
        activate_DefRec=True, "DefRec": (B, N, 3) reconstructed points
        (RegionReconstruction's own output convention -- permuted to
        (B, N, 3) internally). B1/B2 callers that never set
        activate_DefRec=True are unaffected."""
        x = batch.features.clone().detach()

        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        bottleneck_feat = global_average(x, batch.lengths[-1])  # (B, bottleneck_dim)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        # x is now (total_points, seg_feat_dim) at full (layer-0) resolution

        logits = {}
        c = self.cls_drop1(torch.relu(self.cls_bn1(self.cls_fc1(bottleneck_feat))))
        c = self.cls_drop2(torch.relu(self.cls_bn2(self.cls_fc2(c))))
        logits["cls"] = self.cls_fc3(c)
        logits["feat"] = bottleneck_feat

        n_per_cloud = batch.lengths[0]  # (B,), always N (e.g. 4096) for every sample here
        B = n_per_cloud.shape[0]
        N = int(n_per_cloud[0].item())
        logits["seg_feat"] = x.view(B, N, -1).permute(0, 2, 1).contiguous()  # (B, C, N)

        if activate_DefRec:
            logits["DefRec"] = self.DefRec(logits["seg_feat"])

        return logits

    def seg_logits_for_species(self, seg_feat: torch.Tensor, species: str) -> torch.Tensor:
        """Same contract as DGCNN_ClsSeg/PointNet2_ClsSeg's method of the
        same name: seg_feat is (B, C, N) already selected for one species,
        returns (B, N, num_classes_for_species)."""
        return self.seg_heads[species](seg_feat)
