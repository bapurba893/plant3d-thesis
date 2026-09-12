"""
kpconv_collate.py

Builds the multi-layer "stacked batch" input format KPConv-PyTorch's
block classes expect (models/blocks.py: SimpleBlock/ResnetBottleneckBlock/
NearestUpsampleBlock/GlobalAverageBlock all read `batch.points[layer]`/
`batch.neighbors[layer]`/`batch.pools[layer]`/`batch.upsamples[layer]`/
`batch.lengths[layer]`) -- a fundamentally different data representation
than DGCNN/PointNet2's fixed (B, 3, N) tensors: KPConv concatenates every
sample's points into one flat (total_points, 3) array per layer, using
`stack_lengths` to record where each sample's points start/end, plus
precomputed neighbor/pooling/upsampling index tables built by the
compiled C++ extensions (cpp_wrappers/, see PointNet2_Integration.md-style
compatibility notes in step_notes/B1_KPConv_DA0.md for how those were
made to compile under numpy>=2 / modern setuptools).

Reuses `KPConv_and_PyTorch.datasets.common.PointCloudDataset.segmentation_inputs`
(and the `batch_neighbors`/`batch_grid_subsampling` helpers it calls, which
wrap the compiled extensions) completely unmodified -- the same
multi-layer neighbor/pool/upsample construction every one of the
reference repo's own dataset classes (S3DIS, ModelNet40, ...) uses.
`KPConvBatchBuilder` is a minimal `PointCloudDataset` subclass whose only
job is holding `self.config` so `segmentation_inputs` can read
`config.architecture`/`config.conv_radius`/etc. -- not a real Dataset,
never used for `__len__`/`__getitem__`, just a legitimate, intended way to
reuse that bound method (every real dataset class in the reference repo
does the same subclassing to get access to it).

`KPConvBatch` is our own, much simpler batch container than the
reference repo's per-dataset CustomBatch classes (e.g. S3DISCustomBatch) --
those also carry scale/rotation/crop-index bookkeeping needed for
room-scale scene segmentation with sliding-window sampling, none of which
applies here (every one of our point clouds is already a fixed, whole-object
N=4096 array from Stage-0 preprocessing, no cropping/re-centering at
collate time). Adds one field the reference repo's segmentation batches
don't carry: `species_labels` (whole-cloud classification target,
alongside the per-point `labels` segmentation targets already returned
by `segmentation_inputs`) -- since every row in this project needs joint
cls+seg, unlike the reference repo's own single-task datasets.

`neighborhood_limits` is left as an empty list (skips the reference
repo's neighbor-count calibration pass -- see `PointCloudDataset.
big_neighborhood_filter`: an empty list makes it a no-op, returning
neighbor index matrices uncapped/unfiltered rather than incorrect).
This is a deliberate simplification for this row's first baseline, not a
correctness issue -- calibration only bounds memory/compute by trimming
each neighbor matrix to its needed width; skipping it just means those
matrices may be wider than strictly necessary. Documented as a candidate
follow-up in step_notes/B1_KPConv_DA0.md if runtime/memory become a
problem, since our clouds (N=4096, whole small objects) are far smaller
than this repo's typical scene-segmentation crops (10k-100k+ points),
where calibration matters much more.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KPCONV_ROOT = os.environ.get("KPCONV_ROOT", str(_REPO_ROOT.parent / "KPConv-PyTorch"))
if _KPCONV_ROOT not in sys.path:
    sys.path.insert(0, _KPCONV_ROOT)

from datasets.common import PointCloudDataset  # noqa: E402


class KPConvBatchBuilder(PointCloudDataset):
    """Not a real Dataset (never __len__/__getitem__'d) -- exists purely
    to bind `self.config` so the inherited `segmentation_inputs` method
    can read the architecture/radius parameters it needs. Same reuse
    pattern every real dataset class in the reference repo (S3DISDataset,
    ModelNet40Dataset, ...) already uses."""

    def __init__(self, config):
        super().__init__(config.dataset if hasattr(config, "dataset") else "plant3d")
        self.config = config


class KPConvBatch:
    """Our own lightweight batch container -- see module docstring for how
    this differs from the reference repo's per-dataset CustomBatch classes."""

    def __init__(self, input_list, species_labels):
        L = (len(input_list) - 2) // 5
        ind = 0
        self.points = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.neighbors = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.pools = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.upsamples = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.lengths = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.features = torch.from_numpy(input_list[ind]); ind += 1
        self.labels = torch.from_numpy(input_list[ind])  # per-point seg labels, flat (total_points,)
        self.species_labels = torch.from_numpy(species_labels)  # (B,) whole-cloud cls targets

    def to(self, device):
        self.points = [t.to(device) for t in self.points]
        self.neighbors = [t.to(device) for t in self.neighbors]
        self.pools = [t.to(device) for t in self.pools]
        self.upsamples = [t.to(device) for t in self.upsamples]
        self.lengths = [t.to(device) for t in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.species_labels = self.species_labels.to(device)
        return self


def reshape_seg_labels(batch: "KPConvBatch") -> torch.Tensor:
    """batch.labels is a flat (total_points,) tensor (B contiguous chunks
    of N each, per `KPConvBatch`/collate construction). Reshapes to
    (B, N) to match `logits["seg_feat"]`'s (B, C, N) layout for the
    per-species seg-loss/eval helpers every other row's training script
    already uses unmodified."""
    B = batch.species_labels.shape[0]
    N = int(batch.lengths[0][0].item())
    return batch.labels.view(B, N)


def make_kpconv_collate_fn(config):
    """Returns a collate_fn closing over `config` (needed by
    KPConvBatchBuilder), for use as `DataLoader(..., collate_fn=...)`.
    Expects each sample from the underlying Dataset (e.g.
    `PlantClsSegDataset`) as (pts [N,3] float32, species_label int64,
    seg_label [N] int64) -- exactly what every other row's dataset already
    returns, so no new Dataset class was needed for this row, only a new
    collate_fn."""
    batch_builder = KPConvBatchBuilder(config)

    def collate_fn(samples):
        pts_list, species_list, seg_list = zip(*samples)
        stack_lengths = np.array([len(p) for p in pts_list], dtype=np.int32)
        stacked_points = np.concatenate(pts_list, axis=0).astype(np.float32)
        stacked_labels = np.concatenate(seg_list, axis=0).astype(np.int64)
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)  # in_features_dim=1, geometry-only

        input_list = batch_builder.segmentation_inputs(
            stacked_points, stacked_features, stacked_labels, stack_lengths)

        species_labels = np.array(species_list, dtype=np.int64)
        return KPConvBatch(input_list, species_labels)

    return collate_fn


def make_kpconv_collate_fn_cls_only(config):
    """Same as `make_kpconv_collate_fn`, but for a classification-only
    Dataset (e.g. `PlantSpeciesDataset`, used for Pheno4D held-out eval --
    every other row's target-domain-gap indicator) that returns
    (pts [N,3] float32, species_label int64) 2-tuples with no per-point
    segmentation labels at all.

    `KPConv_ClsSeg.forward` always runs its full encoder+decoder (needed
    for the joint model's architecture, even when the caller only reads
    `logits["cls"]`), so `segmentation_inputs` still needs SOME per-point
    label array to build the batch structure -- this fabricates an
    all-zero placeholder per sample. It is structurally required but
    never semantically used: this collate path is only ever consumed by
    classification-only eval code that reads `batch.species_labels`
    against `logits["cls"]` and never touches `batch.labels` or
    `logits["seg_feat"]` at all."""
    batch_builder = KPConvBatchBuilder(config)

    def collate_fn(samples):
        pts_list, species_list = zip(*samples)
        stack_lengths = np.array([len(p) for p in pts_list], dtype=np.int32)
        stacked_points = np.concatenate(pts_list, axis=0).astype(np.float32)
        stacked_labels = np.zeros((stacked_points.shape[0],), dtype=np.int64)  # placeholder, never read
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)

        input_list = batch_builder.segmentation_inputs(
            stacked_points, stacked_features, stacked_labels, stack_lengths)

        species_labels = np.array(species_list, dtype=np.int64)
        return KPConvBatch(input_list, species_labels)

    return collate_fn
