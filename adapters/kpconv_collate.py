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

`neighborhood_limits` is NOT left uncapped -- see `calibrate_neighborhood_limits` below.
An earlier version of this module left it as an empty list (`PointCloudDataset.
big_neighborhood_filter` treats that as a no-op, returning neighbor index matrices
uncapped/unfiltered), reasoning this was a safe simplification since our clouds (N=4096,
whole small objects) are far smaller than this repo's typical scene-segmentation crops
(10k-100k+ points) where calibration matters most. **That reasoning was wrong in practice**:
job 308845 (batch_size=16, no calibration) stalled for over an hour with zero progress, and a
follow-up debug run crashed with a CUDA OOM (`Tried to allocate 31.86 GiB`) 3 batches in --
some augmented batches have a small fraction of points with pathologically large neighbor
counts even at N=4096. Fixed by calling `calibrate_neighborhood_limits` once at the start of
training and assigning its result to `config.neighborhood_limits` before building any real
DataLoader -- see that function's docstring for the full incident and calibration method (a
simplified version of the reference repo's own `datasets/*.py::calibration` routines). Full
incident writeup in `step_notes/B1_KPConv_DA0.md`.
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
        # PointCloudDataset.__init__ (just ran, via super()) unconditionally resets
        # self.neighborhood_limits to [] -- big_neighborhood_filter reads THIS attribute
        # directly, not self.config.neighborhood_limits, so it must be re-applied here after
        # the super().__init__() call, not just stored on config. Caught via a real CUDA OOM
        # (see calibrate_neighborhood_limits' docstring) -- confirmed by reading
        # datasets/common.py::PointCloudDataset.__init__/big_neighborhood_filter directly
        # rather than assuming config was the right place to store it.
        self.neighborhood_limits = getattr(config, "neighborhood_limits", [])


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


def calibrate_neighborhood_limits(config, dataset, batch_size, num_batches=10, untouched_ratio=0.9,
                                   num_workers=0, collate_fn_factory=None):
    """Computes per-layer neighbor-count caps the same way the reference repo's own dataset
    classes do (e.g. `datasets/ModelNet40.py::calibration`), minus the point-budget/dynamic
    batch-size machinery those use (irrelevant here -- every sample is already a fixed N=4096,
    so a plain fixed `batch_size` DataLoader is used for calibration too, not their custom
    point-budget sampler).

    `collate_fn_factory` defaults to `make_kpconv_collate_fn` (cls+seg 3-tuples, e.g.
    `PlantClsSegDataset`/Crops3D). Pass `make_kpconv_collate_fn_cls_only` to calibrate against a
    cls-only 2-tuple dataset instead (e.g. `PlantSpeciesDataset`/Pheno4D's adaptation pool) --
    added for DA-A rows (B2+), where target-domain batches also flow through the same encoder
    and neighbor caps calibrated on source data ALONE could under-cover target-domain density
    characteristics (see `combine_neighborhood_limits` below and the row's own step_notes for
    why source-only calibration was judged insufficient once a second domain enters training).

    Why this exists: leaving `neighborhood_limits` empty (this module's original approach, see
    module docstring) makes `PointCloudDataset.big_neighborhood_filter` a no-op, returning
    neighbor index matrices uncapped. Flagged as a "candidate follow-up if runtime/memory become
    a problem" in step_notes/B1_KPConv_DA0.md -- it became one: job 308845 (batch_size=16)
    appeared to hang for over an hour with zero visible progress (no per-batch logging existed
    yet), and a follow-up debug run with per-batch timing added (job 308884) crashed with a CUDA
    OOM (`Tried to allocate 31.86 GiB`) inside `KPConv.forward`'s `sq_distances = torch.sum(
    differences ** 2, dim=3)` on only the 3rd training batch -- a small fraction of points in
    some augmented batches have uncapped neighbor counts large enough to blow up that tensor.
    Both symptoms trace to the same missing safeguard: an unbounded neighbor matrix width, either
    slow to construct on CPU (many-neighbor pathological batches) or too large to use on GPU.

    Uses the SAME uncapped collate function to gather a neighbor-count histogram per layer over
    `num_batches` real (augmented) training batches, then sets each layer's limit to the count at
    the `untouched_ratio` (default 0.9, matching the reference repo's own default) percentile of
    the observed distribution -- i.e. at most ~10% of points have their neighbor list truncated,
    the same accepted tradeoff every one of the reference repo's own example configs makes.
    Building the histogram itself never touches the GPU (pure CPU/numpy), so it is safe from the
    same OOM even if a pathological batch is sampled during calibration -- worst case, one
    particular batch's neighbor-matrix construction is slow, not GPU-fatal.

    Returns a plain Python list of per-layer ints, meant to be assigned to
    `config.neighborhood_limits` before building the REAL training/val/eval DataLoaders (so every
    subsequent collate call reuses these fixed caps, not just the calibration pass)."""
    from torch.utils.data import DataLoader

    collate_fn_factory = collate_fn_factory or make_kpconv_collate_fn

    calib_config = type(config)()
    calib_config.__dict__.update(config.__dict__)
    calib_config.neighborhood_limits = []  # uncapped, for measuring the true distribution

    calib_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               collate_fn=collate_fn_factory(calib_config))

    hist_n = 500  # generous upper bound on neighbors/point for this project's dense point clouds
    neighb_hists = None
    batches_seen = 0
    for batch in calib_loader:
        counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1)
                  for neighb_mat in batch.neighbors]
        if neighb_hists is None:
            neighb_hists = np.zeros((len(counts), hist_n), dtype=np.int64)
        hists = [np.bincount(np.clip(c, 0, hist_n - 1), minlength=hist_n) for c in counts]
        neighb_hists += np.vstack(hists)
        batches_seen += 1
        if batches_seen >= num_batches:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
    return percentiles.tolist()


def combine_neighborhood_limits(*limits_lists):
    """Elementwise max across two or more `calibrate_neighborhood_limits` results (same number
    of layers each) -- used by DA-A/DA-D/DA-O rows (any row where target-domain batches flow
    through the same encoder as source) to make sure the applied cap is wide enough for whichever
    domain has the wider neighbor distribution at each layer, not just source. Cheap: this only
    changes which (already bounded) cap gets used, never re-introduces the uncapped-by-default
    risk `calibrate_neighborhood_limits` exists to fix."""
    return [max(vals) for vals in zip(*limits_lists)]


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


def batch_points_bcn(batch: "KPConvBatch") -> torch.Tensor:
    """Extracts a batch's layer-0 (finest, full-resolution) points as a plain `(B, 3, N)`
    tensor -- the counterpart to `build_batch_from_points` below, and the same fixed-N,
    order-preserving reshape `models_kpconv.py::KPConv_ClsSeg.forward` already relies on for its
    `(B, C, N)` seg_feat reshape (see that module's docstring for why this is exact, not
    approximate, given this project's fixed N=4096 per sample)."""
    pts_flat = batch.points[0]  # (B*N, 3)
    B = batch.lengths[0].shape[0]
    N = int(batch.lengths[0][0].item())
    return pts_flat.view(B, N, 3).permute(0, 2, 1).contiguous()  # (B, 3, N)


def build_batch_from_points(config, points_bcn: torch.Tensor, batch_builder=None) -> "KPConvBatch":
    """Rebuilds a fresh `KPConvBatch` (full multi-layer neighbor/pool/upsample structure,
    via the same `segmentation_inputs` call every other collate function here uses) from a
    plain `(B, 3, N)` point tensor -- e.g. the output of `DefRec_and_PCM.DefRec.deform_input`,
    which operates on raw `(B, C, N)` tensors and knows nothing about KPConv's batch format.

    Why this exists (DA-S / row B3b only): DGCNN/PointNet2 take a raw `(B, 3, N)` tensor
    directly as their `forward` input, so DefRec's "deform, then feed through the model" recipe
    needs no extra plumbing for those two backbones -- the model's own forward pass internally
    recomputes whatever geometric structure it needs (farthest-point sampling, ball query, k-NN)
    from the deformed coordinates automatically, every call. KPConv is architecturally different:
    its multi-layer neighbor/pool/upsample structure is precomputed OUTSIDE the model, by this
    module's collate functions, from a specific set of point coordinates -- feeding deformed
    points into an already-built `KPConvBatch` would silently reuse the ORIGINAL (undeformed)
    geometry's neighbor structure, not a bug exactly, but not a faithful analogue of what
    happens automatically for the other two backbones. This helper makes the KPConv path do the
    same thing DGCNN/PointNet2 do implicitly: rebuild geometric structure from whatever
    coordinates are being reconstructed, every time.

    Uses `config.neighborhood_limits` (already calibrated, whatever the caller set it to) via
    `KPConvBatchBuilder` -- no separate calibration needed for deformed geometry: since
    `PointCloudDataset.big_neighborhood_filter` always hard-slices `neighbors[:, :limit]`
    regardless of the true underlying count, this is safe from the same CUDA OOM `calibrate_
    neighborhood_limits` exists to prevent even if deformation happens to produce locally denser
    configurations than calibration observed (an inference, not measured directly, but load-
    bearing: the slice is unconditional, not a probabilistic percentile guarantee).

    `stacked_labels`/`species_labels` are dummy placeholders (never read) -- same convention as
    `make_kpconv_collate_fn_cls_only` -- since DefRec only needs `batch.points/neighbors/pools/
    upsamples/lengths/features` for the forward pass, confirmed by reading `KPConv_ClsSeg.
    forward` directly (it never touches `batch.labels`/`batch.species_labels`)."""
    batch_builder = batch_builder or KPConvBatchBuilder(config)

    points_bnc = points_bcn.permute(0, 2, 1).contiguous()  # (B, N, 3)
    B, N, _ = points_bnc.shape
    stack_lengths = np.full((B,), N, dtype=np.int32)
    stacked_points = points_bnc.reshape(B * N, 3).detach().cpu().numpy().astype(np.float32)
    stacked_labels = np.zeros((B * N,), dtype=np.int64)  # placeholder, never read
    stacked_features = np.ones((B * N, 1), dtype=np.float32)

    input_list = batch_builder.segmentation_inputs(
        stacked_points, stacked_features, stacked_labels, stack_lengths)

    species_labels = np.zeros((B,), dtype=np.int64)  # placeholder, never read
    return KPConvBatch(input_list, species_labels)


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
