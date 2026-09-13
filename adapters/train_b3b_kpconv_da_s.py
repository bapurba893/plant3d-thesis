"""
train_b3b_kpconv_da_s.py

Strategy table row B3b: KPConv backbone, DA-S (self-supervised deformation reconstruction,
DefRec-style), ALL augmentation, L_cls + L_seg + L_defrec -- the KPConv-track counterpart of
adapters/train_c3_dgcnn_da_s.py (row C3) and adapters/train_a3_pointnet2_da_s.py (row A3).

**Why "B3b" and not "B3":** per CLAUDE.md's original strategy table, Block B's third row (B3)
is DA-D (discrepancy-based, Deep CORAL/MMD), not DA-S -- Block B was specced differently from
Block A/C from the start, using B4 (DA-0/L-D) as its density-robustness check instead of a DA-S
row. This row is an EXPLICIT ADDITION to the documented plan (per user instruction, 2026-09-13),
run for direct comparability with A3/C3, alongside B3 (DA-D, still to be run separately, per the
original plan) rather than in place of it. See CLAUDE.md's strategy table section for the
updated, explicit documentation of this addition.

Purpose of this row: A3 (PointNet++, DA-S) reversed C3's finding entirely -- DefRec HURT DGCNN's
target-classification transfer but HELPED PointNet++'s, dramatically (see
step_notes/A3_PointNet2_DA_S.md). B2 (KPConv, DA-A) then showed a second architecture-dependent
reversal: DANN hurt DGCNN and PointNet++ (to different degrees) but did not hurt KPConv at all
overall (see step_notes/B2_KPConv_DA_A.md). This row asks the natural next question: does KPConv
also flip DA-S's effect the way it flipped DA-A's, or does DA-S behave on KPConv the way it did
on one of the other two backbones?

**The one real, KPConv-specific engineering problem this row had to solve, beyond a straight
backbone swap:** DGCNN/PointNet2 take a raw `(B, 3, N)` tensor directly as `forward`'s input, so
DefRec's "deform the input, feed the deformed tensor through the model" recipe needs no extra
plumbing for those two backbones -- their own forward pass internally recomputes whatever
geometric structure it needs (farthest-point sampling, ball query, k-NN) from the deformed
coordinates automatically, every call. KPConv is architecturally different: its multi-layer
neighbor/pool/upsample structure is precomputed OUTSIDE the model, by `adapters/kpconv_collate.py`'s
collate functions, from a specific, fixed set of point coordinates -- simply overwriting an
already-built `KPConvBatch`'s point values with deformed coordinates would silently keep the
ORIGINAL (undeformed) geometry's neighbor structure, not a faithful analogue of what happens
automatically for the other two backbones (whose forward pass genuinely recomputes structure
from the deformed geometry every time). Fixed by adding two small, symmetric helpers to
`adapters/kpconv_collate.py`: `batch_points_bcn` (extract a clean batch's raw `(B,3,N)` points)
and `build_batch_from_points` (rebuild a fresh, fully-structured `KPConvBatch` from a `(B,3,N)`
tensor, deformed or not) -- so KPConv's DA-S path does the same thing DGCNN/PointNet2 do
implicitly: rebuild geometric structure from whatever coordinates are being reconstructed, every
time, not just reuse stale structure with new point values spliced in.

**Real, accepted cost this adds, flagged up front, not discovered mid-run:** this re-collate
happens once per DEFREC_CHUNK-sized chunk (8 samples), per domain, per training step -- on top
of the one clean-source collate B1 already needed and the clean-source+clean-target collate B2
added, this row needs roughly 1 (clean source) + ceil(batch_size/8)*2 (deformed source chunks) +
ceil(batch_size/8)*2 (deformed target chunks) collate calls per step, at batch_size=16 that's
1 + 2 + 2 = 5 collate calls per step, vs. B1's 1 and B2's 2. Expected to be meaningfully slower
than B1/B2's already-fast (2-3h) real-world runtimes, but well within the standing 60h Block B
budget -- monitored with `--verbose_batches`, same discipline as every prior KPConv row.

DefRec's core machinery (`DefRec_and_PCM.DefRec.deform_input`/`.calc_loss`, the chunked-Chamfer-
distance OOM workaround `DEFREC_CHUNK=8`) is fully backbone-agnostic and reused unchanged from
A3/C3 -- it operates on plain `(B, 3, N)` point tensors and a `(B, N, 3)` reconstruction output,
never touching backbone internals; only how those tensors get IN and OUT of KPConv's specific
batch format is new here. `neighborhood_limits` calibrated over BOTH domains (source + target),
same as B2 and for the same reason (target points flow through the encoder here too, via DefRec)
-- reused via `combine_neighborhood_limits`, and the deformed-batch rebuilds reuse this same
already-calibrated `config` (safe from OOM regardless of whether deformation shifts the true
neighbor-count distribution, since `big_neighborhood_filter`'s cap is an unconditional hard
slice, not a probabilistic guarantee -- see `build_batch_from_points`' docstring).

Model selection: same protocol as every prior row -- best epoch by lowest source val total loss
(Kendall(cls, seg, defrec), defrec evaluated on source val data only, never touches target
labels). Full-trajectory analysis from the start, same as every DA-S/DA-A row so far -- see
step_notes/B3b_KPConv_DA_S.md.
"""

import argparse
import copy
import datetime
import os
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from sklearn.metrics import jaccard_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFREC_ROOT = os.environ.get("DEFREC_ROOT", str(_REPO_ROOT.parent / "DefRec_and_PCM"))
if _DEFREC_ROOT not in sys.path:
    sys.path.insert(0, _DEFREC_ROOT)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_kpconv import KPConv_ClsSeg, PlantKPConvConfig  # noqa: E402
from kpconv_collate import (  # noqa: E402
    make_kpconv_collate_fn, make_kpconv_collate_fn_cls_only, reshape_seg_labels,
    calibrate_neighborhood_limits, combine_neighborhood_limits,
    KPConvBatchBuilder, batch_points_bcn, build_batch_from_points,
)
from losses import seg_loss, class_weights_from_counts, KendallUncertaintyWeighting  # noqa: E402
from dataset import (  # noqa: E402
    PlantClsSegDataset, PlantSpeciesDataset,
    IDX_TO_SPECIES, SPECIES_TO_IDX, SEG_NUM_CLASSES,
)
from DefRec_and_PCM import DefRec  # noqa: E402
from utils import pc_utils  # noqa: E402

LAMBDA_LOV = 1.0  # fixed per the strategy table, never learned/swept
# Same chunking workaround as A3/C3 (see those files' docstrings for the CUDA-OOM root cause) --
# KPConv's per-point tensors are the same (B, N, ...) shape/density as DGCNN/PointNet2's, so the
# same Chamfer-distance memory pressure applies once points flow through DefRec.calc_loss.
DEFREC_CHUNK = 8


class IOStream:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.f = open(os.path.join(path, "run.log"), "a")

    def cprint(self, text):
        ts = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
        line = f"{ts}: {text}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def parse_args():
    p = argparse.ArgumentParser(
        description="Row B3b: KPConv, DA-S (self-supervised DefRec), ALL augmentation, "
                    "L_cls+L_seg+L_defrec")
    p.add_argument("--exp_name", type=str, default="B3b_kpconv_da_s")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--DefRec_dist", type=str, default="volume_based_voxels",
                    choices=["volume_based_voxels", "volume_based_radius"])
    p.add_argument("--num_regions", type=int, default=3)
    p.add_argument("--DefRec_weight", type=float, default=1.0,
                    help="neutral (Kendall handles the cls/seg/defrec tradeoff) -- see "
                         "train_c3_dgcnn_da_s.py's docstring for why this differs from the "
                         "upstream default 0.5")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0, help="-1 for cpu")
    p.add_argument("--verbose_batches", action="store_true",
                    help="log per-batch collate/step timing -- standing KPConv-row convention "
                         "since B1's silent-stall incident, extra important here given this "
                         "row's higher expected collate-call count per step (see module docstring)")
    p.add_argument("--num_workers", type=int, default=0,
                    help="kept at the safe default -- B1 found DataLoader(num_workers>0) made "
                         "things worse with these compiled C++ extensions")
    return p.parse_args()


def compute_seg_loss(model, seg_feat, species_labels, seg_labels, seg_class_weights):
    losses, weights = [], []
    for species, idx in SPECIES_TO_IDX.items():
        mask = species_labels == idx
        n = int(mask.sum().item())
        if n == 0:
            continue
        sp_logits = model.seg_logits_for_species(seg_feat[mask], species)
        sp_loss = seg_loss(sp_logits, seg_labels[mask], seg_class_weights[species], LAMBDA_LOV)
        losses.append(sp_loss * n)
        weights.append(n)
    return sum(losses) / sum(weights)


def seg_eval_metrics(model, seg_feat, species_labels, seg_labels, seg_class_weights, stats):
    for species, idx in SPECIES_TO_IDX.items():
        mask = species_labels == idx
        n = int(mask.sum().item())
        if n == 0:
            continue
        sp_logits = model.seg_logits_for_species(seg_feat[mask], species)
        sp_labels = seg_labels[mask]
        sp_loss = seg_loss(sp_logits, sp_labels, seg_class_weights[species], LAMBDA_LOV)
        preds = sp_logits.max(dim=2)[1]
        num_classes = SEG_NUM_CLASSES[species]
        for b in range(n):
            y_true = sp_labels[b].cpu().numpy()
            y_pred = preds[b].cpu().numpy()
            miou = jaccard_score(y_true, y_pred, labels=list(range(num_classes)),
                                  average="macro", zero_division=0)
            stats[species]["mIoU_sum"] += miou
            stats[species]["acc_sum"] += float((y_true == y_pred).mean())
            stats[species]["n"] += 1
        stats[species]["loss_sum"] += sp_loss.item() * n


def _iter_chunks(pts, chunk_size):
    n = pts.size(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield pts[start:end], end - start


def defrec_loss_for_chunk(args, model, lookup, config, batch_builder, pts_orig_chunk, device):
    """Deforms pts_orig_chunk (B_chunk, 3, N), rebuilds a fresh KPConvBatch from the deformed
    geometry (see build_batch_from_points' docstring for why KPConv needs this rebuild step
    unlike DGCNN/PointNet2), forwards it with activate_DefRec=True, returns the scalar Chamfer
    reconstruction loss against the ORIGINAL (undeformed) points."""
    pts_deformed, mask = DefRec.deform_input(
        pts_orig_chunk.clone(), lookup, args.DefRec_dist, device)
    deformed_batch = build_batch_from_points(config, pts_deformed, batch_builder).to(device)
    logits_deform = model(deformed_batch, activate_DefRec=True)
    return DefRec.calc_loss(args, logits_deform, pts_orig_chunk, mask)


def defrec_chunked_backward(args, model, lookup, kendall, config, batch_builder,
                             src_pts_bcn, trgt_pts_bcn, device, chunk_size=DEFREC_CHUNK):
    """Computes L_defrec (DefRec on both domains, sample-count-weighted into one scalar per
    CLAUDE.md's DA-S definition) and immediately backprops it, one small chunk at a time, to
    bound peak CUDA memory. Identical logic/mathematical-equivalence argument as A3/C3's helper
    of the same name -- only the per-chunk batch construction (via defrec_loss_for_chunk) differs."""
    bs_src, bs_trgt = src_pts_bcn.size(0), trgt_pts_bcn.size(0)
    total_n = bs_src + bs_trgt
    chunks = list(_iter_chunks(src_pts_bcn, chunk_size)) + list(_iter_chunks(trgt_pts_bcn, chunk_size))
    num_chunks = len(chunks)
    s_defrec = kendall.log_vars["defrec"]

    loss_sum, n_sum = 0.0, 0
    for chunk_pts, chunk_n in chunks:
        chunk_loss = defrec_loss_for_chunk(args, model, lookup, config, batch_builder,
                                            chunk_pts, device)
        weighted_chunk = (torch.exp(-s_defrec) / 2.0 * chunk_loss * (chunk_n / total_n)
                           + s_defrec / (2.0 * num_chunks))
        weighted_chunk.backward()
        loss_sum += chunk_loss.item() * chunk_n
        n_sum += chunk_n
    return loss_sum / n_sum


def evaluate_defrec(model, loader, args, lookup, config, batch_builder, device):
    """Source-val-only DefRec reconstruction loss, sample-count-weighted mean over the loader.
    Same as A3/C3's helper of the same name."""
    model.eval()
    loss_sum, n_seen = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pts_bcn = batch_points_bcn(batch)
            for chunk_pts, chunk_n in _iter_chunks(pts_bcn, DEFREC_CHUNK):
                l = defrec_loss_for_chunk(args, model, lookup, config, batch_builder,
                                           chunk_pts, device)
                loss_sum += l.item() * chunk_n
                n_seen += chunk_n
    return loss_sum / n_seen


def evaluate_cls_seg(model, loader, cls_criterion, seg_class_weights, device):
    model.eval()
    n_seen, cls_loss_sum = 0, 0.0
    all_true_cls, all_pred_cls = [], []
    seg_stats = {sp: {"loss_sum": 0.0, "mIoU_sum": 0.0, "acc_sum": 0.0, "n": 0}
                 for sp in SPECIES_TO_IDX}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            species_labels = batch.species_labels
            seg_labels = reshape_seg_labels(batch)

            logits = model(batch)
            l_cls = cls_criterion(logits["cls"], species_labels)
            bs = species_labels.size(0)
            cls_loss_sum += l_cls.item() * bs
            preds_cls = logits["cls"].max(dim=1)[1]
            all_true_cls.append(species_labels.cpu().numpy())
            all_pred_cls.append(preds_cls.cpu().numpy())
            n_seen += bs

            seg_eval_metrics(model, logits["seg_feat"], species_labels, seg_labels,
                              seg_class_weights, seg_stats)

    all_true_cls = np.concatenate(all_true_cls)
    all_pred_cls = np.concatenate(all_pred_cls)
    cls_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
    cls_avg_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
    cls_loss = cls_loss_sum / n_seen

    seg_summary, seg_loss_weighted_sum, seg_n_total = {}, 0.0, 0
    for species, s in seg_stats.items():
        if s["n"] == 0:
            continue
        seg_summary[species] = {"loss": s["loss_sum"] / s["n"],
                                 "mIoU": s["mIoU_sum"] / s["n"],
                                 "acc": s["acc_sum"] / s["n"]}
        seg_loss_weighted_sum += s["loss_sum"]
        seg_n_total += s["n"]
    seg_loss_combined = seg_loss_weighted_sum / seg_n_total if seg_n_total else 0.0

    return cls_acc, cls_avg_acc, cls_loss, seg_summary, seg_loss_combined


def evaluate_cls_only(model, loader, criterion, device):
    """Classification-only eval for Pheno4D (target) -- domain-gap indicator, never used for
    training or model selection. Same as every prior row's held-out eval."""
    model.eval()
    losses, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels = batch.species_labels
            logits = model(batch)
            loss = criterion(logits["cls"], labels)
            losses.append(loss.item() * labels.size(0))
            preds = logits["cls"].max(dim=1)[1]
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    acc = metrics.accuracy_score(all_true, all_pred)
    avg_acc = metrics.balanced_accuracy_score(all_true, all_pred)
    mean_loss = sum(losses) / len(all_true)
    conf = metrics.confusion_matrix(all_true, all_pred, labels=list(IDX_TO_SPECIES.keys()))
    return acc, avg_acc, mean_loss, conf, all_true, all_pred


def main():
    args = parse_args()
    exp_path = os.path.join(args.out_path, args.exp_name)
    io = IOStream(exp_path)
    io.cprint(str(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    if cuda:
        torch.cuda.manual_seed_all(args.seed)
        io.cprint(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        io.cprint("Using CPU")

    data_dir = _REPO_ROOT / "data"
    manifest_csv = data_dir / "preprocessed" / "preprocessed_manifest.csv"
    src_train_set = PlantClsSegDataset(
        data_dir / "crops3d_train.csv", manifest_csv, augment=True, seed=args.seed)
    src_val_set = PlantClsSegDataset(
        data_dir / "crops3d_val.csv", manifest_csv, augment=False)
    trgt_adapt_set = PlantSpeciesDataset(
        data_dir / "pheno4d_adaptation_pool.csv", manifest_csv, augment=True, seed=args.seed)
    trgt_eval_set = PlantSpeciesDataset(
        data_dir / "pheno4d_heldout_eval.csv", manifest_csv, augment=False)

    io.cprint(f"Crops3D train (source): {len(src_train_set)} {src_train_set.class_counts()}")
    io.cprint(f"Crops3D val (source):   {len(src_val_set)} {src_val_set.class_counts()}")
    io.cprint(f"Pheno4D adaptation pool (target, DefRec-only -- self-supervised, no labels "
              f"used, species shown for bookkeeping only): "
              f"{len(trgt_adapt_set)} {trgt_adapt_set.class_counts()}")
    io.cprint(f"Pheno4D held-out eval (target, never trained on, cls-only): "
              f"{len(trgt_eval_set)} {trgt_eval_set.class_counts()}")

    counts = src_train_set.class_counts()
    cls_counts_t = torch.tensor([counts.get(sp, 0) for sp in IDX_TO_SPECIES.values()],
                                 dtype=torch.float32)
    cls_class_weights = class_weights_from_counts(cls_counts_t).to(device)
    io.cprint(f"L_cls class weights (w_c=(f_c+eps)^-1, normalized): "
              f"{dict(zip(IDX_TO_SPECIES.values(), cls_class_weights.tolist()))}")

    seg_class_weights = {}
    for species in SEG_NUM_CLASSES:
        hist = src_train_set.seg_class_histogram(species)
        w = class_weights_from_counts(torch.tensor(hist, dtype=torch.float32)).to(device)
        seg_class_weights[species] = w
        io.cprint(f"L_seg class weights for {species} (counts={hist.tolist()}): {w.tolist()}")

    config = PlantKPConvConfig()
    config.dropout = args.dropout

    # Calibrate over BOTH domains, same as B2 and for the same reason (target points flow
    # through the encoder here too, via DefRec) -- see B2's step_notes for the full incident
    # this guards against.
    src_limits = calibrate_neighborhood_limits(
        config, src_train_set, args.batch_size, num_workers=args.num_workers)
    trgt_limits = calibrate_neighborhood_limits(
        config, trgt_adapt_set, args.batch_size, num_workers=args.num_workers,
        collate_fn_factory=make_kpconv_collate_fn_cls_only)
    config.neighborhood_limits = combine_neighborhood_limits(src_limits, trgt_limits)
    io.cprint(f"Calibrated neighborhood_limits: source-only={src_limits}, "
              f"target-only={trgt_limits}, combined (elementwise max, used)="
              f"{config.neighborhood_limits}")

    src_train_loader = DataLoader(src_train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True,
                                   collate_fn=make_kpconv_collate_fn(config))
    src_val_loader = DataLoader(src_val_set, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=make_kpconv_collate_fn(config))
    trgt_adapt_loader = DataLoader(trgt_adapt_set, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, drop_last=True,
                                    collate_fn=make_kpconv_collate_fn_cls_only(config))
    trgt_eval_loader = DataLoader(trgt_eval_set, batch_size=args.test_batch_size, shuffle=False,
                                   num_workers=args.num_workers,
                                   collate_fn=make_kpconv_collate_fn_cls_only(config))

    n_batches_per_epoch = len(src_train_loader)
    io.cprint(f"Batches/epoch: source={n_batches_per_epoch} (drives epoch length), "
              f"target_adapt={len(trgt_adapt_loader)} (cycled)")

    # Shared batch builder for every deformed-batch rebuild (DefRec chunks) -- constructed once,
    # reused across the whole run, same as the collate functions' own internal builders.
    defrec_batch_builder = KPConvBatchBuilder(config)

    lookup = torch.Tensor(pc_utils.region_mean(args.num_regions)).to(device)

    model = KPConv_ClsSeg(config, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    kendall = KendallUncertaintyWeighting(
        {"cls": "classification", "seg": "classification", "defrec": "regression"}).to(device)
    best_model = copy.deepcopy(model)
    best_kendall = copy.deepcopy(kendall)

    opt = optim.Adam(list(model.parameters()) + list(kendall.parameters()),
                      lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(opt, args.epochs)
    cls_criterion = nn.CrossEntropyLoss(weight=cls_class_weights)
    target_eval_criterion = nn.CrossEntropyLoss()

    best_val_total_loss, best_epoch = float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        train_cls_loss, train_seg_loss, train_defrec_loss, n_seen = 0.0, 0.0, 0.0, 0
        t_batch_start = time.time()

        trgt_iter = cycle(trgt_adapt_loader)
        for batch_idx, (src_batch, trgt_batch) in enumerate(zip(src_train_loader, trgt_iter)):
            t_collate_done = time.time()
            src_batch = src_batch.to(device)
            src_species = src_batch.species_labels
            src_seg = reshape_seg_labels(src_batch)
            trgt_batch = trgt_batch.to(device)
            # target species labels intentionally never read below (DA-S never uses target
            # labels, same discipline as A3/C3)

            opt.zero_grad()

            # Clean (undeformed) forward pass -- L_cls/L_seg, source only. Backprop immediately
            # so this graph's activations are freed before DefRec's much larger per-chunk
            # rebuild+forward+Chamfer-distance tensors are built (same reasoning as A3/C3).
            src_logits = model(src_batch)
            l_cls = cls_criterion(src_logits["cls"], src_species)
            l_seg = compute_seg_loss(model, src_logits["seg_feat"], src_species,
                                      src_seg, seg_class_weights)
            (kendall.weighted_term("cls", l_cls)
             + kendall.weighted_term("seg", l_seg)).backward()
            src_pts_bcn = batch_points_bcn(src_batch)
            trgt_pts_bcn = batch_points_bcn(trgt_batch)
            del src_logits

            # DefRec on BOTH domains (per CLAUDE.md's DA-S definition), chunked + backprop'd
            # incrementally, each chunk rebuilding a fresh KPConvBatch from its deformed points
            # (see defrec_loss_for_chunk / build_batch_from_points docstrings for why).
            l_defrec = defrec_chunked_backward(args, model, lookup, kendall, config,
                                                defrec_batch_builder, src_pts_bcn, trgt_pts_bcn,
                                                device)

            opt.step()

            bs = src_species.size(0)
            train_cls_loss += l_cls.item() * bs
            train_seg_loss += l_seg.item() * bs
            train_defrec_loss += l_defrec * bs
            n_seen += bs
            if args.verbose_batches:
                t_step_done = time.time()
                io.cprint(f"  epoch {epoch} batch {batch_idx}: collate wait "
                          f"{t_collate_done - t_batch_start:.1f}s, fwd+bwd+step "
                          f"{t_step_done - t_collate_done:.1f}s")
                t_batch_start = t_step_done
        scheduler.step()

        s_cls = kendall.log_vars["cls"].item()
        s_seg = kendall.log_vars["seg"].item()
        s_defrec = kendall.log_vars["defrec"].item()
        io.cprint(f"Trn - Source+Target {epoch}, cls loss: {train_cls_loss / n_seen:.4f}, "
                  f"seg loss: {train_seg_loss / n_seen:.4f}, "
                  f"defrec loss (src+trgt): {train_defrec_loss / n_seen:.4f}, "
                  f"kendall s_cls: {s_cls:.4f}, s_seg: {s_seg:.4f}, s_defrec: {s_defrec:.4f}")

        val_cls_acc, val_cls_avg_acc, val_cls_loss, val_seg_summary, val_seg_loss = \
            evaluate_cls_seg(model, src_val_loader, cls_criterion, seg_class_weights, device)
        val_defrec_loss = evaluate_defrec(model, src_val_loader, args, lookup, config,
                                           defrec_batch_builder, device)
        with torch.no_grad():
            val_total_loss = kendall({
                "cls": torch.tensor(val_cls_loss, device=device),
                "seg": torch.tensor(val_seg_loss, device=device),
                "defrec": torch.tensor(val_defrec_loss, device=device),
            }).item()
        io.cprint(f"Val - Source {epoch}, cls acc: {val_cls_acc:.4f}, "
                  f"cls avg acc: {val_cls_avg_acc:.4f}, cls loss: {val_cls_loss:.4f}, "
                  f"seg loss: {val_seg_loss:.4f}, defrec loss: {val_defrec_loss:.4f}, "
                  f"total loss: {val_total_loss:.4f}")
        for species, m in val_seg_summary.items():
            io.cprint(f"Val - Source {epoch}, {species} seg: "
                      f"loss={m['loss']:.4f}, mIoU={m['mIoU']:.4f}, acc={m['acc']:.4f}")

        trgt_acc, trgt_avg_acc, trgt_loss, _, _, _ = evaluate_cls_only(
            model, trgt_eval_loader, target_eval_criterion, device)
        io.cprint(f"Eval(no-train) - Target {epoch}, acc: {trgt_acc:.4f}, "
                  f"avg acc: {trgt_avg_acc:.4f}, cls loss: {trgt_loss:.4f}  "
                  f"<- domain gap indicator (DA-S, cls-only, held-out -- never trained on)")

        if val_total_loss < best_val_total_loss:
            best_val_total_loss = val_total_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            best_kendall = copy.deepcopy(kendall)
            torch.save(model.state_dict(), os.path.join(exp_path, "model.pt"))

    io.cprint(f"Best model at epoch {best_epoch}, source val total loss {best_val_total_loss:.4f}")

    test_cls_acc, test_cls_avg_acc, test_cls_loss, test_seg_summary, test_seg_loss = \
        evaluate_cls_seg(best_model, src_val_loader, cls_criterion, seg_class_weights, device)
    io.cprint(f"FINAL best-model source val: cls acc: {test_cls_acc:.4f}, "
              f"cls avg acc: {test_cls_avg_acc:.4f}, seg loss: {test_seg_loss:.4f}")
    for species, m in test_seg_summary.items():
        io.cprint(f"FINAL best-model source val, {species} seg: "
                  f"loss={m['loss']:.4f}, mIoU={m['mIoU']:.4f}, acc={m['acc']:.4f}")

    trgt_test_acc, trgt_test_avg_acc, trgt_test_loss, conf, test_true, test_pred = \
        evaluate_cls_only(best_model, trgt_eval_loader, target_eval_criterion, device)
    io.cprint(f"FINAL target (Pheno4D held-out) test accuracy: {trgt_test_acc:.4f}, "
              f"avg acc: {trgt_test_avg_acc:.4f}, loss: {trgt_test_loss:.4f}  "
              f"<- classification-only; target segmentation not evaluated (see module docstring)")
    io.cprint(f"Confusion matrix (rows=true, cols=pred, order={list(IDX_TO_SPECIES.values())}):\n{conf}")
    report = metrics.classification_report(
        test_true, test_pred, labels=list(IDX_TO_SPECIES.keys()),
        target_names=list(IDX_TO_SPECIES.values()), digits=4, zero_division=0)
    io.cprint(f"Per-class precision/recall/F1 on target test:\n{report}")


if __name__ == "__main__":
    main()
