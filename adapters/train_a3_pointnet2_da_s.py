"""
train_a3_pointnet2_da_s.py

Strategy table row A3: PointNet++ backbone, DA-S (self-supervised
deformation reconstruction, DefRec-style), ALL augmentation, L_cls +
L_seg + L_defrec -- the PointNet++-track counterpart of
adapters/train_c3_dgcnn_da_s.py (row C3, DGCNN). Deliberately mirrors that
script almost line-for-line: same DA-S protocol, same self-supervised
machinery (`DefRec_and_PCM.DefRec.deform_input`/`.calc_loss`, imported
unmodified, exactly as C3 uses them), same chunked-Chamfer-distance OOM
workaround (see DEFREC_CHUNK below, carried over unchanged from C3 --
`PointNet2_ClsSeg` builds the same (B, 4096, ...) per-point tensors DGCNN
does, so the same memory pressure applies) -- only the backbone
(adapters/models_pointnet2.py::PointNet2_ClsSeg in place of
adapters/models.py::DGCNN_ClsSeg) differs.

Unlike A2 (which needed a from-scratch adversarial-machinery port),
PointNet2's DefRec support required composing ONE existing reference-repo
building block that A1/A2 hadn't needed yet: `PointDA.Models.RegionReconstruction`
-- a generic per-point Conv1d stack over an arbitrary input feature width,
the exact same class DGCNN_ClsSeg inherits via its DGCNN base class. It
was added to `PointNet2_ClsSeg.__init__`/`.forward` (fed the SAME
`seg_feat` tensor already computed for segmentation, since that tensor is
already "per-point features concatenated with the pooled global feature"
-- exactly DefRec's expected input shape). See models_pointnet2.py's
docstring for the reasoning; no reference-repo code was modified.

Purpose of this row, per explicit user instruction: C3 was DGCNN's
LEAST-BAD domain-adaptation result (full-run mean target acc 0.703 vs.
C1's 0.791 -- an 8.8-point gap, versus DA-A's much larger ~15.8-point gap
in C2) -- this tests whether that "DA-S is the closest of the three
methods tried" pattern holds on a different backbone too, which matters
more right now than starting a third backbone on a still-incomplete
picture. Compares against A1 the same way C3 compares against C1: both
use the identical `pheno4d_heldout_eval.csv` target test set/protocol.

Full-trajectory analysis from the start, same as A2/C2/C3/C4 -- see
step_notes/A3_PointNet2_DA_S.md.

Model selection: same protocol as A1/A2/C3 -- best epoch by lowest source
val total loss (Kendall(cls, seg, defrec), defrec evaluated on source val
data only). See train_c3_dgcnn_da_s.py's docstring for the full reasoning
(unchanged here).
"""

import argparse
import copy
import datetime
import os
import sys
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
from models_pointnet2 import PointNet2_ClsSeg  # noqa: E402
from losses import seg_loss, class_weights_from_counts, KendallUncertaintyWeighting  # noqa: E402
from dataset import (  # noqa: E402
    PlantClsSegDataset, PlantSpeciesDataset,
    IDX_TO_SPECIES, SPECIES_TO_IDX, SEG_NUM_CLASSES,
)
from DefRec_and_PCM import DefRec  # noqa: E402
from utils import pc_utils  # noqa: E402

NWORKERS = 2
LAMBDA_LOV = 1.0  # fixed per the strategy table, never learned/swept

# Same chunking workaround as C3 (see that file's docstring for the
# CUDA-OOM root cause) -- PointNet2's per-point tensors are the same
# (B, N, ...) shape/density as DGCNN's, so the same Chamfer-distance
# memory pressure applies.
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
        description="Row A3: PointNet++, DA-S (self-supervised DefRec), ALL augmentation, "
                    "L_cls+L_seg+L_defrec")
    p.add_argument("--exp_name", type=str, default="A3_pointnet2_da_s")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=32)
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
    return p.parse_args()


def compute_seg_loss(model, seg_feat, species_labels, seg_labels, seg_class_weights):
    """Routes each sample to its species' segmentation head, returns a
    single sample-count-weighted L_seg scalar for the batch."""
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
    """Accumulates per-species loss/mIoU/pixel-accuracy into `stats`
    (dict[species] -> {"loss_sum", "mIoU_sum", "acc_sum", "n"})."""
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


def defrec_loss_for_batch(args, model, lookup, pts_orig, device):
    """Deforms pts_orig (B, 3, N), reconstructs via the model's DefRec
    head, returns the scalar Chamfer reconstruction loss. Same as C3's
    helper of the same name."""
    pts_deformed, mask = DefRec.deform_input(
        pts_orig.clone(), lookup, args.DefRec_dist, device)
    logits_deform = model(pts_deformed, activate_DefRec=True)
    return DefRec.calc_loss(args, logits_deform, pts_orig, mask)


def _iter_chunks(pts, chunk_size):
    n = pts.size(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield pts[start:end], end - start


def defrec_chunked_backward(args, model, lookup, kendall, src_pts, trgt_pts, device,
                             chunk_size=DEFREC_CHUNK):
    """Computes L_defrec (DefRec on both domains, sample-count-weighted
    into one scalar per CLAUDE.md's DA-S definition) and immediately
    backprops it, one small chunk at a time, to bound peak CUDA memory.
    Identical logic and mathematical-equivalence argument as C3's helper
    of the same name -- see train_c3_dgcnn_da_s.py's docstring."""
    bs_src, bs_trgt = src_pts.size(0), trgt_pts.size(0)
    total_n = bs_src + bs_trgt
    chunks = list(_iter_chunks(src_pts, chunk_size)) + list(_iter_chunks(trgt_pts, chunk_size))
    num_chunks = len(chunks)
    s_defrec = kendall.log_vars["defrec"]

    loss_sum, n_sum = 0.0, 0
    for chunk_pts, chunk_n in chunks:
        chunk_loss = defrec_loss_for_batch(args, model, lookup, chunk_pts, device)
        weighted_chunk = (torch.exp(-s_defrec) / 2.0 * chunk_loss * (chunk_n / total_n)
                           + s_defrec / (2.0 * num_chunks))
        weighted_chunk.backward()
        loss_sum += chunk_loss.item() * chunk_n
        n_sum += chunk_n
    return loss_sum / n_sum


def evaluate_cls_seg(model, loader, cls_criterion, seg_class_weights, device):
    model.eval()
    n_seen, cls_loss_sum = 0, 0.0
    all_true_cls, all_pred_cls = [], []
    seg_stats = {sp: {"loss_sum": 0.0, "mIoU_sum": 0.0, "acc_sum": 0.0, "n": 0}
                 for sp in SPECIES_TO_IDX}
    with torch.no_grad():
        for pts, species_labels, seg_labels in loader:
            pts = pts.permute(0, 2, 1).to(device)
            species_labels = species_labels.to(device)
            seg_labels = seg_labels.to(device)

            logits = model(pts, activate_DefRec=False)
            l_cls = cls_criterion(logits["cls"], species_labels)
            cls_loss_sum += l_cls.item() * pts.size(0)
            preds_cls = logits["cls"].max(dim=1)[1]
            all_true_cls.append(species_labels.cpu().numpy())
            all_pred_cls.append(preds_cls.cpu().numpy())
            n_seen += pts.size(0)

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
    """Classification-only eval for Pheno4D (target) -- domain-gap
    indicator, never used for training or model selection."""
    model.eval()
    losses, all_true, all_pred = [], [], []
    with torch.no_grad():
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            pts = pts.permute(0, 2, 1)
            logits = model(pts, activate_DefRec=False)
            loss = criterion(logits["cls"], labels)
            losses.append(loss.item() * pts.size(0))
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


def evaluate_defrec(model, loader, args, lookup, device):
    """Source-val-only DefRec reconstruction loss, sample-count-weighted
    mean over the loader. Same as C3's helper of the same name."""
    model.eval()
    loss_sum, n_seen = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pts = batch[0]
            pts = pts.permute(0, 2, 1).to(device)
            for chunk_pts, chunk_n in _iter_chunks(pts, DEFREC_CHUNK):
                l = defrec_loss_for_batch(args, model, lookup, chunk_pts, device)
                loss_sum += l.item() * chunk_n
                n_seen += chunk_n
    return loss_sum / n_seen


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

    src_train_loader = DataLoader(src_train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=NWORKERS, drop_last=True)
    src_val_loader = DataLoader(src_val_set, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=NWORKERS)
    trgt_adapt_loader = DataLoader(trgt_adapt_set, batch_size=args.batch_size, shuffle=True,
                                    num_workers=NWORKERS, drop_last=True)
    trgt_eval_loader = DataLoader(trgt_eval_set, batch_size=args.test_batch_size, shuffle=False,
                                   num_workers=NWORKERS)

    n_batches_per_epoch = len(src_train_loader)
    io.cprint(f"Batches/epoch: source={n_batches_per_epoch} (drives epoch length), "
              f"target_adapt={len(trgt_adapt_loader)} (cycled)")

    lookup = torch.Tensor(pc_utils.region_mean(args.num_regions)).to(device)

    model_args = argparse.Namespace(dropout=args.dropout)
    model = PointNet2_ClsSeg(model_args, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
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

        trgt_iter = cycle(trgt_adapt_loader)
        for src_batch, trgt_batch in zip(src_train_loader, trgt_iter):
            src_pts, src_species, src_seg = src_batch
            trgt_pts, _trgt_species_unused = trgt_batch  # target labels never read

            src_pts = src_pts.permute(0, 2, 1).to(device)
            src_species = src_species.to(device)
            src_seg = src_seg.to(device)
            trgt_pts = trgt_pts.permute(0, 2, 1).to(device)

            opt.zero_grad()

            # Clean (undeformed) forward pass -- L_cls/L_seg, source only.
            # Backprop immediately so this graph's activations are freed
            # before DefRec's much larger Chamfer-distance tensors are
            # built (same reasoning as C3).
            src_logits = model(src_pts, activate_DefRec=False)
            l_cls = cls_criterion(src_logits["cls"], src_species)
            l_seg = compute_seg_loss(model, src_logits["seg_feat"], src_species,
                                      src_seg, seg_class_weights)
            (kendall.weighted_term("cls", l_cls)
             + kendall.weighted_term("seg", l_seg)).backward()
            del src_logits

            # DefRec on BOTH domains (per CLAUDE.md's DA-S definition),
            # chunked + backprop'd incrementally.
            l_defrec = defrec_chunked_backward(args, model, lookup, kendall,
                                                src_pts, trgt_pts, device)

            opt.step()

            bs = src_pts.size(0)
            train_cls_loss += l_cls.item() * bs
            train_seg_loss += l_seg.item() * bs
            train_defrec_loss += l_defrec * bs
            n_seen += bs
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
        val_defrec_loss = evaluate_defrec(model, src_val_loader, args, lookup, device)
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
