"""
train_b3_kpconv_da_d.py

Strategy table row B3: KPConv backbone, DA-D (discrepancy-based domain adaptation, Deep CORAL),
ALL augmentation, L_cls + L_seg + L_CORAL. This is the row CLAUDE.md's strategy table actually
specifies as B3 (see that file's Backbones/strategy-table sections for the B3-vs-B3b naming
mix-up this project already caught and corrected once) -- not a substitute for B3b (DA-S,
already done, see step_notes/B3b_KPConv_DA_S.md), which stays a separate addition.

Purpose of this row, per explicit user instruction: B3b showed the strongest non-Oracle result
in the whole project (KPConv + DA-S, full-run mean target acc 0.629 -> 0.860, +23.1 points, the
most stable trajectory of any row trained so far). This row asks whether B3b's win is specific
to the self-supervised reconstruction mechanism, or whether KPConv simply responds well to ANY
domain-adaptation pressure -- DA-D is a natural second data point: a genuinely different
mechanism (feature-covariance alignment, no discriminator, no reconstruction pretext) from both
DA-A (B2, roughly neutral on KPConv) and DA-S (B3b, strongly positive).

DA-D has no reference implementation to adapt: DefRec_and_PCM implements DefRec+PCM only (no
CORAL/MMD, confirmed by grep -- see adapters/coral.py's docstring), and unlike DA-A (already
built from scratch for C2, ported unchanged here) this is the first DA-D row in this project on
any backbone, so `adapters/coral.py` is new, from-scratch, standard Deep CORAL (Sun & Saenko,
2016), not adapted from any existing repo code.

Much simpler plumbing than B2 (DA-A) or B3b (DA-S): no domain discriminator, no Gradient
Reversal Layer, no lambda_p/lambda_ent schedule, no entropy minimization, and -- unlike B3b's
DefRec -- no deformation, so no per-chunk batch rebuilds are needed at all. Structurally this is
much closer to B2's clean-batch shape (one source forward for L_cls/L_seg, one target forward
for its pooled feature) than to B3b's, so per-epoch cost is expected to be similar to B1/B2's
(a few hours), not B3b's ~5h.

`neighborhood_limits` calibrated over BOTH domains, same as B2/B3b and for the same reason
(target points flow through the encoder here too, to get their pooled feature for CORAL).

Model selection: same protocol as every prior row -- best epoch by lowest source val total loss
(Kendall(cls, seg, coral), coral evaluated using source val vs. a held-out slice of the target
adaptation pool -- see `evaluate_coral` below). L_CORAL is Kendall-weighted (a cooperative DA
loss per CLAUDE.md's loss architecture), unlike DA-A's L_dom/L_ent which are fixed/scheduled and
excluded from Kendall entirely -- a real, substantive difference in how these two DA methods'
losses are combined, not just in what machinery each one needs.
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
)
from coral import coral_loss  # noqa: E402
from losses import seg_loss, class_weights_from_counts, KendallUncertaintyWeighting  # noqa: E402
from dataset import (  # noqa: E402
    PlantClsSegDataset, PlantSpeciesDataset,
    IDX_TO_SPECIES, SPECIES_TO_IDX, SEG_NUM_CLASSES,
)

LAMBDA_LOV = 1.0  # fixed per the strategy table, never learned/swept


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
        description="Row B3: KPConv, DA-D (Deep CORAL), ALL augmentation, L_cls+L_seg+L_CORAL")
    p.add_argument("--exp_name", type=str, default="B3_kpconv_da_d")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0, help="-1 for cpu")
    p.add_argument("--verbose_batches", action="store_true",
                    help="log per-batch collate/step timing, standing KPConv-row convention "
                         "since B1's silent-stall incident")
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


def evaluate_coral(model, src_loader, trgt_loader, device):
    """Source-val vs. target-adapt-pool CORAL loss, batch-paired and averaged -- same role as
    A3/C3/B3b's evaluate_defrec: a held-out (never-backprop'd) reading of the DA loss term, fed
    into the same Kendall total used for model selection. Both loaders run in eval mode (no
    augmentation on src_loader since it's already `augment=False`; trgt_loader here reuses the
    training-time adaptation-pool loader deliberately, since CORAL doesn't distinguish a
    separate held-out target split the way defrec's chunking didn't either -- consistent with
    every DA method in this project training-and-evaluating L_DA off the same target adaptation
    pool, never off `pheno4d_heldout_eval.csv`, which stays reserved for the never-trained-on
    accuracy diagnostic)."""
    model.eval()
    loss_sum, n_batches = 0.0, 0
    with torch.no_grad():
        trgt_iter = cycle(trgt_loader)
        for src_batch, trgt_batch in zip(src_loader, trgt_iter):
            src_batch = src_batch.to(device)
            trgt_batch = trgt_batch.to(device)
            src_feat = model(src_batch)["feat"]
            trgt_feat = model(trgt_batch)["feat"]
            loss_sum += coral_loss(src_feat, trgt_feat).item()
            n_batches += 1
    return loss_sum / n_batches


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
    io.cprint(f"Pheno4D adaptation pool (target, unlabeled for training -- "
              f"species labels below shown for bookkeeping only, never used in the loss): "
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

    # Calibrate over BOTH domains, same as B2/B3b and for the same reason (target points flow
    # through the encoder here too, to get their pooled feature for CORAL).
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

    model = KPConv_ClsSeg(config, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    kendall = KendallUncertaintyWeighting(
        {"cls": "classification", "seg": "classification", "coral": "regression"}).to(device)
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
        train_cls_loss, train_seg_loss, train_coral_loss, n_seen = 0.0, 0.0, 0.0, 0
        t_batch_start = time.time()

        trgt_iter = cycle(trgt_adapt_loader)
        for batch_idx, (src_batch, trgt_batch) in enumerate(zip(src_train_loader, trgt_iter)):
            t_collate_done = time.time()
            src_batch = src_batch.to(device)
            src_species = src_batch.species_labels
            src_seg = reshape_seg_labels(src_batch)
            trgt_batch = trgt_batch.to(device)
            # target species labels intentionally never read below (DA-D never uses target
            # labels, same discipline as DA-A/DA-S)

            opt.zero_grad()

            src_logits = model(src_batch)
            l_cls = cls_criterion(src_logits["cls"], src_species)
            l_seg = compute_seg_loss(model, src_logits["seg_feat"], src_species,
                                      src_seg, seg_class_weights)

            trgt_logits = model(trgt_batch)
            l_coral = coral_loss(src_logits["feat"], trgt_logits["feat"])

            total = kendall({"cls": l_cls, "seg": l_seg, "coral": l_coral})
            total.backward()
            opt.step()

            bs = src_species.size(0)
            train_cls_loss += l_cls.item() * bs
            train_seg_loss += l_seg.item() * bs
            train_coral_loss += l_coral.item() * bs
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
        s_coral = kendall.log_vars["coral"].item()
        io.cprint(f"Trn - Source+Target {epoch}, cls loss: {train_cls_loss / n_seen:.4f}, "
                  f"seg loss: {train_seg_loss / n_seen:.4f}, "
                  f"coral loss: {train_coral_loss / n_seen:.4f}, "
                  f"kendall s_cls: {s_cls:.4f}, s_seg: {s_seg:.4f}, s_coral: {s_coral:.4f}")

        val_cls_acc, val_cls_avg_acc, val_cls_loss, val_seg_summary, val_seg_loss = \
            evaluate_cls_seg(model, src_val_loader, cls_criterion, seg_class_weights, device)
        val_coral_loss = evaluate_coral(model, src_val_loader, trgt_adapt_loader, device)
        with torch.no_grad():
            val_total_loss = kendall({
                "cls": torch.tensor(val_cls_loss, device=device),
                "seg": torch.tensor(val_seg_loss, device=device),
                "coral": torch.tensor(val_coral_loss, device=device),
            }).item()
        io.cprint(f"Val - Source {epoch}, cls acc: {val_cls_acc:.4f}, "
                  f"cls avg acc: {val_cls_avg_acc:.4f}, cls loss: {val_cls_loss:.4f}, "
                  f"seg loss: {val_seg_loss:.4f}, coral loss: {val_coral_loss:.4f}, "
                  f"total loss: {val_total_loss:.4f}")
        for species, m in val_seg_summary.items():
            io.cprint(f"Val - Source {epoch}, {species} seg: "
                      f"loss={m['loss']:.4f}, mIoU={m['mIoU']:.4f}, acc={m['acc']:.4f}")

        trgt_acc, trgt_avg_acc, trgt_loss, _, _, _ = evaluate_cls_only(
            model, trgt_eval_loader, target_eval_criterion, device)
        io.cprint(f"Eval(no-train) - Target {epoch}, acc: {trgt_acc:.4f}, "
                  f"avg acc: {trgt_avg_acc:.4f}, cls loss: {trgt_loss:.4f}  "
                  f"<- domain gap indicator (DA-D, cls-only, held-out -- never trained on)")

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
