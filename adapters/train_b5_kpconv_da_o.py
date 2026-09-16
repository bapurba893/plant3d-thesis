"""
train_b5_kpconv_da_o.py

Strategy table row B5: KPConv backbone, DA-O (Oracle upper bound), ALL augmentation,
L_cls + L_seg -- the KPConv-track counterpart of adapters/train_c5_dgcnn_da_o.py (row C5).
Mirrors that script's Oracle protocol (domain roles swapped: trains directly on labeled Pheno4D
data instead of adapting from Crops3D) line-for-line, with KPConv's batch/collate mechanics from
adapters/train_b1_kpconv_da0.py substituted for DGCNN's raw-tensor forward pass -- only the
backbone and the batch format it requires differ.

Per CLAUDE.md: "DA-O ... NEVER a real candidate strategy -- exists only as the upper-bound
reference point. Do not treat DA-O results as 'the best model.'" This script exists purely to
measure how much of B1-B4's target-domain gap is closeable in principle (an upper bound), not to
produce a deployable model. Completes Block B (B1-B5 all done after this row).

Per explicit user instruction, this row also directly tests whether KPConv's wrong-signed
selection-signal correlation (`corr(selection-loss, target acc)` positive in EVERY KPConv
configuration tried so far -- B1 +0.316, B2 +0.238, B3 +0.268, B3b +0.291, B4 +0.470, see
step_notes/B4_KPConv_DA0_LD.md) is specifically a symptom of the domain-adaptation problem
(source/target distribution mismatch corrupting what the selection loss can see), or a more
fundamental property of this backbone+dataset pairing that persists even when the model is
trained and selected directly on labeled target data with no domain gap to cross at all. C5
(DGCNN, DA-O) already answered this for DGCNN: corr flips to strongly negative (-0.823, the most
informative of any Block C row) once real target supervision is available -- this row checks
whether KPConv matches that pattern or is the first row where even Oracle training doesn't fix
the selection-signal problem.

Domain roles are swapped relative to every other Block B row (see train_c5_dgcnn_da_o.py's
docstring for the full reasoning, identical here):
  - "labeled train/val" is Pheno4D's OWN annotated subset, split at the PLANT level into
    data/pheno4d_oracle_train.csv (8 plants, 72 scans) and data/pheno4d_oracle_val.csv (2
    plants, 18 scans) -- the SAME split C5 already carved out and validated, reused unchanged
    (plant-disjoint from data/pheno4d_heldout_eval.csv's test plants).
  - "target test" is the SAME data/pheno4d_heldout_eval.csv every other Block B row (B1-B4)
    reports its target cls acc against, so B5's number is directly comparable to theirs. Also
    evaluates target SEGMENTATION on that file's annotated scans -- an Oracle-only metric, not
    comparable to B1-B4's source-val seg mIoU (different domain/label space).

Model is trained FROM SCRATCH on Pheno4D, same reading as C5 ("Oracle" conventionally means
training as if the target were the only labeled dataset available).

Hyperparameters: batch_size=8 (same data-driven necessity as C5 -- 72 training samples,
batch_size=16+drop_last=True would drop 8/72 samples and yield only 4 batches/epoch; batch_size=8
uses all 72 across 9 batches), everything else matches B1.

`neighborhood_limits` calibrated on the Oracle train set ONLY -- unlike B2/B3/B3b (which push
both Crops3D and Pheno4D batches through the encoder and need both-domain calibration), this row
only ever sees Pheno4D data, so single-domain calibration (same pattern as B1's source-only
calibration) is sufficient and correct here, not an oversight.
"""

import argparse
import copy
import datetime
import os
import sys
import time
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
    calibrate_neighborhood_limits,
)
from losses import seg_loss, class_weights_from_counts, KendallUncertaintyWeighting  # noqa: E402
from dataset import (  # noqa: E402
    PlantClsSegDataset, PlantSpeciesDataset,
    IDX_TO_SPECIES, SPECIES_TO_IDX,
    PHENO4D_SEG_NUM_CLASSES, pheno4d_collapse_organ_labels,
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
    p = argparse.ArgumentParser(description="Row B5: KPConv, DA-O (Oracle), ALL augmentation, L_cls+L_seg")
    p.add_argument("--exp_name", type=str, default="B5_kpconv_da_o")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
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
        num_classes = PHENO4D_SEG_NUM_CLASSES[species]
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
    """Classification-only eval -- used here for the target (Pheno4D held-out) test set,
    exactly the same file/metric B1-B4 report, so the number is directly comparable across all
    of B1-B5."""
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

    # "Labeled train/val" = Pheno4D's own annotated subset (Oracle's analog of B1's Crops3D
    # train/val), plant-disjoint from the held-out test plants below. Same split C5 already
    # carved out and validated, reused unchanged.
    oracle_train_set = PlantClsSegDataset(
        data_dir / "pheno4d_oracle_train.csv", manifest_csv, augment=True, seed=args.seed,
        label_transform=pheno4d_collapse_organ_labels, num_classes=PHENO4D_SEG_NUM_CLASSES)
    oracle_val_set = PlantClsSegDataset(
        data_dir / "pheno4d_oracle_val.csv", manifest_csv, augment=False,
        label_transform=pheno4d_collapse_organ_labels, num_classes=PHENO4D_SEG_NUM_CLASSES)

    # "Target test" = the SAME file B1-B4 report their target cls acc against, for a directly
    # comparable number.
    trgt_eval_cls_set = PlantSpeciesDataset(
        data_dir / "pheno4d_heldout_eval.csv", manifest_csv, augment=False)
    # Target SEGMENTATION eval (annotated subset, auto-filtered by PlantClsSegDataset) --
    # Oracle-only metric, no B1-B4 counterpart.
    trgt_eval_seg_set = PlantClsSegDataset(
        data_dir / "pheno4d_heldout_eval.csv", manifest_csv, augment=False,
        label_transform=pheno4d_collapse_organ_labels, num_classes=PHENO4D_SEG_NUM_CLASSES)

    io.cprint(f"Pheno4D oracle train: {len(oracle_train_set)} {oracle_train_set.class_counts()}")
    io.cprint(f"Pheno4D oracle val:   {len(oracle_val_set)} {oracle_val_set.class_counts()}")
    io.cprint(f"Pheno4D held-out eval (target test, never trained on, cls): "
              f"{len(trgt_eval_cls_set)} {trgt_eval_cls_set.class_counts()}")
    io.cprint(f"Pheno4D held-out eval (target test, annotated subset, seg): "
              f"{len(trgt_eval_seg_set)} {trgt_eval_seg_set.class_counts()}  "
              f"<- Oracle-only metric, NOT comparable to B1-B4's source-val seg mIoU "
              f"(different domain/label space, see PHENO4D_SEG_NUM_CLASSES docstring)")

    counts = oracle_train_set.class_counts()
    cls_counts_t = torch.tensor([counts.get(sp, 0) for sp in IDX_TO_SPECIES.values()],
                                 dtype=torch.float32)
    cls_class_weights = class_weights_from_counts(cls_counts_t).to(device)
    io.cprint(f"L_cls class weights (w_c=(f_c+eps)^-1, normalized): "
              f"{dict(zip(IDX_TO_SPECIES.values(), cls_class_weights.tolist()))}")

    seg_class_weights = {}
    for species in PHENO4D_SEG_NUM_CLASSES:
        hist = oracle_train_set.seg_class_histogram(species)
        w = class_weights_from_counts(torch.tensor(hist, dtype=torch.float32)).to(device)
        seg_class_weights[species] = w
        io.cprint(f"L_seg class weights for {species} (counts={hist.tolist()}): {w.tolist()}")

    config = PlantKPConvConfig()
    config.dropout = args.dropout

    # Calibrate on the Oracle train set only -- this row never touches Crops3D at all, so
    # single-domain calibration (same pattern as B1's source-only calibration) is correct here,
    # not an oversight (unlike B2/B3/B3b, which need both-domain calibration because they push
    # both Crops3D and Pheno4D batches through the encoder).
    config.neighborhood_limits = calibrate_neighborhood_limits(
        config, oracle_train_set, args.batch_size, num_workers=args.num_workers)
    io.cprint(f"Calibrated neighborhood_limits (per layer, 90th-percentile cap): "
              f"{config.neighborhood_limits}")

    oracle_train_loader = DataLoader(oracle_train_set, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, drop_last=True,
                                      collate_fn=make_kpconv_collate_fn(config))
    oracle_val_loader = DataLoader(oracle_val_set, batch_size=args.test_batch_size, shuffle=False,
                                    num_workers=args.num_workers,
                                    collate_fn=make_kpconv_collate_fn(config))
    trgt_eval_cls_loader = DataLoader(trgt_eval_cls_set, batch_size=args.test_batch_size, shuffle=False,
                                       num_workers=args.num_workers,
                                       collate_fn=make_kpconv_collate_fn_cls_only(config))
    trgt_eval_seg_loader = DataLoader(trgt_eval_seg_set, batch_size=args.test_batch_size, shuffle=False,
                                       num_workers=args.num_workers,
                                       collate_fn=make_kpconv_collate_fn(config))

    model = KPConv_ClsSeg(config, num_class=2, seg_num_classes=PHENO4D_SEG_NUM_CLASSES).to(device)
    kendall = KendallUncertaintyWeighting(
        {"cls": "classification", "seg": "classification"}).to(device)
    best_model = copy.deepcopy(model)

    opt = optim.Adam(list(model.parameters()) + list(kendall.parameters()),
                      lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(opt, args.epochs)
    cls_criterion = nn.CrossEntropyLoss(weight=cls_class_weights)
    target_eval_criterion = nn.CrossEntropyLoss()

    best_val_total_loss, best_epoch = float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        train_cls_loss, train_seg_loss, n_seen = 0.0, 0.0, 0
        t_batch_start = time.time()
        for batch_idx, batch in enumerate(oracle_train_loader):
            t_collate_done = time.time()
            batch = batch.to(device)
            species_labels = batch.species_labels
            seg_labels = reshape_seg_labels(batch)

            opt.zero_grad()
            logits = model(batch)
            l_cls = cls_criterion(logits["cls"], species_labels)
            l_seg = compute_seg_loss(model, logits["seg_feat"], species_labels,
                                      seg_labels, seg_class_weights)
            total = kendall({"cls": l_cls, "seg": l_seg})
            total.backward()
            opt.step()

            bs = species_labels.size(0)
            train_cls_loss += l_cls.item() * bs
            train_seg_loss += l_seg.item() * bs
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
        io.cprint(f"Trn - Oracle {epoch}, cls loss: {train_cls_loss / n_seen:.4f}, "
                  f"seg loss: {train_seg_loss / n_seen:.4f}, "
                  f"kendall s_cls: {s_cls:.4f}, s_seg: {s_seg:.4f}")

        val_cls_acc, val_cls_avg_acc, val_cls_loss, val_seg_summary, val_seg_loss = \
            evaluate_cls_seg(model, oracle_val_loader, cls_criterion, seg_class_weights, device)
        with torch.no_grad():
            val_total_loss = kendall({
                "cls": torch.tensor(val_cls_loss, device=device),
                "seg": torch.tensor(val_seg_loss, device=device),
            }).item()
        io.cprint(f"Val - Oracle {epoch}, cls acc: {val_cls_acc:.4f}, "
                  f"cls avg acc: {val_cls_avg_acc:.4f}, cls loss: {val_cls_loss:.4f}, "
                  f"seg loss: {val_seg_loss:.4f}, total loss: {val_total_loss:.4f}")
        for species, m in val_seg_summary.items():
            io.cprint(f"Val - Oracle {epoch}, {species} seg: "
                      f"loss={m['loss']:.4f}, mIoU={m['mIoU']:.4f}, acc={m['acc']:.4f}")

        trgt_acc, trgt_avg_acc, trgt_loss, _, _, _ = evaluate_cls_only(
            model, trgt_eval_cls_loader, target_eval_criterion, device)
        io.cprint(f"Eval(no-train) - Target 'test' {epoch}, acc: {trgt_acc:.4f}, "
                  f"avg acc: {trgt_avg_acc:.4f}, cls loss: {trgt_loss:.4f}  "
                  f"<- comparable to B1-B4's target cls acc; DA-O trains ON Pheno4D "
                  f"directly (upper bound, never a candidate strategy per CLAUDE.md)")

        if val_total_loss < best_val_total_loss:
            best_val_total_loss = val_total_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), os.path.join(exp_path, "model.pt"))

    io.cprint(f"Best model at epoch {best_epoch}, oracle val total loss {best_val_total_loss:.4f}")

    test_cls_acc, test_cls_avg_acc, test_cls_loss, test_seg_summary, test_seg_loss = \
        evaluate_cls_seg(best_model, oracle_val_loader, cls_criterion, seg_class_weights, device)
    io.cprint(f"FINAL best-model oracle val: cls acc: {test_cls_acc:.4f}, "
              f"cls avg acc: {test_cls_avg_acc:.4f}, seg loss: {test_seg_loss:.4f}")
    for species, m in test_seg_summary.items():
        io.cprint(f"FINAL best-model oracle val, {species} seg: "
                  f"loss={m['loss']:.4f}, mIoU={m['mIoU']:.4f}, acc={m['acc']:.4f}")

    trgt_test_acc, trgt_test_avg_acc, trgt_test_loss, conf, test_true, test_pred = \
        evaluate_cls_only(best_model, trgt_eval_cls_loader, target_eval_criterion, device)
    io.cprint(f"FINAL target (Pheno4D held-out) test accuracy: {trgt_test_acc:.4f}, "
              f"avg acc: {trgt_test_avg_acc:.4f}, loss: {trgt_test_loss:.4f}  "
              f"<- directly comparable to B1-B4's target cls acc (same file/metric)")
    io.cprint(f"Confusion matrix (rows=true, cols=pred, order={list(IDX_TO_SPECIES.values())}):\n{conf}")
    report = metrics.classification_report(
        test_true, test_pred, labels=list(IDX_TO_SPECIES.keys()),
        target_names=list(IDX_TO_SPECIES.values()), digits=4, zero_division=0)
    io.cprint(f"Per-class precision/recall/F1 on target test:\n{report}")

    _, _, _, trgt_test_seg_summary, _ = evaluate_cls_seg(
        best_model, trgt_eval_seg_loader, cls_criterion, seg_class_weights, device)
    io.cprint("FINAL target (Pheno4D held-out, annotated subset) segmentation -- "
              "Oracle-only metric, NOT comparable to B1-B4's source-val seg mIoU "
              "(different domain/label space):")
    for species, m in trgt_test_seg_summary.items():
        io.cprint(f"FINAL target seg, {species}: "
                  f"loss={m['loss']:.4f}, mIoU={m['mIoU']:.4f}, acc={m['acc']:.4f}")


if __name__ == "__main__":
    main()
