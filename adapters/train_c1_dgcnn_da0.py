"""
train_c1_dgcnn_da0.py

Strategy table row C1: DGCNN backbone, DA-0 (source-only, no adaptation),
ALL augmentation. Classification-only interim (species Tomato-vs-Maize) --
L_seg is deferred until Crops3D_IS point-wise labels are available.

Trains on Crops3D (source) only -- no target data or target loss of any
kind, since DA-0 is the "no adaptation" lower bound. Pheno4D (target) is
used purely for held-out evaluation, to measure the sensor-domain gap;
its labels are never used for training, matching the DA-0 protocol.

Reuses DGCNN from the cloned DefRec_and_PCM reference repo
(PointDA/Models.py) rather than reimplementing the backbone.
"""

import argparse
import copy
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFREC_ROOT = os.environ.get("DEFREC_ROOT", str(_REPO_ROOT.parent / "DefRec_and_PCM"))
if _DEFREC_ROOT not in sys.path:
    sys.path.insert(0, _DEFREC_ROOT)

from PointDA.Models import DGCNN  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import PlantSpeciesDataset, IDX_TO_SPECIES, SPECIES_TO_IDX  # noqa: E402

NWORKERS = 2


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
    p = argparse.ArgumentParser(description="Row C1: DGCNN, DA-0, ALL augmentation")
    p.add_argument("--exp_name", type=str, default="C1_dgcnn_da0")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0, help="-1 for cpu")
    return p.parse_args()


def evaluate(model, loader, criterion, device, split_name):
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
    src_train_set = PlantSpeciesDataset(
        data_dir / "crops3d_train.csv", data_dir / "preprocessed" / "preprocessed_manifest.csv",
        augment=True, seed=args.seed)
    src_val_set = PlantSpeciesDataset(
        data_dir / "crops3d_val.csv", data_dir / "preprocessed" / "preprocessed_manifest.csv",
        augment=False)
    trgt_eval_set = PlantSpeciesDataset(
        data_dir / "pheno4d_heldout_eval.csv", data_dir / "preprocessed" / "preprocessed_manifest.csv",
        augment=False)

    io.cprint(f"Crops3D train: {len(src_train_set)} {src_train_set.class_counts()}")
    io.cprint(f"Crops3D val:   {len(src_val_set)} {src_val_set.class_counts()}")
    io.cprint(f"Pheno4D held-out eval (target, never trained on): "
              f"{len(trgt_eval_set)} {trgt_eval_set.class_counts()}")

    # Class-imbalance-aware weighting, same w_c = (f_c + eps)^-1 pattern the strategy
    # table defines for L_seg's per-class weights -- Crops3D train is 192 Maize vs 71
    # Tomato (2.7:1), and plain CE let the model settle into a majority-class shortcut
    # (frozen at exactly the majority-baseline accuracy) before eventually escaping it;
    # weighting removes the incentive for that shortcut instead of relying on luck.
    counts = src_train_set.class_counts()
    eps = 1e-6
    n_train = len(src_train_set)
    class_weights = torch.zeros(len(SPECIES_TO_IDX))
    for species, idx in SPECIES_TO_IDX.items():
        freq = counts.get(species, 0) / n_train
        class_weights[idx] = 1.0 / (freq + eps)
    class_weights = class_weights / class_weights.sum() * len(SPECIES_TO_IDX)  # normalize, keeps LR/wd scale meaningful
    io.cprint(f"Class weights (w_c = (f_c+eps)^-1, normalized): "
              f"{dict(zip(SPECIES_TO_IDX.keys(), class_weights.tolist()))}")
    class_weights = class_weights.to(device)

    src_train_loader = DataLoader(src_train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=NWORKERS, drop_last=True)
    src_val_loader = DataLoader(src_val_set, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=NWORKERS)
    trgt_eval_loader = DataLoader(trgt_eval_set, batch_size=args.test_batch_size, shuffle=False,
                                   num_workers=NWORKERS)

    model_args = argparse.Namespace(model="dgcnn", cuda=cuda, dropout=args.dropout)
    model = DGCNN(model_args, num_class=2).to(device)
    best_model = copy.deepcopy(model)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(opt, args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc, best_epoch = 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        train_loss, n_seen = 0.0, 0
        for pts, labels in src_train_loader:
            pts, labels = pts.to(device), labels.to(device)
            pts = pts.permute(0, 2, 1)
            opt.zero_grad()
            logits = model(pts, activate_DefRec=False)
            loss = criterion(logits["cls"], labels)
            loss.backward()
            opt.step()
            train_loss += loss.item() * pts.size(0)
            n_seen += pts.size(0)
        scheduler.step()
        io.cprint(f"Trn - Source {epoch}, cls loss: {train_loss / n_seen:.4f}")

        val_acc, val_avg_acc, val_loss, _, _, _ = evaluate(model, src_val_loader, criterion, device, "Source Val")
        io.cprint(f"Val - Source {epoch}, acc: {val_acc:.4f}, avg acc: {val_avg_acc:.4f}, cls loss: {val_loss:.4f}")

        trgt_acc, trgt_avg_acc, trgt_loss, _, _, _ = evaluate(model, trgt_eval_loader, criterion, device, "Target Eval")
        io.cprint(f"Eval(no-train) - Target {epoch}, acc: {trgt_acc:.4f}, avg acc: {trgt_avg_acc:.4f}, "
                  f"cls loss: {trgt_loss:.4f}  <- domain gap indicator (DA-0 baseline)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), os.path.join(exp_path, "model.pt"))

    io.cprint(f"Best model at epoch {best_epoch}, source val acc {best_val_acc:.4f}")

    test_acc, test_avg_acc, test_loss, conf, test_true, test_pred = evaluate(
        best_model, trgt_eval_loader, criterion, device, "Target Test")
    io.cprint(f"FINAL target (Pheno4D held-out) test accuracy: {test_acc:.4f}, avg acc: {test_avg_acc:.4f}, "
              f"loss: {test_loss:.4f}")
    io.cprint(f"Confusion matrix (rows=true, cols=pred, order={list(IDX_TO_SPECIES.values())}):\n{conf}")
    report = metrics.classification_report(
        test_true, test_pred, labels=list(IDX_TO_SPECIES.keys()),
        target_names=list(IDX_TO_SPECIES.values()), digits=4, zero_division=0)
    io.cprint(f"Per-class precision/recall/F1 on target test:\n{report}")


if __name__ == "__main__":
    main()
