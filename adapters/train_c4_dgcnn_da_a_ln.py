"""
train_c4_dgcnn_da_a_ln.py

Strategy table row C4: DGCNN backbone, DA-A (adversarial domain
adaptation), L-N augmentation ONLY (Gaussian jitter, no rotation/scale/
flip/crop/dropout/cubic symmetry) -- isolates DGCNN's known noise
weakness (CLAUDE.md: "unrestricted k-NN lets noisy points become spurious
graph edges") by comparing against C2 (same backbone, same DA-A method,
ALL augmentation). Byte-for-byte the same adversarial machinery as C2
(adapters/dann.py, unmodified from C2's corrected version -- see below),
only the augmentation pipeline differs, via PlantClsSegDataset/
PlantSpeciesDataset's `augment_mode="ln_only"` (adapters/dataset.py,
scripts/augmentations.py::compose_pipeline_ln_only). This script is a
near-line-for-line copy of train_c2_dgcnn_da_a.py with exactly that one
variable changed, per CLAUDE.md's row-ordering convention of isolating
one variable at a time.

Uses dann.py's ramped-lambda_ent fix from the start (adapters/dann.py's
"REVISED 2026-09-11" note, adapters/train_c2_dgcnn_da_a.py's same-dated
note): before starting C4, C2's first run (unramped, fixed lambda_ent)
was found to repeatedly collapse target predictions to source's majority
class (Crops3D is 73% Maize, Pheno4D is 62-63% Tomato -- opposite
majorities), and was re-run with lambda_ent ramped alongside the GRL
alpha instead. C4 reuses that corrected dann.py unchanged, never the
original unramped version, so C2 and C4 stay comparable as intended.

Per explicit user instruction: this row's results get the SAME
full-trajectory analysis done for C2 (parsing every epoch's target-acc/
val-loss diagnostic, not just the protocol-selected checkpoint) as soon
as the run finishes, not as an afterthought -- this is also a direct test
of whether C2's instability (even after the entropy-ramp fix) is
fundamental to the adversarial method itself or specific to the ALL
augmentation mix C2 used, since C4 keeps the method identical and swaps
only the augmentation.

Model selection: identical protocol to C1/A1/C2 -- best checkpoint by
lowest source val total loss (Kendall(cls,seg) on Crops3D val only),
never touching target labels or L_dom/L_ent. See train_c2_dgcnn_da_a.py's
docstring for the full reasoning (identical here).
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
from models import DGCNN_ClsSeg  # noqa: E402
from dann import (  # noqa: E402
    DomainDiscriminator, grad_reverse, lambda_p_schedule, lambda_ent_schedule, entropy_loss,
)
from losses import seg_loss, class_weights_from_counts, KendallUncertaintyWeighting  # noqa: E402
from dataset import (  # noqa: E402
    PlantClsSegDataset, PlantSpeciesDataset,
    IDX_TO_SPECIES, SPECIES_TO_IDX, SEG_NUM_CLASSES,
)

NWORKERS = 2
LAMBDA_LOV = 1.0  # fixed per the strategy table, never learned/swept
AUGMENT_MODE = "ln_only"  # the one variable that differs from C2


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
        description="Row C4: DGCNN, DA-A (adversarial), L-N augmentation only, "
                    "L_cls+L_seg+L_dom+L_ent")
    p.add_argument("--exp_name", type=str, default="C4_dgcnn_da_a_ln")
    p.add_argument("--out_path", type=str, default=str(_REPO_ROOT / "results"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--disc_dropout", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=10.0,
                    help="lambda_p(p) = 2/(1+exp(-gamma*p)) - 1, GRL alpha schedule")
    p.add_argument("--lambda_ent", type=float, default=0.1,
                    help="MAX weight on target entropy minimization -- ramped by lambda_p(p) "
                         "via dann.lambda_ent_schedule (see adapters/dann.py docstring)")
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
    indicator, never used for training or model selection. Same as
    C1/A1/C2/C3's held-out eval."""
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
    manifest_csv = data_dir / "preprocessed" / "preprocessed_manifest.csv"
    src_train_set = PlantClsSegDataset(
        data_dir / "crops3d_train.csv", manifest_csv, augment=True, seed=args.seed,
        augment_mode=AUGMENT_MODE)
    src_val_set = PlantClsSegDataset(
        data_dir / "crops3d_val.csv", manifest_csv, augment=False)
    # Target adaptation pool: points only, L-N augmentation applied (same
    # as source, per the row's DA-A/L-N spec) -- labels are loaded by
    # PlantSpeciesDataset (needed for its __getitem__ signature) but must
    # NEVER be used for anything except the separate held-out eval set
    # below. See module docstring.
    trgt_adapt_set = PlantSpeciesDataset(
        data_dir / "pheno4d_adaptation_pool.csv", manifest_csv, augment=True, seed=args.seed,
        augment_mode=AUGMENT_MODE)
    trgt_eval_set = PlantSpeciesDataset(
        data_dir / "pheno4d_heldout_eval.csv", manifest_csv, augment=False)

    io.cprint(f"Augment mode: {AUGMENT_MODE} (L-N only -- Gaussian jitter, no rotation/scale/"
              f"flip/crop/dropout/cubic symmetry)")
    io.cprint(f"Crops3D train (source): {len(src_train_set)} {src_train_set.class_counts()}")
    io.cprint(f"Crops3D val (source):   {len(src_val_set)} {src_val_set.class_counts()}")
    io.cprint(f"Pheno4D adaptation pool (target, unlabeled for training -- "
              f"species labels below shown for bookkeeping only, never used in the loss): "
              f"{len(trgt_adapt_set)} {trgt_adapt_set.class_counts()}")
    io.cprint(f"Pheno4D held-out eval (target, never trained on, cls-only): "
              f"{len(trgt_eval_set)} {trgt_eval_set.class_counts()}")

    # L_cls species-level class weights -- same w_c = (f_c+eps)^-1 pattern as C1/C2.
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
    # drop_last=True: BatchNorm in DomainDiscriminator needs batch size > 1;
    # this loader is cycled against src_train_loader (see below), so a
    # ragged final batch would just get dropped and re-cycled anyway.
    trgt_adapt_loader = DataLoader(trgt_adapt_set, batch_size=args.batch_size, shuffle=True,
                                    num_workers=NWORKERS, drop_last=True)
    trgt_eval_loader = DataLoader(trgt_eval_set, batch_size=args.test_batch_size, shuffle=False,
                                   num_workers=NWORKERS)

    n_batches_per_epoch = len(src_train_loader)  # epoch length is source-driven, as in C2
    io.cprint(f"Batches/epoch: source={n_batches_per_epoch} (drives epoch length), "
              f"target_adapt={len(trgt_adapt_loader)} (cycled)")

    model_args = argparse.Namespace(model="dgcnn", cuda=cuda, dropout=args.dropout)
    model = DGCNN_ClsSeg(model_args, num_class=2, seg_num_classes=SEG_NUM_CLASSES).to(device)
    domain_disc = DomainDiscriminator(in_dim=1024, dropout=args.disc_dropout).to(device)
    kendall = KendallUncertaintyWeighting(
        {"cls": "classification", "seg": "classification"}).to(device)
    best_model = copy.deepcopy(model)
    best_kendall = copy.deepcopy(kendall)

    # Single optimizer over backbone + discriminator + Kendall log-vars --
    # the GRL (not a separate optimizer/alternation) is what makes this a
    # min-max game inside one backward() call, standard DANN practice.
    opt = optim.Adam(
        list(model.parameters()) + list(domain_disc.parameters()) + list(kendall.parameters()),
        lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(opt, args.epochs)
    cls_criterion = nn.CrossEntropyLoss(weight=cls_class_weights)
    dom_criterion = nn.CrossEntropyLoss()  # unweighted -- source/target adaptation-pool sizes fixed per batch (drop_last on both)
    target_eval_criterion = nn.CrossEntropyLoss()

    best_val_total_loss, best_epoch = float("inf"), 0
    total_steps = args.epochs * n_batches_per_epoch
    for epoch in range(args.epochs):
        model.train()
        domain_disc.train()
        train_cls_loss, train_seg_loss, train_dom_loss, train_ent_loss, n_seen = 0.0, 0.0, 0.0, 0.0, 0
        last_lambda_p, last_lambda_ent = 0.0, 0.0

        trgt_iter = cycle(trgt_adapt_loader)
        for batch_idx, (src_batch, trgt_batch) in enumerate(zip(src_train_loader, trgt_iter)):
            src_pts, src_species, src_seg = src_batch
            trgt_pts, _trgt_species_unused = trgt_batch  # target labels intentionally never read

            src_pts = src_pts.permute(0, 2, 1).to(device)
            src_species = src_species.to(device)
            src_seg = src_seg.to(device)
            trgt_pts = trgt_pts.permute(0, 2, 1).to(device)

            step = epoch * n_batches_per_epoch + batch_idx
            p = step / total_steps
            lambda_p = lambda_p_schedule(p, gamma=args.gamma)
            lambda_ent = lambda_ent_schedule(p, args.lambda_ent, gamma=args.gamma)
            last_lambda_p, last_lambda_ent = lambda_p, lambda_ent

            opt.zero_grad()

            src_logits = model(src_pts, activate_DefRec=False)
            l_cls = cls_criterion(src_logits["cls"], src_species)
            l_seg = compute_seg_loss(model, src_logits["seg_feat"], src_species,
                                      src_seg, seg_class_weights)
            task_total = kendall({"cls": l_cls, "seg": l_seg})

            trgt_logits = model(trgt_pts, activate_DefRec=False)
            l_ent = entropy_loss(trgt_logits["cls"])  # unlabeled target predictions only

            src_feat_rev = grad_reverse(src_logits["feat"], lambda_p)
            trgt_feat_rev = grad_reverse(trgt_logits["feat"], lambda_p)
            dom_logits = torch.cat([domain_disc(src_feat_rev), domain_disc(trgt_feat_rev)], dim=0)
            dom_labels = torch.cat([
                torch.zeros(src_pts.size(0), dtype=torch.long, device=device),   # source = 0
                torch.ones(trgt_pts.size(0), dtype=torch.long, device=device),   # target = 1
            ], dim=0)
            l_dom = dom_criterion(dom_logits, dom_labels)

            # L_dom weight is 1 here (GRL alpha=lambda_p already ramps the
            # backbone's adversarial gradient); L_ent uses lambda_ent_schedule
            # (ramped by lambda_p, capped at --lambda_ent -- see
            # adapters/dann.py docstring, "REVISED 2026-09-11"). Neither
            # goes through `kendall`.
            total = task_total + l_dom + lambda_ent * l_ent
            total.backward()
            opt.step()

            bs = src_pts.size(0)
            train_cls_loss += l_cls.item() * bs
            train_seg_loss += l_seg.item() * bs
            train_dom_loss += l_dom.item() * bs
            train_ent_loss += l_ent.item() * bs
            n_seen += bs
        scheduler.step()

        s_cls = kendall.log_vars["cls"].item()
        s_seg = kendall.log_vars["seg"].item()
        io.cprint(f"Trn - Source {epoch}, cls loss: {train_cls_loss / n_seen:.4f}, "
                  f"seg loss: {train_seg_loss / n_seen:.4f}, "
                  f"dom loss: {train_dom_loss / n_seen:.4f}, "
                  f"ent loss (target): {train_ent_loss / n_seen:.4f}, "
                  f"lambda_p: {last_lambda_p:.4f}, lambda_ent: {last_lambda_ent:.4f}, "
                  f"kendall s_cls: {s_cls:.4f}, s_seg: {s_seg:.4f}")

        val_cls_acc, val_cls_avg_acc, val_cls_loss, val_seg_summary, val_seg_loss = \
            evaluate_cls_seg(model, src_val_loader, cls_criterion, seg_class_weights, device)
        with torch.no_grad():
            val_total_loss = kendall({
                "cls": torch.tensor(val_cls_loss, device=device),
                "seg": torch.tensor(val_seg_loss, device=device),
            }).item()
        io.cprint(f"Val - Source {epoch}, cls acc: {val_cls_acc:.4f}, "
                  f"cls avg acc: {val_cls_avg_acc:.4f}, cls loss: {val_cls_loss:.4f}, "
                  f"seg loss: {val_seg_loss:.4f}, total loss: {val_total_loss:.4f}")
        for species, m in val_seg_summary.items():
            io.cprint(f"Val - Source {epoch}, {species} seg: "
                      f"loss={m['loss']:.4f}, mIoU={m['mIoU']:.4f}, acc={m['acc']:.4f}")

        trgt_acc, trgt_avg_acc, trgt_loss, _, _, _ = evaluate_cls_only(
            model, trgt_eval_loader, target_eval_criterion, device)
        io.cprint(f"Eval(no-train) - Target {epoch}, acc: {trgt_acc:.4f}, "
                  f"avg acc: {trgt_avg_acc:.4f}, cls loss: {trgt_loss:.4f}  "
                  f"<- domain gap indicator (DA-A, cls-only, held-out -- never trained on)")

        # Model selection: lowest SOURCE val total loss only (never target
        # labels, never L_dom/L_ent -- see module docstring).
        if val_total_loss < best_val_total_loss:
            best_val_total_loss = val_total_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            best_kendall = copy.deepcopy(kendall)
            torch.save(model.state_dict(), os.path.join(exp_path, "model.pt"))
            torch.save(domain_disc.state_dict(), os.path.join(exp_path, "domain_disc.pt"))

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
