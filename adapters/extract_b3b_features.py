"""
extract_b3b_features.py

D1 (Block D fusion baseline) prerequisite: extracts and caches B3b's
pooled global feature ("deep geometric features", F1 per the design
docs -- adapters.models_kpconv.KPConv_ClsSeg.forward's logits["feat"],
confirmed 256-dim) for every Pheno4D scan, so D1's actual training loop
runs on cheap precomputed tensors instead of repeating KPConv's
CPU-bound collate step (the documented bottleneck for every Block B row)
on every epoch.

B3b's encoder is FROZEN for D1 (confirmed with the user -- standard
choice for a small downstream dataset on top of an already-trained
encoder, and keeps D1-D6 comparisons clean since only the fusion input
changes, not backbone weights). This script only ever runs a forward
pass under torch.no_grad() -- there is no training here, so "frozen"
isn't even a runtime flag, it's simply the only thing this script does.

Reuses adapters.infer_segmentation.load_b3b_model (exact model
reconstruction: PlantKPConvConfig -> KPConv_ClsSeg -> load_state_dict,
same pattern every KPConv row in this project follows) and
calibrate_neighborhood_limits (same convention as every other KPConv
inference/training script -- recalibrated per run since B3b's own
checkpoint never persisted its training-time neighborhood_limits, see
infer_segmentation.py's module docstring for the full reasoning).

Output: one .npz per scan, data/features/Pheno4D/<species>/<stem>.npz,
holding {"feat": (256,) float32} -- mirrors the segmented-cache layout
convention (data/segmented/Pheno4D/<species>/<stem>.npz) established in
infer_segmentation.py.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFREC_ROOT = os.environ.get("DEFREC_ROOT", str(_REPO_ROOT.parent / "DefRec_and_PCM"))
if _DEFREC_ROOT not in sys.path:
    sys.path.insert(0, _DEFREC_ROOT)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_segmentation import load_b3b_model, DEFAULT_CHECKPOINT  # noqa: E402
from kpconv_collate import calibrate_neighborhood_limits, make_kpconv_collate_fn_cls_only  # noqa: E402
from dataset import PlantSpeciesDataset  # noqa: E402


def extract_features(checkpoint_path=DEFAULT_CHECKPOINT, dropout=0.5, device="cpu",
                      batch_size=16, num_workers=0, out_dir=None, limit=None):
    if out_dir is None:
        out_dir = _REPO_ROOT / "data" / "features" / "Pheno4D"
    out_dir = Path(out_dir)

    data_dir = _REPO_ROOT / "data"
    manifest_csv = data_dir / "preprocessed" / "preprocessed_manifest.csv"

    pool_set = PlantSpeciesDataset(
        data_dir / "pheno4d_adaptation_pool.csv", manifest_csv, augment=False)
    heldout_set = PlantSpeciesDataset(
        data_dir / "pheno4d_heldout_eval.csv", manifest_csv, augment=False)
    print(f"[extract] pool: {len(pool_set)}, heldout: {len(heldout_set)}, "
          f"total: {len(pool_set) + len(heldout_set)} (expect 223)")

    model, config = load_b3b_model(checkpoint_path, dropout, device)
    model.eval()

    from torch.utils.data import ConcatDataset
    calib_set = ConcatDataset([pool_set, heldout_set])
    config.neighborhood_limits = calibrate_neighborhood_limits(
        config, calib_set, batch_size, num_workers=num_workers,
        collate_fn_factory=make_kpconv_collate_fn_cls_only)
    print(f"[extract] calibrated neighborhood_limits: {config.neighborhood_limits}")

    n_written = 0
    for split_name, dataset in [("adaptation_pool", pool_set), ("heldout_eval", heldout_set)]:
        n_this_split = min(limit, len(dataset)) if limit else len(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers,
                             collate_fn=make_kpconv_collate_fn_cls_only(config))
        sample_i = 0
        n_processed_this_split = 0
        for batch in loader:
            if limit and n_processed_this_split >= limit:
                break
            bs = batch.species_labels.shape[0]
            cache_paths_in_batch = [dataset.samples[sample_i + j][0] for j in range(bs)]
            sample_i += bs

            batch = batch.to(device)
            with torch.no_grad():
                logits = model(batch, activate_DefRec=False)
            feat = logits["feat"].cpu().numpy()  # (B, 256)

            for j in range(bs):
                if limit and n_processed_this_split >= limit:
                    break
                cache_path = Path(cache_paths_in_batch[j])
                species_name = cache_path.parent.name
                dest = out_dir / species_name / cache_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, feat=feat[j].astype(np.float32))
                n_written += 1
                n_processed_this_split += 1
        print(f"[extract] {split_name}: done ({n_processed_this_split} scans)")

    print(f"[extract] wrote {n_written} feature .npz files to {out_dir}")
    return n_written


def parse_args():
    p = argparse.ArgumentParser(description="D1: extract+cache B3b's frozen pooled features")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=-1, help="-1 for cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--limit", type=int, default=None,
                    help="only process the first N scans per split (for smoke testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print(f"Using {'GPU ' + str(args.gpu) if cuda else 'CPU'}")
    extract_features(args.checkpoint, args.dropout, device, args.batch_size,
                      args.num_workers, args.out_dir, args.limit)
