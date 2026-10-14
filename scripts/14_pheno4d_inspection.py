"""
14_pheno4d_inspection.py

Block PP: why does the soil-removed baseline (P4) score much lower on the held-out Pheno4D
clouds than the soil-kept baseline (P4a)? Look-and-measure only: nothing is trained and no
existing file is changed. Everything is read from the preprocessed caches
(data/preprocessed_variants/{P4,P4a}) and the saved checkpoints results/PP_{P4,P4a}_s{1,2,3}/
model.pt -- the raw clouds are not on the cluster.

Three parts, all written to results/pp_checks/pheno4d_before_after/:

1. Cloud statistics (cloud_stats.csv, stat_shift.csv)
   Per cloud, for the 63 held-out Pheno4D clouds and the 263 Crops3D training clouds, in both
   caches. Stored in the manifest: n_raw, n_after_soil_removal (points before voxel/FPS),
   n_predecimated (points entering FPS), voxel size and the normalisation scale (converted to
   mm; Crops3D coordinates are in metres -- inferred from its scales of 0.3-0.5, Pheno4D's are
   in mm). Measured on the 4096 normalised points the network sees: height (z range), width
   (larger of the x and y ranges), height/width, bottom z, mean radius, and local spacing
   (mean distance to the nearest neighbour; also converted to mm). For annotated Pheno4D
   clouds: fraction of the 4096 points labelled soil (raw id 0, per
   adapters/dataset.py::pheno4d_collapse_organ_labels -- an inferred mapping) and the fraction
   of plant points the soil cut removed, read from results/pp_checks/
   pheno4d_soil_removal_annotated.csv (margin 2.0 mm, computed earlier on the raw clouds).
   Crops3D maize: fraction of points labelled soil/pot (ids 0, 5).
   stat_shift.csv: per Pheno4D species and statistic, the two-sample Kolmogorov-Smirnov
   statistic against Crops3D training clouds of the same species and of the other species,
   for each cache.

2. Predictions (predictions.csv, per_cloud.csv, confusion.csv, acc_by_stage.csv)
   Each P4/P4a seed's protocol-selected checkpoint is evaluated on its own cache with the
   same model/data code as scripts/11_cross_eval.py; accuracy is checked against the
   run.log FINAL line. Growth stage = scan date split into three equal-count groups of the
   distinct dates per species (early/mid/late). A cloud is "P4 wrong, P4a right" when at
   least 2 of 3 P4 seeds are wrong and at least 2 of 3 P4a seeds are right.

3. Renders (PNG, one per cloud): the P4a cloud (top row) and the P4 cloud (bottom row) of the
   same scan, front (x-z), side (y-z) and top (x-y) projections in normalised space (what the
   network sees), grey shade by height, soil-labelled points in black where labels exist.
   12 clouds: up to 6 "P4 wrong, P4a right" clouds, the rest chosen to cover both species and
   all three stages.

Usage:
  python scripts/14_pheno4d_inspection.py --gpu -1
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import ks_2samp, mannwhitneyu  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "cross_eval", Path(__file__).resolve().parent / "11_cross_eval.py")
xe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xe)  # also puts adapters/ on sys.path and imports the C1 module
c1 = xe.c1
from dataset import pheno4d_collapse_organ_labels, SPECIES_TO_IDX  # noqa: E402

CACHES = ("P4a", "P4")
SEEDS = (1, 2, 3)
HELDOUT_CSV = _REPO_ROOT / "data" / "pheno4d_heldout_eval.csv"
TRAIN_CSV = _REPO_ROOT / "data" / "crops3d_train.csv"
SOIL_CUT_CSV = _REPO_ROOT / "results" / "pp_checks" / "pheno4d_soil_removal_annotated.csv"
SOIL_MARGIN_MM = 2.0  # preprocessing_variants.PHENO4D_SOIL_MARGIN_MM
UNIT_TO_MM = {"Crops3D": 1000.0, "Pheno4D": 1.0}
N_RENDER, N_RENDER_FLIPPED = 12, 6
STATS = ["n_raw", "n_after_soil_removal", "n_predecimated", "voxel_mm", "scale_mm",
         "height_mm", "height", "width", "aspect_h_w", "z_bottom", "mean_radius",
         "nn_spacing", "nn_spacing_mm", "soil_frac"]


def strip(p):
    p = str(p).replace("\\", "/")
    while p.startswith("../"):
        p = p[3:]
    return p


def load_manifest(cache):
    m = pd.read_csv(xe.manifest_for(cache))
    m["key"] = m.source_filepath.map(strip)
    return m.set_index("key")


def cloud_stats(row, cache):
    with np.load(_REPO_ROOT / strip(row.cache_path)) as d:
        pts = d["points"].astype(np.float64)
        labels = d["labels"] if "labels" in d.files else None
    unit = UNIT_TO_MM[row.dataset]
    x, y, z = pts.T
    height = z.max() - z.min()
    width = max(np.ptp(x), np.ptp(y))
    nn = cKDTree(pts).query(pts, k=2)[0][:, 1].mean()
    soil_frac = np.nan
    if labels is not None:
        if row.dataset == "Pheno4D":
            soil_frac = float((pheno4d_collapse_organ_labels(labels) == 0).mean())
        elif row.species == "Maize":
            soil_frac = float(np.isin(labels.reshape(-1), (0, 5)).mean())
        else:
            soil_frac = 0.0
    scale_mm = row.norm_scale * unit
    return dict(
        cache=cache, dataset=row.dataset, species=row.species, file=Path(row.source_filepath).name,
        plant_id=row.plant_id, scan_date=row.scan_date, n_raw=row.n_raw,
        n_after_soil_removal=row.n_after_soil_removal, n_predecimated=row.n_predecimated,
        voxel_mm=row.voxel_size_mm * unit, scale_mm=scale_mm, height_mm=height * scale_mm,
        height=height, width=width, aspect_h_w=height / width, z_bottom=z.min(),
        mean_radius=np.linalg.norm(pts, axis=1).mean(), nn_spacing=nn, nn_spacing_mm=nn * scale_mm,
        soil_frac=soil_frac)


def compute_stats(manifests, heldout, train):
    rows = []
    for cache in CACHES:
        m = manifests[cache]
        for split, df in (("pheno4d_heldout", heldout), ("crops3d_train", train)):
            for fp in df.filepath:
                r = m.loc[strip(fp)]
                s = cloud_stats(r, cache)
                s["split"] = split
                rows.append(s)
    st = pd.DataFrame(rows)
    cut = pd.read_csv(SOIL_CUT_CSV)
    cut = cut[np.isclose(cut.margin, SOIL_MARGIN_MM)][["file", "plant_lost_frac", "soil_removed_frac"]]
    cut["file"] = cut.file.str.replace(".txt", "", regex=False)
    st["stem"] = st.file.str.rsplit(".", n=1).str[0]
    st = st.merge(cut.rename(columns={"file": "stem"}), on="stem", how="left").drop(columns="stem")
    return st


def stat_shift(st):
    out = []
    for cache in CACHES:
        s = st[st.cache == cache]
        for tsp in ("Tomato", "Maize"):
            tgt = s[(s.split == "pheno4d_heldout") & (s.species == tsp)]
            for stat in STATS:
                tv = tgt[stat].dropna().to_numpy()
                row = {"cache": cache, "target_species": tsp, "stat": stat, "n_target": len(tv),
                       "target_median": np.median(tv) if len(tv) else np.nan}
                for ssp in ("Tomato", "Maize"):
                    sv = s[(s.split == "crops3d_train") & (s.species == ssp)][stat].dropna().to_numpy()
                    tag = "same" if ssp == tsp else "other"
                    row[f"src_{tag}_median"] = np.median(sv) if len(sv) else np.nan
                    row[f"ks_vs_src_{tag}"] = ks_2samp(tv, sv).statistic if len(tv) and len(sv) else np.nan
                out.append(row)
    sh = pd.DataFrame(out)
    wide = sh.pivot_table(index=["target_species", "stat"], columns="cache",
                          values=["ks_vs_src_same", "ks_vs_src_other"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["ks_same_change_P4_minus_P4a"] = wide.ks_vs_src_same_P4 - wide.ks_vs_src_same_P4a
    return sh, wide


@torch.no_grad()
def predict(cache, seed, device, cuda):
    run = f"PP_{cache}_s{seed}"
    model = xe.load_model(run, device, cuda).eval()
    ds = c1.PlantSpeciesDataset(HELDOUT_CSV, xe.manifest_for(cache), augment=False)
    if len(ds) != 63:
        raise RuntimeError(f"{cache}: expected 63 held-out clouds, got {len(ds)}")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    probs, true = [], []
    for pts, lab in loader:
        logits = model(pts.to(device).permute(0, 2, 1), activate_DefRec=False)["cls"]
        probs.append(torch.softmax(logits, 1)[:, SPECIES_TO_IDX["Maize"]].cpu().numpy())
        true.append(lab.numpy())
    p_maize, true = np.concatenate(probs), np.concatenate(true)
    pred = (p_maize > 0.5).astype(int)  # == argmax for 2 classes (ties impossible in practice)
    acc, bal = (pred == true).mean(), c1.metrics.balanced_accuracy_score(true, pred)
    ok, msg = xe.check_own(run, cache, acc, bal)
    print(msg, flush=True)
    if not ok:
        raise RuntimeError("evaluation does not reproduce run.log FINAL line: " + msg)
    files = [Path(p).stem for p, _ in ds.samples]
    return pd.DataFrame({"cache": cache, "seed": seed, "file": files, "true": true, "pred": pred,
                         "p_maize": p_maize})


def add_stage(heldout):
    h = heldout.copy()
    h["file"] = h.filepath.map(lambda p: Path(p).stem)
    h["stage"] = ""
    for sp, g in h.groupby("species"):
        dates = np.sort(g.scan_date.unique())
        for name, chunk in zip(("early", "mid", "late"), np.array_split(dates, 3)):
            h.loc[(h.species == sp) & h.scan_date.isin(chunk), "stage"] = name
    return h


def render(file, st_row, manifests, out_dir, preds_txt):
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2), dpi=72)
    views = [("front (x-z)", 0, 2), ("side (y-z)", 1, 2), ("top (x-y)", 0, 1)]
    for r, cache in enumerate(CACHES):
        row = manifests[cache].loc[strip(st_row.filepath)]
        with np.load(_REPO_ROOT / strip(row.cache_path)) as d:
            pts = d["points"]
            soil = (pheno4d_collapse_organ_labels(d["labels"]) == 0) if "labels" in d.files else None
        z = pts[:, 2]
        shade = 0.25 + 0.5 * (z - z.min()) / max(np.ptp(z), 1e-9)
        for c, (name, i, j) in enumerate(views):
            ax = axes[r, c]
            order = np.argsort(pts[:, 1 if c == 0 else 0] if c < 2 else z)  # rough depth sort
            ax.scatter(pts[order, i], pts[order, j], s=0.4, c=shade[order], cmap="gray",
                       vmin=0, vmax=1, linewidths=0)
            if soil is not None and soil.any():
                ax.scatter(pts[soil, i], pts[soil, j], s=0.4, c="black", linewidths=0)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=8)
        axes[r, 0].set_ylabel(f"{cache}\nscale {row.norm_scale * UNIT_TO_MM['Pheno4D']:.0f} mm",
                              fontsize=8)
    fig.suptitle(f"{file}  ({st_row.species}, date {int(st_row.scan_date)}, {st_row.stage})\n"
                 f"{preds_txt}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = out_dir / f"{st_row.species.lower()}_{file}.png"
    fig.savefig(path, dpi=72)
    plt.close(fig)
    return path.name


def main():
    ap = argparse.ArgumentParser(description="Pheno4D P4 vs P4a inspection (no training)")
    ap.add_argument("--gpu", type=int, default=-1, help="-1 for cpu")
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "results" / "pp_checks" / "pheno4d_before_after"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)

    manifests = {c: load_manifest(c) for c in CACHES}
    heldout = add_stage(pd.read_csv(HELDOUT_CSV))
    train = pd.read_csv(TRAIN_CSV)

    # ---- 1. statistics ----
    st = compute_stats(manifests, heldout, train)
    st.to_csv(out_dir / "cloud_stats.csv", index=False, float_format="%.6g")
    sh, wide = stat_shift(st)
    sh.to_csv(out_dir / "stat_shift.csv", index=False, float_format="%.4g")
    print("\n== KS statistic, Pheno4D held-out vs Crops3D train (same species / other species)")
    print(wide.sort_values(["target_species", "ks_vs_src_same_P4"], ascending=[True, False])
          .to_string(index=False, float_format="%.3f"))
    med = (st.groupby(["cache", "split", "species"])[STATS].median().T)
    print("\n== Medians per cache / split / species\n" + med.to_string(float_format="%.4g"))

    # ---- 2. predictions ----
    preds = pd.concat([predict(c, s, device, cuda) for c in CACHES for s in SEEDS], ignore_index=True)
    preds = preds.merge(heldout[["file", "species", "plant_id", "scan_date", "stage", "is_annotated"]],
                        on="file", how="left")
    preds["correct"] = (preds.pred == preds.true).astype(int)
    preds.to_csv(out_dir / "predictions.csv", index=False, float_format="%.4f")

    conf_rows = []
    for (cache, seed), g in preds.groupby(["cache", "seed"]):
        for sp, gs in g.groupby("species"):
            conf_rows.append({"cache": cache, "seed": seed, "true": sp, "n": len(gs),
                              "pred_Tomato": int((gs.pred == 0).sum()), "pred_Maize": int((gs.pred == 1).sum()),
                              "recall": gs.correct.mean()})
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(out_dir / "confusion.csv", index=False, float_format="%.4f")
    print("\n== Confusion matrix and per-class recall per seed\n" + conf.to_string(index=False))
    print("\n== Per-class recall, mean over seeds\n"
          + conf.groupby(["cache", "true"]).recall.agg(["mean", "std"]).to_string(float_format="%.3f"))

    by_stage = (preds.groupby(["cache", "species", "stage", "seed"]).correct.mean()
                .groupby(["cache", "species", "stage"]).agg(["mean", "std"]).reset_index())
    n_stage = preds[(preds.cache == "P4") & (preds.seed == 1)].groupby(["species", "stage"]).size()
    by_stage["n_clouds"] = [n_stage[(r.species, r.stage)] for r in by_stage.itertuples()]
    by_date = (preds.groupby(["cache", "species", "scan_date"]).correct.mean().unstack("cache").reset_index())
    by_stage.to_csv(out_dir / "acc_by_stage.csv", index=False, float_format="%.4f")
    by_date.to_csv(out_dir / "acc_by_date.csv", index=False, float_format="%.4f")
    print("\n== Accuracy by growth stage (mean ± std over 3 seeds)\n" + by_stage.to_string(index=False, float_format="%.3f"))
    print("\n== Accuracy by scan date (mean over 3 seeds)\n" + by_date.to_string(index=False, float_format="%.3f"))

    # per-cloud: seeds correct per cache + P4/P4a statistics side by side
    pc = (preds.groupby(["file", "cache"]).agg(n_correct=("correct", "sum"), p_maize=("p_maize", "mean"))
          .unstack("cache"))
    pc.columns = [f"{a}_{b}" for a, b in pc.columns]
    pc = pc.reset_index().merge(heldout[["file", "filepath", "species", "plant_id", "scan_date", "stage",
                                         "is_annotated"]], on="file")
    sth = st[st.split == "pheno4d_heldout"].copy()
    sth["file"] = sth.file.str.rsplit(".", n=1).str[0]
    keep = ["height_mm", "scale_mm", "height", "width", "aspect_h_w", "z_bottom", "nn_spacing",
            "nn_spacing_mm", "n_after_soil_removal", "soil_frac"]
    for cache in CACHES:
        s = sth[sth.cache == cache].set_index("file")[keep].add_suffix(f"_{cache}")
        pc = pc.merge(s, left_on="file", right_index=True)
    pc = pc.merge(sth[sth.cache == "P4"].set_index("file")[["plant_lost_frac", "soil_removed_frac"]],
                  left_on="file", right_index=True)
    pc["group"] = np.select(
        [(pc.n_correct_P4 <= 1) & (pc.n_correct_P4a >= 2), (pc.n_correct_P4 >= 2) & (pc.n_correct_P4a >= 2),
         (pc.n_correct_P4 >= 2) & (pc.n_correct_P4a <= 1)],
        ["P4 wrong, P4a right", "both right", "P4 right, P4a wrong"], "both wrong")
    pc = pc.sort_values(["species", "scan_date", "plant_id"])
    pc.drop(columns="filepath").to_csv(out_dir / "per_cloud.csv", index=False, float_format="%.4g")
    print("\n== Cloud groups (majority over 3 seeds)\n"
          + pc.groupby(["species", "group"]).size().unstack(fill_value=0).to_string())
    flipped = pc[pc.group == "P4 wrong, P4a right"]
    print("\n== P4 wrong, P4a right\n" + flipped[["file", "species", "plant_id", "scan_date", "stage",
                                                  "n_correct_P4", "n_correct_P4a", "p_maize_P4",
                                                  "p_maize_P4a", "height_mm_P4", "height_mm_P4a",
                                                  "soil_frac_P4", "soil_frac_P4a", "plant_lost_frac"]]
          .to_string(index=False, float_format="%.3f"))

    # flipped vs the rest within each species: which statistics separate them?
    comp = []
    for sp, g in pc.groupby("species"):
        a, b = g[g.group == "P4 wrong, P4a right"], g[g.group != "P4 wrong, P4a right"]
        for col in [c for c in pc.columns if c.endswith("_P4") and not c.startswith(("n_correct", "p_maize"))] \
                + ["plant_lost_frac", "scan_date"]:
            va, vb = a[col].dropna(), b[col].dropna()
            p = mannwhitneyu(va, vb).pvalue if len(va) >= 3 and len(vb) >= 3 else np.nan
            comp.append({"species": sp, "stat": col, "n_flipped": len(va), "n_other": len(vb),
                         "median_flipped": va.median() if len(va) else np.nan,
                         "median_other": vb.median() if len(vb) else np.nan, "mannwhitney_p": p})
    comp = pd.DataFrame(comp)
    comp.to_csv(out_dir / "flipped_vs_rest.csv", index=False, float_format="%.4g")
    print("\n== Flipped clouds vs rest, P4 statistics (Mann-Whitney p, small N -- descriptive only)\n"
          + comp.to_string(index=False, float_format="%.4g"))

    # ---- 3. renders ----
    for f in out_dir.glob("*.png"):
        f.unlink()  # only this script's own outputs live here
    chosen = []
    fl = flipped.sort_values(["species", "stage"])
    # spread the flipped picks over species/stages first
    for _, g in fl.groupby(["species", "stage"]):
        chosen.append(g.iloc[0].file)
    for f in fl.file:
        if len(chosen) >= N_RENDER_FLIPPED:
            break
        if f not in chosen:
            chosen.append(f)
    chosen = chosen[:N_RENDER_FLIPPED]
    rest = pc[~pc.file.isin(chosen)]
    for _, g in rest.groupby(["species", "stage"]):  # one per species x stage
        if len(chosen) < N_RENDER:
            chosen.append(g.sort_values("plant_id").iloc[len(g) // 2].file)
    for f in rest.file:
        if len(chosen) >= N_RENDER:
            break
        if f not in chosen:
            chosen.append(f)
    names = []
    for f in chosen:
        r = pc[pc.file == f].iloc[0]
        g = preds[preds.file == f].sort_values("seed")
        txt = "  |  ".join(
            f"{c} preds s1-s3: " + ",".join("T" if p == 0 else "M" for p in g[g.cache == c].pred)
            for c in CACHES) + f"  |  {r.group}"
        names.append((render(f, r, manifests, out_dir, txt), r.group))
    pd.DataFrame(names, columns=["png", "group"]).to_csv(out_dir / "renders.csv", index=False)
    print("\n== Renders\n" + "\n".join(f"  {n}  [{g}]" for n, g in names))


if __name__ == "__main__":
    main()
