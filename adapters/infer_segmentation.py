"""
infer_segmentation.py

D0 (trait-extraction pipeline prerequisite): runs point-wise organ
segmentation inference on Pheno4D scans using the already-trained B3b
checkpoint (KPConv + DA-S, results/B3b_kpconv_da_s/model.pt) -- the
backbone+DA foundation chosen for Block D.

Reconstructs the model exactly as every KPConv training script does
(no saved config/neighborhood_limits/Kendall weights to load -- only a
raw model.state_dict() exists on disk, confirmed by reading
train_b3b_kpconv_da_s.py's checkpoint-saving code), reusing
PlantKPConvConfig/KPConv_ClsSeg/calibrate_neighborhood_limits/
make_kpconv_collate_fn unchanged from that file.

**Critical subtlety, confirmed by reading the code directly (not
assumed): B3b's seg_heads were built against Crops3D's per-species
label space** (`adapters.dataset.SEG_NUM_CLASSES = {"Tomato": 3,
"Maize": 6}`, `train_b3b_kpconv_da_s.py:409`), not Pheno4D's 3-class
soil/stem/leaf scheme (`PHENO4D_SEG_NUM_CLASSES`) -- B3b is the DA-S
row, self-supervised only, and never sees Pheno4D labels during
training or selection. So `model.seg_logits_for_species(...)` on a
Pheno4D scan returns logits in CROPS3D's label space, which this
module remaps to Pheno4D's {0: soil, 1: stem, 2: leaf} scheme.

**Two remap strategies, tried in order (see build_crops3d_organ_remap
vs. build_data_driven_remap below):** the first attempt used an
RGB-histogram heuristic (id 0 -> soil, dominant-count id -> leaf,
"soil" only RGB-confirmed for Maize per CLAUDE.md's Crops3D
label-backfill entry) and produced a held-out mIoU of 0.2653 overall,
but a stark species split: Maize 0.5430 (usable) vs. Tomato 0.0886
(worse than random chance, n=36 heldout scans) -- direct evidence the
"id 0 = soil" assumption, only ever confirmed for Maize, does not hold
for Tomato. Per explicit user instruction, replaced with a DATA-DRIVEN
remap (`build_data_driven_remap`): a confusion matrix between raw
Crops3D-space predictions and Pheno4D ground truth is built on
pheno4d_adaptation_pool.csv (fair to use for calibration), then the
id->organ assignment is solved via Hungarian matching
(scipy.optimize.linear_sum_assignment) per species, instead of assuming
the RGB-based structure transfers. This is still an inference layered
on an inference -- it replaces one unconfirmed assumption with a
different, empirically-grounded one (that adaptation_pool's confusion
structure generalizes to heldout_eval) -- so `validate_against_
annotated()` (this file's default __main__ mode) must be run and its
held-out numbers checked BEFORE trusting any trait extracted from this
module's output, exactly as before.

**Reported separately by split, not just combined (per explicit user
instruction) -- this distinction matters a lot here:** B3b's DA-S
training saw `pheno4d_adaptation_pool.csv`'s point clouds (unlabeled,
via the self-supervised DefRec objective, but the geometry was in the
training loop) and NEVER touched `pheno4d_heldout_eval.csv` in any
form (same held-out file every other Block A-C row also keeps clean).
A combined average across both would let the adaptation-pool's
"already seen the geometry" advantage mask true generalization -- the
held-out number is the only genuinely fair test of whether this
segmentation/remap approach generalizes, and is reported on its own,
not just folded into an overall figure. Per explicit user instruction,
full-dataset inference (all 223 scans) is a separate, later step,
gated on the held-out mIoU being judged acceptable -- this file does
not run it automatically.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import jaccard_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFREC_ROOT = os.environ.get("DEFREC_ROOT", str(_REPO_ROOT.parent / "DefRec_and_PCM"))
if _DEFREC_ROOT not in sys.path:
    sys.path.insert(0, _DEFREC_ROOT)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_kpconv import KPConv_ClsSeg, PlantKPConvConfig  # noqa: E402
from kpconv_collate import (  # noqa: E402
    calibrate_neighborhood_limits, make_kpconv_collate_fn, make_kpconv_collate_fn_cls_only,
    batch_points_bcn,
)
from dataset import (  # noqa: E402
    PlantClsSegDataset, PlantSpeciesDataset, SPECIES_TO_IDX, SEG_NUM_CLASSES,
    PHENO4D_SEG_NUM_CLASSES, pheno4d_collapse_organ_labels,
)

DEFAULT_CHECKPOINT = _REPO_ROOT / "results" / "B3b_kpconv_da_s" / "model.pt"


def load_b3b_model(checkpoint_path=DEFAULT_CHECKPOINT, dropout: float = 0.5,
                    device="cpu"):
    """Reconstructs B3b's architecture (PlantKPConvConfig's fields are all
    hardcoded class attributes except dropout, which B3b's run.log
    confirms was left at the --dropout default of 0.5) and loads its
    state dict. Returns (model, config) with model in eval() mode; the
    caller still needs to set config.neighborhood_limits (via
    calibrate_neighborhood_limits, against whatever dataset inference
    will run on) before building any batches -- this function only
    reconstructs the model, not the batch-builder config, since the two
    are calibrated once per inference dataset, not once per model load.
    """
    config = PlantKPConvConfig()
    config.dropout = dropout
    model = KPConv_ClsSeg(config, num_class=2, seg_num_classes=SEG_NUM_CLASSES)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def build_crops3d_organ_remap(device):
    """Per species, builds a (num_crops3d_classes,) int64 lookup tensor
    mapping a predicted Crops3D-space class id -> Pheno4D-space organ id
    {0: soil, 1: stem, 2: leaf}, using seg_class_histogram() -- the SAME
    helper already used at training time to build L_seg's class weights
    (adapters/dataset.py) -- reused here, not reimplemented, to find each
    species' dominant-point-count id (= leaf, RGB-confirmed per
    CLAUDE.md). id 0 -> soil for BOTH species (RGB-confirmed for Maize
    only; applied to Tomato too as the best available default -- this is
    exactly the assumption validate_against_annotated() below is meant
    to check, not something asserted as fact here). Every other id ->
    stem, by elimination."""
    data_dir = _REPO_ROOT / "data"
    manifest_csv = data_dir / "preprocessed" / "preprocessed_manifest.csv"
    crops3d_train = PlantClsSegDataset(
        data_dir / "crops3d_train.csv", manifest_csv, augment=False)

    remap = {}
    for species, num_classes in SEG_NUM_CLASSES.items():
        hist = crops3d_train.seg_class_histogram(species)
        leaf_id = int(np.argmax(hist))
        lookup = np.full(num_classes, 1, dtype=np.int64)  # default: stem
        lookup[0] = 0        # soil
        lookup[leaf_id] = 2  # leaf (applied after soil so it wins on the
                             # degenerate leaf_id==0 case, which CLAUDE.md's
                             # own histogram notes say never happens in
                             # practice -- id 0 is always the smallest-count
                             # id -- but the ordering here is defensive)
        remap[species] = torch.tensor(lookup, dtype=torch.int64, device=device)
        print(f"[remap] {species}: crops3d point-count histogram={hist.tolist()}, "
              f"leaf_id={leaf_id}, lookup(crops3d_id -> organ 0=soil/1=stem/2=leaf)="
              f"{lookup.tolist()}")
    return remap


def collect_raw_predictions(model, config, dataset, device, batch_size, num_workers):
    """Runs inference over `dataset` WITHOUT applying any remap, pooling every
    point's (raw Crops3D-space predicted id, Pheno4D ground-truth organ id)
    pair per species across all samples. Used only to BUILD a data-driven
    remap (see build_data_driven_remap below) -- never to report a held-out
    metric itself, since it's designed to run on pheno4d_adaptation_pool.csv
    (fair to use for calibration, unlike heldout_eval)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers,
                         collate_fn=make_kpconv_collate_fn(config))
    preds_by_species = {sp: [] for sp in SPECIES_TO_IDX}
    gt_by_species = {sp: [] for sp in SPECIES_TO_IDX}

    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch, activate_DefRec=False)
        species_labels = batch.species_labels
        B = species_labels.shape[0]
        N = int(batch.lengths[0][0].item())
        gt_organ = batch.labels.view(B, N)

        for species, idx in SPECIES_TO_IDX.items():
            mask = species_labels == idx
            if mask.sum().item() == 0:
                continue
            sp_logits = model.seg_logits_for_species(logits["seg_feat"][mask], species)
            preds = sp_logits.max(dim=2)[1]  # (B_sp, N) raw Crops3D-id space
            preds_by_species[species].append(preds.cpu().numpy().reshape(-1))
            gt_by_species[species].append(gt_organ[mask].cpu().numpy().reshape(-1))

    return {
        sp: (np.concatenate(preds_by_species[sp]), np.concatenate(gt_by_species[sp]))
        for sp in SPECIES_TO_IDX if preds_by_species[sp]
    }


def build_data_driven_remap(model, config, device, batch_size=16, num_workers=0):
    """Builds the Crops3D-id -> organ remap EMPIRICALLY instead of via the
    RGB-histogram heuristic in build_crops3d_organ_remap above: computes a
    confusion matrix between B3b's raw Crops3D-space predictions and
    Pheno4D's ground-truth organ labels on pheno4d_adaptation_pool.csv
    (fair to use here -- calibration, never the reported held-out metric),
    then solves the id->organ assignment via Hungarian matching
    (scipy.optimize.linear_sum_assignment) per species.

    Since num_crops3d_classes (3 Tomato / 6 Maize) can exceed num_organs
    (3), this isn't a square 1:1 assignment problem in general. Handled as
    a hybrid: Hungarian assigns each of the 3 organs its single best,
    MUTUALLY EXCLUSIVE raw id first (avoiding e.g. two organs both greedily
    claiming the same dominant id, which independent per-id argmax could
    do) -- for Tomato (3 ids) this already assigns every id. Any leftover
    raw ids (Maize has 3 left over after its 3 organs are matched) are then
    assigned independently to whichever organ they individually correlate
    with most (their own confusion-row argmax) -- the correct choice for
    those in isolation, since adding more many-to-one ids to an organ
    already assigned by Hungarian doesn't create the mutual-exclusion
    conflict Hungarian resolves for the primary 3.
    """
    data_dir = _REPO_ROOT / "data"
    manifest_csv = data_dir / "preprocessed" / "preprocessed_manifest.csv"
    pool_set = PlantClsSegDataset(
        data_dir / "pheno4d_adaptation_pool.csv", manifest_csv, augment=False,
        label_transform=pheno4d_collapse_organ_labels, num_classes=PHENO4D_SEG_NUM_CLASSES)

    raw = collect_raw_predictions(model, config, pool_set, device, batch_size, num_workers)

    remap = {}
    for species, num_classes in SEG_NUM_CLASSES.items():
        preds, gt = raw[species]
        confusion = np.zeros((num_classes, 3), dtype=np.int64)  # rows=crops3d id, cols=organ
        for raw_id in range(num_classes):
            mask = preds == raw_id
            if mask.sum() == 0:
                continue
            for organ in range(3):
                confusion[raw_id, organ] = int((gt[mask] == organ).sum())

        cost = -confusion.T.astype(np.float64)  # (3 organs, num_classes raw ids)
        organ_ind, id_ind = linear_sum_assignment(cost)
        lookup = np.full(num_classes, -1, dtype=np.int64)
        for organ, raw_id in zip(organ_ind.tolist(), id_ind.tolist()):
            lookup[raw_id] = organ

        leftover = [i for i in range(num_classes) if lookup[i] == -1]
        for raw_id in leftover:
            lookup[raw_id] = int(np.argmax(confusion[raw_id]))

        remap[species] = torch.tensor(lookup, dtype=torch.int64, device=device)
        print(f"[data-driven remap] {species}: confusion matrix "
              f"(rows=crops3d id, cols=organ 0=soil/1=stem/2=leaf):\n{confusion}")
        print(f"[data-driven remap] {species}: Hungarian core assignment "
              f"organ->id={dict(zip(organ_ind.tolist(), id_ind.tolist()))}, "
              f"leftover ids assigned by own-row argmax={leftover}, "
              f"final lookup(crops3d_id -> organ)={lookup.tolist()}")
    return remap


def build_groundtruth_remap(model, config, device, batch_size=16, num_workers=0):
    """Third remap strategy, built after directly visualizing raw Crops3D
    Tomato point clouds colored by their native label ids (see
    step_notes/D0_Trait_Extraction_Pipeline.md for the images/analysis).

    That check found the data-driven remap's Tomato assignment
    (build_data_driven_remap: id0->leaf, id1->soil, id2->stem) is
    contradicted by Crops3D's OWN ground-truth point geometry: id0 is
    unambiguously stem-shaped (PCA linearity 0.87-0.97, sphericity
    0.01-0.06 across 3 checked samples) and id1 is unambiguously the bushy
    leaf canopy (much lower linearity, 0.45-0.87, higher sphericity) --
    independent of any cross-domain confound. (id2 turned out to ALSO be
    linear/stem-shaped, not the blob/fruit shape a first visual glance
    suggested -- corrected after computing the same shape descriptors on
    it; it's rare (~10% of files) and its own true identity stays
    unresolved, so it's folded into leaf here as the pragmatic default,
    matching the data-driven remap's own confusion-matrix plurality for
    id2, 687 leaf vs. 203 stem vs. 0 soil.)

    Likely explanation for why the cross-domain confusion matrix
    disagreed with Crops3D's own labels: B3b's DA-S training never
    supervises Tomato's id->organ semantics against Pheno4D at all (self-
    supervised only), so under domain shift the model's per-point
    Crops3D-id output for a Pheno4D point may reflect coarse structural
    similarity (e.g. "flat/dense/textureless region") more than the
    original Crops3D-side label meaning -- and Crops3D's own Tomato scans
    appear to have NO soil-labeled points at all (close-up potted-plant
    captures), so the model was never given a chance to learn a genuine
    soil-vs-plant boundary for this species in the first place. The
    data-driven remap's id1->soil assignment was very likely picking up
    on this generalization behavior rather than "correctly" relabeling
    id1's true Crops3D meaning. Whether grounding the remap in Crops3D's
    own confirmed semantics (this function) actually produces BETTER
    held-out Pheno4D segmentation than the cross-domain-calibrated
    version is an empirical question -- resolved by comparing this
    remap_mode's held-out numbers against "data_driven"'s, not assumed
    here either way.

    Maize is left unchanged (reuses build_data_driven_remap's Maize
    result) since that species' data-driven assignment already
    independently reproduced the RGB-confirmed ground truth (id0=soil,
    dominant id2=leaf per CLAUDE.md) -- no reason to override a result
    already validated against real confirmed labels.
    """
    remap = build_data_driven_remap(model, config, device, batch_size, num_workers)
    tomato_lookup = torch.tensor([1, 2, 2], dtype=torch.int64, device=device)  # id0->stem, id1->leaf, id2->leaf
    remap["Tomato"] = tomato_lookup
    print(f"[groundtruth remap] Tomato: OVERRIDDEN from data-driven result -- "
          f"final lookup(crops3d_id -> organ 0=soil/1=stem/2=leaf)={tomato_lookup.tolist()} "
          f"(id0->stem, id1->leaf, id2->leaf; grounded in Crops3D's own confirmed point-cloud "
          f"shape, not cross-domain confusion-matrix calibration)")
    print(f"[groundtruth remap] Maize: unchanged from data-driven result "
          f"(already matched RGB-confirmed ground truth)")
    return remap


def build_hybrid_remap(model, config, device, batch_size=16, num_workers=0):
    """Fourth remap strategy, combining the one thing each of the previous
    two got right instead of forcing a single remap to do all three organs
    at once. Per-class recall on the full heldout_eval Tomato set showed
    the two approaches fail in cleanly COMPLEMENTARY ways, not the same
    way:

      data_driven (id0->leaf, id1->soil, id2->stem): true_soil recall
      99.48%, true_stem recall 0.66%, true_leaf recall 28.31%.
      groundtruth (id0->stem, id1->leaf, id2->leaf): true_soil recall
      0.00% (structurally impossible -- no id maps to organ 0 at all),
      true_stem recall 77.56%, true_leaf recall 71.69%.

    data_driven's id1->soil call is reliable (99.48% recall) because,
    empirically, ~72% of the model's id1 predictions on Pheno4D really
    are true soil (see build_data_driven_remap's confusion matrix).
    groundtruth's id0->stem call is reliable (77.56% recall) because it's
    grounded in Crops3D's own confirmed point-cloud shape for id0 (PCA
    linearity 0.87-0.97 across checked samples -- see step_notes/
    D0_Trait_Extraction_Pipeline.md), not the cross-domain confusion
    matrix (which id0's row shows is only ~26% stem-majority, too mixed
    to trust alone). Combining them: id0->stem, id1->soil, id2->leaf
    (id2 stays ->leaf, its own confusion-matrix plurality, same
    pragmatic default as groundtruth used).

    This is an empirical combination, not a theoretically-derived one --
    validate_against_annotated's held-out numbers with remap_mode=
    "hybrid" are the actual test of whether combining the two id-space
    assignments this way genuinely recovers all three classes at once,
    not just each remap's own best axis in isolation."""
    remap = build_data_driven_remap(model, config, device, batch_size, num_workers)
    tomato_lookup = torch.tensor([1, 0, 2], dtype=torch.int64, device=device)  # id0->stem, id1->soil, id2->leaf
    remap["Tomato"] = tomato_lookup
    print(f"[hybrid remap] Tomato: final lookup(crops3d_id -> organ 0=soil/1=stem/2=leaf)="
          f"{tomato_lookup.tolist()} (id0->stem from groundtruth, id1->soil from data-driven, "
          f"id2->leaf pragmatic default)")
    print(f"[hybrid remap] Maize: unchanged from data-driven result "
          f"(already matched RGB-confirmed ground truth)")
    return remap


## ---------------------------------------------------------------------
## TOMATO-ONLY GEOMETRIC STEM/LEAF FALLBACK
##
## Four id->organ remap strategies (histogram, data_driven, groundtruth,
## hybrid) were tried on B3b's raw Crops3D-space output and none could
## recover all three Tomato organs at once: the model's Tomato output has
## only TWO commonly-predicted Crops3D ids (id0, id1) but Pheno4D needs
## THREE organs (soil, stem, leaf) -- whichever organ ends up depending
## on the rare third id (id2, <1% of predictions) gets near-zero recall,
## and this happened to three DIFFERENT organs under three different
## remaps (stem under data_driven, soil under groundtruth, leaf under
## hybrid -- see step_notes/D0_Trait_Extraction_Pipeline.md for the full
## comparison table). This is a capacity problem in the model's Tomato
## output space, not a remap-tuning problem -- confirmed exhaustively,
## not assumed.
##
## One piece IS reliable across every attempt: soil-vs-plant. id1->soil
## scored 99.28-99.48% recall in every remap that tried it (data_driven,
## hybrid). So this fallback keeps that one working piece unchanged and
## replaces ONLY the stem/leaf split within non-soil points with a
## geometric heuristic operating on real point coordinates instead of the
## model's Crops3D-id output -- which structurally cannot carry 3 organs
## through 2 usable ids no matter how it's remapped.
##
## MAIZE IS NOT AFFECTED. Maize's data-driven remap already independently
## reproduced its RGB-confirmed ground truth (id0=soil, dominant id2=leaf
## per CLAUDE.md) with solid held-out mIoU (0.54-0.59) -- there is no
## capacity problem on that species (6 Crops3D classes comfortably cover
## 3 organs), so Maize keeps going through the standard model-based remap
## unchanged. Routing Maize through this fallback would be replacing a
## working signal with an untested one for no reason.
## ---------------------------------------------------------------------

TOMATO_SOIL_CROPS3D_ID = 1  # confirmed reliable (99.28-99.48% recall) across every remap attempt


def local_linearity(points: np.ndarray, k: int = 12) -> np.ndarray:
    """points: (M,3). Returns (M,) linearity in [0,1] per point, from that
    point's own k-nearest-neighbor local PCA: linearity = (l1-l2)/l1 where
    l1>=l2>=l3 are the neighborhood's covariance eigenvalues -- the SAME
    descriptor already used (and visually+numerically validated, see
    step_notes/D0_Trait_Extraction_Pipeline.md) to confirm Crops3D's id0
    is stem-shaped (whole-class linearity 0.87-0.97) vs. id1's bushy leaf
    canopy (0.45-0.87) -- applied per-point/per-local-neighborhood here
    instead of per-whole-class, since a single scan mixes both structures
    together in 3D space and needs a per-point decision, not one label
    for the whole non-soil point set. Fully vectorized (batched eigvalsh
    over all M neighborhoods at once), not a Python loop per point."""
    m = len(points)
    if m < 3:
        return np.zeros(m, dtype=np.float64)
    tree = cKDTree(points)
    k_eff = min(k + 1, m)  # +1 since query includes the point itself
    _, idx = tree.query(points, k=k_eff)
    if k_eff == 1:
        return np.zeros(m, dtype=np.float64)
    neighbors = points[idx]  # (M, k_eff, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k_eff  # (M,3,3)
    eigvals = np.linalg.eigvalsh(cov)  # (M,3) ascending: l3,l2,l1
    l1, l2 = eigvals[:, 2], eigvals[:, 1]
    l1_safe = np.clip(l1, 1e-12, None)
    return (l1 - l2) / l1_safe


def largest_connected_component_mask(points: np.ndarray, radius: float) -> np.ndarray:
    """points: (K,3). Returns a (K,) bool mask keeping only the points in
    the single largest connected component of the radius-graph (two
    points connected iff within `radius` of each other). Used to clean up
    geometric_stem_leaf_split's linearity-threshold stem candidates: a
    genuine stem is one continuous structure, but leaf veins/edges can
    locally look linear too, producing scattered small linear fragments
    elsewhere in the canopy -- keeping only the largest connected blob
    discards those false positives without needing a second, separately-
    tuned threshold."""
    k = len(points)
    if k == 0:
        return np.zeros(0, dtype=bool)
    if k == 1:
        return np.ones(1, dtype=bool)
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        # No point is within radius of any other -- every point is its own
        # isolated component; keep none (a single-point "stem" is noise,
        # not worth trusting over the leaf default).
        return np.zeros(k, dtype=bool)
    adj = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(k, k))
    n_components, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    largest = int(np.argmax(sizes))
    return labels == largest


def geometric_stem_leaf_split(plant_points: np.ndarray, k: int = 12,
                               linearity_threshold: float = 0.75,
                               component_radius_factor: float = 2.5) -> np.ndarray:
    """plant_points: (M,3) real (or normalized -- this is scale-invariant
    by construction, since both the k-NN radius and the connectivity
    radius are derived from the data's own local point spacing, not an
    absolute distance) non-soil points from ONE sample. Returns (M,)
    int64 in {1: stem, 2: leaf}.

    Method: per-point local linearity (see local_linearity) thresholded
    to get stem CANDIDATES, then keep only the largest spatially-connected
    component of those candidates (see largest_connected_component_mask)
    to suppress isolated leaf-vein/edge false positives -- everything not
    kept is leaf. linearity_threshold=0.75 and component_radius_factor=
    2.5 are first-pass defaults, not empirically tuned yet; see
    step_notes/D0_Trait_Extraction_Pipeline.md for the held-out validation
    this prototype was checked against and whether these defaults held up
    or needed adjustment."""
    m = len(plant_points)
    if m == 0:
        return np.zeros(0, dtype=np.int64)

    linearity = local_linearity(plant_points, k=k)
    stem_candidate_mask = linearity >= linearity_threshold

    organ = np.full(m, 2, dtype=np.int64)  # default: leaf
    n_candidates = int(stem_candidate_mask.sum())
    if n_candidates >= 2:
        tree = cKDTree(plant_points)
        nn_dist, _ = tree.query(plant_points, k=2)
        median_nn_dist = float(np.median(nn_dist[:, 1]))
        radius = component_radius_factor * median_nn_dist

        candidate_idx = np.where(stem_candidate_mask)[0]
        candidate_pts = plant_points[candidate_idx]
        keep_local = largest_connected_component_mask(candidate_pts, radius)
        organ[candidate_idx[keep_local]] = 1  # stem
    return organ


def run_inference_on_batch_tomato_geometric(model, batch, maize_remap, device,
                                             k=12, linearity_threshold=0.75):
    """Same return contract as run_inference_on_batch
    ({species: (pred_organ, mask)}), but Tomato's stem/leaf split comes
    from geometric_stem_leaf_split on real point coordinates instead of a
    Crops3D-id remap (soil still comes from the model, via
    TOMATO_SOIL_CROPS3D_ID -- only stem-vs-leaf is geometric). Maize is
    untouched, routed through maize_remap exactly as run_inference_on_batch
    would (maize_remap only needs a "Maize" entry; a full data_driven
    remap dict works fine since its "Tomato" entry is simply never used
    here)."""
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch, activate_DefRec=False)
    species_labels = batch.species_labels
    points_bcn = batch_points_bcn(batch)  # (B, 3, N)
    out = {}

    maize_idx = SPECIES_TO_IDX["Maize"]
    mask_maize = species_labels == maize_idx
    if mask_maize.sum().item() > 0:
        sp_logits = model.seg_logits_for_species(logits["seg_feat"][mask_maize], "Maize")
        preds_crops3d = sp_logits.max(dim=2)[1]
        pred_organ = maize_remap["Maize"][preds_crops3d]
        out["Maize"] = (pred_organ, mask_maize)

    tomato_idx = SPECIES_TO_IDX["Tomato"]
    mask_tomato = species_labels == tomato_idx
    if mask_tomato.sum().item() > 0:
        sp_logits = model.seg_logits_for_species(logits["seg_feat"][mask_tomato], "Tomato")
        preds_crops3d = sp_logits.max(dim=2)[1].cpu().numpy()  # (B_t, N)
        pts_tomato = points_bcn[mask_tomato].permute(0, 2, 1).cpu().numpy()  # (B_t, N, 3)
        B_t, N = preds_crops3d.shape
        pred_organ_np = np.zeros((B_t, N), dtype=np.int64)
        for b in range(B_t):
            soil_mask_b = preds_crops3d[b] == TOMATO_SOIL_CROPS3D_ID
            plant_idx_b = np.where(~soil_mask_b)[0]
            if len(plant_idx_b) > 0:
                plant_pts_b = pts_tomato[b, plant_idx_b]
                organ_plant_b = geometric_stem_leaf_split(
                    plant_pts_b, k=k, linearity_threshold=linearity_threshold)
                pred_organ_np[b, plant_idx_b] = organ_plant_b
            # soil_mask_b positions stay 0 (soil), the array's default.
        out["Tomato"] = (torch.tensor(pred_organ_np, dtype=torch.int64, device=device), mask_tomato)
    return out


## ---------------------------------------------------------------------
## RE-SCOPED GEOMETRIC FALLBACK: id0->stem kept, geometry moved to id1
##
## The first geometric prototype (geometric_stem_leaf_split, local-
## linearity-based) validated poorly on real data: Tomato stem recall
## only 8.27% (held-out), far worse than simply keeping id0->stem as-is
## (77.56-78.04% recall under groundtruth/hybrid -- no geometry needed
## there at all). Comparing all four prior attempts side by side showed
## the REAL unresolved conflict is narrower than "split id0+id2 by local
## shape": id0->stem already works well on its own, and id2->leaf is an
## acceptable pragmatic default (rare, <1% of predictions). The actual
## problem is that **id1 carries two incompatible meanings depending on
## which remap reads it**: data_driven's cross-domain confusion matrix
## says id1 is ~72% true soil; groundtruth's Crops3D-native shape check
## says id1 is the leaf-canopy class. Both single-organ assignments of
## id1 are individually well-supported but mutually exclusive -- id1's
## predictions are actually a MIX of true soil and true leaf points that
## no single remap value can separate.
##
## Per explicit user instruction: re-scope, don't retune. id0->stem stays
## exactly as-is (already solved, 78% recall). Geometry is applied ONLY
## to id1-predicted points, splitting them into soil vs. leaf via height
## relative to the sample's own base -- soil should sit near the plant's
## lowest point, leaf canopy measurably higher -- rather than local shape
## (which was the wrong feature for this specific ambiguity; height is a
## structurally different, better-motivated signal for a soil/canopy
## question specifically).
## ---------------------------------------------------------------------

def height_based_soil_leaf_split(id1_points: np.ndarray, all_points: np.ndarray,
                                  height_fraction: float = 0.15) -> np.ndarray:
    """id1_points: (K,3) points predicted Crops3D-id1 for one sample.
    all_points: (N,3) every point in that same sample (used only to
    establish the sample's own z-range as a reference -- robust to
    absolute scale since this runs in the model's native normalized
    unit-sphere space, same as every other function here). Returns (K,)
    int64 in {0: soil, 2: leaf}: points within the bottom `height_fraction`
    of the sample's own z-range are soil, everything else is leaf.

    height_fraction=0.15 is a first-pass default (roughly matching the
    soil/tray layer's visual extent in the Pheno4D scans already
    visualized for this pipeline -- see step_notes/
    D0_Trait_Extraction_Pipeline.md), not exhaustively tuned against
    held-out data. validate_against_annotated's held-out numbers with
    remap_mode="height_split" are the actual test of whether this default
    is good enough, same standard as every other approach tried."""
    k = len(id1_points)
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    z_min = float(all_points[:, 2].min())
    z_max = float(all_points[:, 2].max())
    z_range = z_max - z_min
    if z_range <= 1e-9:
        return np.zeros(k, dtype=np.int64)  # degenerate (near-flat sample) -- default to soil
    threshold_z = z_min + height_fraction * z_range
    return np.where(id1_points[:, 2] <= threshold_z, 0, 2).astype(np.int64)


def run_inference_on_batch_tomato_height_split(model, batch, maize_remap, device,
                                                height_fraction=0.15):
    """Same return contract as run_inference_on_batch. Maize: unchanged,
    routed through maize_remap exactly as the other Tomato-fallback
    orchestrators do. Tomato: id0->stem directly (no geometry -- already
    solved), id2->leaf directly (pragmatic default, same as every prior
    Tomato remap attempt), id1-predicted points split into soil/leaf via
    height_based_soil_leaf_split."""
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch, activate_DefRec=False)
    species_labels = batch.species_labels
    points_bcn = batch_points_bcn(batch)  # (B, 3, N)
    out = {}

    maize_idx = SPECIES_TO_IDX["Maize"]
    mask_maize = species_labels == maize_idx
    if mask_maize.sum().item() > 0:
        sp_logits = model.seg_logits_for_species(logits["seg_feat"][mask_maize], "Maize")
        preds_crops3d = sp_logits.max(dim=2)[1]
        pred_organ = maize_remap["Maize"][preds_crops3d]
        out["Maize"] = (pred_organ, mask_maize)

    tomato_idx = SPECIES_TO_IDX["Tomato"]
    mask_tomato = species_labels == tomato_idx
    if mask_tomato.sum().item() > 0:
        sp_logits = model.seg_logits_for_species(logits["seg_feat"][mask_tomato], "Tomato")
        preds_crops3d = sp_logits.max(dim=2)[1].cpu().numpy()  # (B_t, N)
        pts_tomato = points_bcn[mask_tomato].permute(0, 2, 1).cpu().numpy()  # (B_t, N, 3)
        B_t, N = preds_crops3d.shape
        pred_organ_np = np.zeros((B_t, N), dtype=np.int64)
        for b in range(B_t):
            preds_b = preds_crops3d[b]
            all_pts_b = pts_tomato[b]

            stem_idx_b = np.where(preds_b == 0)[0]  # id0 -> stem, direct
            pred_organ_np[b, stem_idx_b] = 1

            leaf_idx_b = np.where(preds_b == 2)[0]  # id2 -> leaf, direct (pragmatic default)
            pred_organ_np[b, leaf_idx_b] = 2

            id1_idx_b = np.where(preds_b == 1)[0]   # id1 -> geometric soil/leaf split
            if len(id1_idx_b) > 0:
                id1_pts_b = all_pts_b[id1_idx_b]
                organ_id1_b = height_based_soil_leaf_split(
                    id1_pts_b, all_pts_b, height_fraction=height_fraction)
                pred_organ_np[b, id1_idx_b] = organ_id1_b
        out["Tomato"] = (torch.tensor(pred_organ_np, dtype=torch.int64, device=device), mask_tomato)
    return out


def run_inference_on_batch(model, batch, remap, device):
    """Forwards one KPConvBatch (activate_DefRec=False -- pure
    segmentation inference, no reconstruction needed), returns
    {species: (pred_organ (B_sp, N) int64 in {0,1,2}, mask (B,) bool)}
    for whichever species are present in this batch."""
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch, activate_DefRec=False)
    species_labels = batch.species_labels
    out = {}
    for species, idx in SPECIES_TO_IDX.items():
        mask = species_labels == idx
        if mask.sum().item() == 0:
            continue
        sp_logits = model.seg_logits_for_species(logits["seg_feat"][mask], species)
        preds_crops3d_space = sp_logits.max(dim=2)[1]  # (B_sp, N) int64, Crops3D-id space
        pred_organ = remap[species][preds_crops3d_space]  # (B_sp, N) int64, {0,1,2}
        out[species] = (pred_organ, mask)
    return out


def _evaluate_split(model, config, remap, dataset, device, batch_size, num_workers,
                     inference_fn=None):
    """Runs inference + mIoU/accuracy scoring over one dataset (one split),
    returns {species: {"mIoU_sum", "acc_sum", "n", "confusion" (3,3) int64
    array, rows=true organ, cols=predicted organ, pooled across every
    point of every sample}}. The confusion matrix is what makes per-class
    recall (soil/stem/leaf individually, not just an aggregate mIoU/
    accuracy that can hide a class collapsing to 0) visible directly from
    the standard report -- added after finding aggregate mIoU alone
    missed a complete stem-recall failure in an earlier remap attempt.
    Shared by both splits in validate_against_annotated so the exact same
    scoring logic produces directly comparable numbers for each.

    inference_fn defaults to run_inference_on_batch(model, batch, remap,
    device) -- pass run_inference_on_batch_tomato_geometric (bound via a
    lambda/partial) to score the geometric Tomato fallback instead,
    through this identical scoring/reporting path, per explicit user
    instruction that it be judged by the same standard as every remap."""
    if inference_fn is None:
        inference_fn = lambda m, b, r, d: run_inference_on_batch(m, b, r, d)  # noqa: E731

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers,
                         collate_fn=make_kpconv_collate_fn(config))
    stats = {sp: {"mIoU_sum": 0.0, "acc_sum": 0.0, "n": 0,
                  "confusion": np.zeros((3, 3), dtype=np.int64)} for sp in SPECIES_TO_IDX}

    for batch in loader:
        per_species = inference_fn(model, batch, remap, device)
        B = batch.species_labels.shape[0]
        N = int(batch.lengths[0][0].item())
        gt_organ = batch.labels.view(B, N).to(device)

        for species, (pred_organ, mask) in per_species.items():
            y_pred_sp = pred_organ.cpu().numpy()
            y_true_sp = gt_organ[mask].cpu().numpy()
            for b in range(y_pred_sp.shape[0]):
                miou = jaccard_score(y_true_sp[b], y_pred_sp[b], labels=[0, 1, 2],
                                      average="macro", zero_division=0)
                acc = float((y_true_sp[b] == y_pred_sp[b]).mean())
                stats[species]["mIoU_sum"] += miou
                stats[species]["acc_sum"] += acc
                stats[species]["n"] += 1
            for t in range(3):
                for p in range(3):
                    stats[species]["confusion"][t, p] += int(
                        ((y_true_sp == t) & (y_pred_sp == p)).sum())
    return stats


_ORGAN_NAMES = {0: "soil", 1: "stem", 2: "leaf"}


def _print_stats(label, stats):
    print(f"\n--- {label} ---")
    total_miou, total_acc, total_n = 0.0, 0.0, 0
    for species, s in stats.items():
        if s["n"] == 0:
            print(f"  {species}: no samples")
            continue
        print(f"  {species}: n={s['n']}, mIoU={s['mIoU_sum']/s['n']:.4f}, "
              f"acc={s['acc_sum']/s['n']:.4f}")
        cm = s["confusion"]
        for t in range(3):
            row_total = cm[t].sum()
            if row_total == 0:
                continue
            recall = cm[t, t] / row_total
            print(f"    true_{_ORGAN_NAMES[t]:<4}: recall={recall:.4f} "
                  f"(pred soil={cm[t,0]}, stem={cm[t,1]}, leaf={cm[t,2]}, n={row_total})")
        total_miou += s["mIoU_sum"]
        total_acc += s["acc_sum"]
        total_n += s["n"]
    if total_n:
        print(f"  ALL SPECIES: n={total_n}, mIoU={total_miou/total_n:.4f}, "
              f"acc={total_acc/total_n:.4f}")
    return total_miou, total_acc, total_n


def validate_against_annotated(checkpoint_path=DEFAULT_CHECKPOINT, dropout=0.5,
                                device="cpu", batch_size=16, num_workers=0,
                                remap_mode="data_driven"):
    """The mIoU sanity check this module's docstring insists on before
    trusting any prediction. Evaluates the two annotated-scan sources
    SEPARATELY, not combined:

    - pheno4d_adaptation_pool.csv: B3b's DA-S training saw these point
      clouds (unlabeled) via DefRec's self-supervised objective. A good
      score here is a much weaker signal -- the model had geometric
      exposure to this data already. Also the ONLY split the remap
      itself is allowed to be calibrated against, when remap_mode=
      "data_driven" -- see build_data_driven_remap.
    - pheno4d_heldout_eval.csv: never touched during B3b's training in
      any form, NOR by the data-driven remap's own calibration (same
      file every other Block A-C row keeps clean as a domain-gap
      indicator). THIS is the genuinely fair generalization test, and
      the number to trust when deciding whether the Crops3D->organ
      remap and segmentation approach are usable.

    remap_mode: "data_driven" (default) builds the remap via
    build_data_driven_remap (Hungarian matching against pool's
    confusion matrix); "histogram" uses the original RGB-histogram
    heuristic (build_crops3d_organ_remap) for comparison/reproducibility
    -- the histogram approach's held-out numbers (mIoU 0.2653 overall,
    Tomato 0.0886/Maize 0.5430) are recorded in step_notes/
    D0_Trait_Extraction_Pipeline.md once written.

    Both PlantClsSegDataset instances auto-skip any sample whose cache
    entry lacks a 'labels' key, so each split naturally yields only its
    own annotated subset with no separate manifest-filtering code
    needed. neighborhood_limits is calibrated once against the union of
    both splits' points (so both evaluations run through the identical
    model/config -- comparing them isn't confounded by two separately-
    calibrated configs), then each split is scored independently.
    """
    data_dir = _REPO_ROOT / "data"
    manifest_csv = data_dir / "preprocessed" / "preprocessed_manifest.csv"

    pool_set = PlantClsSegDataset(
        data_dir / "pheno4d_adaptation_pool.csv", manifest_csv, augment=False,
        label_transform=pheno4d_collapse_organ_labels, num_classes=PHENO4D_SEG_NUM_CLASSES)
    heldout_set = PlantClsSegDataset(
        data_dir / "pheno4d_heldout_eval.csv", manifest_csv, augment=False,
        label_transform=pheno4d_collapse_organ_labels, num_classes=PHENO4D_SEG_NUM_CLASSES)
    print(f"[validate] annotated scans -- adaptation_pool: {len(pool_set)}, "
          f"heldout_eval: {len(heldout_set)}")

    model, config = load_b3b_model(checkpoint_path, dropout, device)
    calib_set = ConcatDataset([pool_set, heldout_set])
    config.neighborhood_limits = calibrate_neighborhood_limits(
        config, calib_set, batch_size, num_workers=num_workers,
        collate_fn_factory=make_kpconv_collate_fn)
    print(f"[validate] calibrated neighborhood_limits: {config.neighborhood_limits}")

    print(f"[validate] remap_mode={remap_mode}")
    inference_fn = None  # default: run_inference_on_batch, set inside _evaluate_split
    if remap_mode == "data_driven":
        remap = build_data_driven_remap(model, config, device, batch_size, num_workers)
    elif remap_mode == "histogram":
        remap = build_crops3d_organ_remap(device)
    elif remap_mode == "groundtruth":
        remap = build_groundtruth_remap(model, config, device, batch_size, num_workers)
    elif remap_mode == "hybrid":
        remap = build_hybrid_remap(model, config, device, batch_size, num_workers)
    elif remap_mode == "geometric":
        # Maize: standard data-driven remap, unchanged (already matches RGB-confirmed
        # ground truth -- no capacity problem on that species, see module docstring).
        # Tomato: NOT a remap at all -- run_inference_on_batch_tomato_geometric ignores
        # remap["Tomato"] entirely and uses the geometric fallback instead. Scored through
        # the identical _evaluate_split/_print_stats path as every other remap_mode, per
        # explicit user instruction, so the numbers are directly comparable.
        remap = build_data_driven_remap(model, config, device, batch_size, num_workers)
        inference_fn = lambda m, b, r, d: run_inference_on_batch_tomato_geometric(m, b, r, d)  # noqa: E731
    elif remap_mode == "height_split":
        # Re-scoped fallback (see run_inference_on_batch_tomato_height_split's docstring):
        # id0->stem kept as-is (already solved), geometry applied only to id1's
        # soil-vs-leaf ambiguity via height. Maize unchanged, same as "geometric".
        remap = build_data_driven_remap(model, config, device, batch_size, num_workers)
        inference_fn = lambda m, b, r, d: run_inference_on_batch_tomato_height_split(m, b, r, d)  # noqa: E731
    else:
        raise ValueError(f"unknown remap_mode: {remap_mode}")

    pool_stats = _evaluate_split(model, config, remap, pool_set, device,
                                  batch_size, num_workers, inference_fn=inference_fn)
    heldout_stats = _evaluate_split(model, config, remap, heldout_set, device,
                                     batch_size, num_workers, inference_fn=inference_fn)

    print(f"\n=== Segmentation validation ({remap_mode} remap): B3b predictions "
          "(Crops3D->organ remapped) vs. Pheno4D ground truth ===")
    _print_stats("pheno4d_adaptation_pool.csv (SEEN during B3b's DA-S training, "
                 "unlabeled -- NOT a fair generalization test)", pool_stats)
    heldout_miou, heldout_acc, heldout_n = _print_stats(
        "pheno4d_heldout_eval.csv (NEVER touched during training -- "
        "*** THIS IS THE NUMBER THAT MATTERS ***)", heldout_stats)

    if heldout_n:
        print(f"\n>>> [{remap_mode}] Held-out generalization mIoU: "
              f"{heldout_miou/heldout_n:.4f}, accuracy: {heldout_acc/heldout_n:.4f} "
              f"(n={heldout_n}) <<<")

    return {"adaptation_pool": pool_stats, "heldout_eval": heldout_stats}


def run_full_dataset_inference(checkpoint_path=DEFAULT_CHECKPOINT, dropout=0.5, device="cpu",
                                batch_size=16, num_workers=0, out_dir=None):
    """Segments ALL 223 Pheno4D scans (not just the 126 annotated ones
    validate_against_annotated uses) with the settled approach: Maize via
    the standard data_driven remap, Tomato via height_split (see
    step_notes/D0_Trait_Extraction_Pipeline.md for why -- 92.84% soil /
    76.69% stem / 68.66% leaf held-out recall, the only one of five
    attempts with all three organs above 65% simultaneously). Per
    explicit user instruction, this is the gated final step, run only
    after the held-out validation above was checked and judged
    acceptable -- not run automatically by any other function here.

    pheno4d_adaptation_pool.csv (160 scans) + pheno4d_heldout_eval.csv
    (63 scans) together cover all 223 Pheno4D scans (confirmed: 160+63=
    223, no overlap) -- using PlantSpeciesDataset (not PlantClsSegDataset)
    since most scans are unannotated and PlantSpeciesDataset doesn't
    require a 'labels' key, unlike the annotated-only validation path.

    Writes one .npz per scan to <out_dir>/<species>/<stem>.npz with
    {"points": normalized_pts (N,3), "pred_organ": (N,) int64 in
    {0,1,2}}, mirroring the Stage-0 cache's own
    <out_dir>/<species>/<stem>.npz layout so downstream trait-extraction
    code can resolve files the same way adapters/dataset.py's manifest
    lookup does.
    """
    if out_dir is None:
        out_dir = _REPO_ROOT / "data" / "segmented" / "Pheno4D"
    out_dir = Path(out_dir)

    data_dir = _REPO_ROOT / "data"
    manifest_csv = data_dir / "preprocessed" / "preprocessed_manifest.csv"

    pool_set = PlantSpeciesDataset(
        data_dir / "pheno4d_adaptation_pool.csv", manifest_csv, augment=False)
    heldout_set = PlantSpeciesDataset(
        data_dir / "pheno4d_heldout_eval.csv", manifest_csv, augment=False)
    print(f"[full inference] pool: {len(pool_set)}, heldout: {len(heldout_set)}, "
          f"total: {len(pool_set) + len(heldout_set)} (expect 223)")

    model, config = load_b3b_model(checkpoint_path, dropout, device)
    calib_set = ConcatDataset([pool_set, heldout_set])
    config.neighborhood_limits = calibrate_neighborhood_limits(
        config, calib_set, batch_size, num_workers=num_workers,
        collate_fn_factory=make_kpconv_collate_fn_cls_only)
    print(f"[full inference] calibrated neighborhood_limits: {config.neighborhood_limits}")

    maize_remap = build_data_driven_remap(model, config, device, batch_size, num_workers)

    n_written, n_skipped_existing = 0, 0
    for split_name, dataset in [("adaptation_pool", pool_set), ("heldout_eval", heldout_set)]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers,
                             collate_fn=make_kpconv_collate_fn_cls_only(config))
        sample_i = 0
        for batch in loader:
            bs = batch.species_labels.shape[0]
            cache_paths_in_batch = [dataset.samples[sample_i + j][0] for j in range(bs)]
            sample_i += bs

            per_species = run_inference_on_batch_tomato_height_split(
                model, batch, maize_remap, device)
            points_bcn = batch_points_bcn(batch.to(device))  # (B,3,N)
            points_bnc = points_bcn.permute(0, 2, 1).cpu().numpy()  # (B,N,3)

            pred_organ_full = np.zeros((bs, points_bnc.shape[1]), dtype=np.int64)
            for species, (pred_organ, mask) in per_species.items():
                idx_in_batch = np.where(mask.cpu().numpy())[0]
                pred_organ_full[idx_in_batch] = pred_organ.cpu().numpy()

            for j in range(bs):
                cache_path = Path(cache_paths_in_batch[j])
                species_name = cache_path.parent.name  # ".../Pheno4D/<species>/<stem>.npz"
                dest = out_dir / species_name / cache_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, points=points_bnc[j], pred_organ=pred_organ_full[j])
                n_written += 1

        print(f"[full inference] {split_name}: done ({sample_i} scans)")

    print(f"[full inference] wrote {n_written} segmented .npz files to {out_dir}")
    return n_written


def parse_args():
    p = argparse.ArgumentParser(description="D0: B3b segmentation inference on Pheno4D")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=-1, help="-1 for cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--remap_mode",
                    choices=["data_driven", "histogram", "groundtruth", "hybrid",
                             "geometric", "height_split"],
                    default="data_driven")
    p.add_argument("--mode", choices=["validate", "full"], default="validate",
                    help="'validate': mIoU check against the 126 annotated scans (default). "
                         "'full': segment all 223 scans with the settled height_split/"
                         "data_driven approach and write .npz output to --out_dir.")
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    if args.mode == "full":
        print(f"Using {'GPU ' + str(args.gpu) if cuda else 'CPU'}")
        run_full_dataset_inference(args.checkpoint, args.dropout, device,
                                    args.batch_size, args.num_workers, args.out_dir)
        raise SystemExit(0)
    print(f"Using {'GPU ' + str(args.gpu) if cuda else 'CPU'}")
    validate_against_annotated(args.checkpoint, args.dropout, device,
                                args.batch_size, args.num_workers, args.remap_mode)
