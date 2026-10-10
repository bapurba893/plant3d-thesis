# Block PP, Stage 1: all six prerequisite checks

Scope: **all six boxes** of Stage 1 of `docs/PREPROCESSING_COMPARISON_PLAN.md`. Part A (below)
covers the three label-identification checks (done in an earlier pass). Part B (new) covers the
remaining three: up-axis from renders, Crops3D tomato units/scale, and the RANSAC soil cut for
unannotated Pheno4D scans.

## Part A: the three label checks

All numbers below come from command output on the raw data in `/home/apurba/plant3d-data`,
treated as read-only (every script only reads; nothing in that tree was written or moved). The
diagnostics were run from the session scratchpad rather than added to `scripts/`, so this block
has not yet changed any repo file other than this note and the plan's checklist.

Raw layout actually found (differs slightly from CLAUDE.md's description — noted, not acted on):
Crops3D PLYs are at `crops3d/<Species>/*.ply` (225 maize, 83 tomato), and Pheno4D scans are
`.txt`, not `.xyz`, at `pheno4d/<Species><NN>/<code>_<date>[_a].txt` (77 annotated tomato,
49 annotated maize).

---

## Check 1 — Crops3D maize: which `scalar_sf` id is the pot

### Method

1. Census of `scalar_sf` ids over **all 225** maize clouds (unique ids + point counts per cloud).
2. Per-id evidence on a **15-cloud sample stratified by id-set**, proportional to how common each
   id-set is in the 225-cloud census (so that id 5, which is in 158/225 clouds, is not
   under-represented the way a "first 15 files" sample would make it). Per id: point count and
   fraction, mean RGB, height along the up axis (relative to the cloud's own z range), outer
   radius, a **hollowness** statistic (what fraction of an id's own points lie inside half its
   own 99th-percentile radius — low = a ring/annulus, high = a filled disc), best-fit-plane
   normal alignment with the up axis, and planarity (smallest PCA eigenvalue share; small = flat).
3. Confirmation of the decisive statistics on **every** cloud containing both id 0 and id 5
   (156 clouds), not just the sample.
4. Colour-channel ordering of the mean RGB per id, over every cloud containing that id (all 225).

Up axis for Crops3D maize is **+z** and soil is at the low end: over the sampled clouds the
normalised position of the known soil id 0 along z is 0.007–0.175 (median ≈ 0.01), while along x
and y it sits mid-cloud (0.20–0.76); vote 12/12 for axis z. z extent reaches 2.1–2.35 m on mature
clouds, consistent with metres.

### Evidence per id (15-cloud stratified sample; z_rel is 0 = bottom, 1 = top of that cloud)

| id | clouds in sample | mean frac of cloud | mean RGB (point-weighted) | z_rel median | z_rel p05→p95 | outer radius r99 | hollowness | plane normal ∥ up | planarity |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 12 | 0.123 | (60, 40, 21) | 0.052 | 0.042→0.065 | 0.170 | 0.455 | 0.998 | 0.0058 |
| 1 | 15 | 0.079 | (73, 80, 22) | 0.374 | 0.092→0.637 | 0.031 | 0.740 | 0.046 | 0.0016 |
| 2 | 15 | 0.666 | (66, 86, 24) | 0.681 | 0.296→0.901 | 0.332 | 0.728 | 0.112 | 0.0356 |
| 3 | 3 | 0.043 | (122, 93, 60) | 0.894 | 0.843→0.975 | 0.169 | 0.538 | 0.202 | 0.0750 |
| 4 | 2 | 0.050 | (78, 72, 17) | 0.399 | 0.345→0.440 | 0.089 | 0.381 | 0.046 | 0.0191 |
| 5 | 9 | 0.235 | (48, 48, 47) | 0.070 | 0.020→0.100 | 0.145 | **0.077** | 0.840 | 0.0567 |

Census over all 225 clouds — id 0: 214/225 clouds, 2.66 M pts; id 1: 225/225, 1.27 M; id 2:
225/225, 11.25 M; id 3: 30/225, 0.16 M; id 4: 21/225, 0.22 M; id 5: 158/225, 2.86 M.
Co-occurrence: both id 0 and id 5 in 156 clouds, id 0 only in 58, id 5 only in 2, neither in 9.

Colour-channel ordering of the mean RGB, over every cloud containing the id (all 225 clouds):

| id | clouds | mean R | mean G | mean B | % clouds R>G>B (brown-like) | % clouds B≥R (grey/blue-like) | % clouds G>R (green-like) |
|---|---|---|---|---|---|---|---|
| 0 | 214 | 60.2 | 43.5 | 24.0 | **97.7%** | 0.5% | 1.4% |
| 1 | 225 | 62.6 | 75.1 | 22.0 | 15.6% | 0.0% | 84.4% |
| 2 | 225 | 64.5 | 92.3 | 27.9 | 3.1% | 0.0% | 96.9% |
| 3 | 30 | 138.4 | 111.7 | 56.3 | 100.0% | 0.0% | 0.0% |
| 4 | 21 | 82.7 | 76.0 | 17.1 | 85.7% | 0.0% | 14.3% |
| 5 | 158 | 46.4 | 49.6 | 50.4 | 12.7% | **62.0%** | 83.5% |

### The decisive test: id 5 is a hollow ring enclosing the flat id-0 disc

Run on **all 156 clouds containing both ids**:

| statistic | id 0 (known soil) | id 5 (candidate pot) |
|---|---|---|
| hollowness (own points inside half own r99) | median 0.398 (mean 0.381) | median **0.0026** (mean 0.108) |
| outer radius r99 (source units) | median 0.1140 | median 0.1352 |
| vertical span p95−p05 (source units) | median 0.0175 | median **0.1347** |
| colour saturation (max−min)/max of mean RGB | median 0.559 | median **0.160** |

Per-cloud agreement, out of 156:

- id 5 radius > id 0 radius (id 5 encloses id 0): **155/156**
- id 5 hollower than id 0: 139/156
- id 5 vertical span > id 0 vertical span: 149/156
- id 5 hollowness < 0.25 (an unambiguous ring): 126/156
- id 5 less colour-saturated than id 0: 142/156

An ASCII radius-vs-height occupancy map of the bottom 20% of two clouds makes the same thing
visible directly. In `2-102.ply`, id 0 occupies a **single height row across radii 0–14/18** (a
flat horizontal disc) while id 5 occupies **a single radius column across all 12 height rows**
(a vertical wall at fixed radius). In `2-142.ply`, id 0 is again a flat disc at the bottom and id
5 sits at the outer radius spanning several times id 0's height.

### Finding

**CONFIRMED by the data: `scalar_sf` id 5 is the pot in Crops3D maize.** It is a thin-walled,
hollow ring (median hollowness 0.0026 — essentially no points in its own interior) of radius
larger than the soil disc in 155 of 156 clouds, with ~7.7× the soil's vertical span, and a
near-achromatic mean colour (46, 50, 50; B≥R in 62% of clouds) consistent with grey/blue plastic
and clearly distinct from soil's brown (60, 44, 24; R>G>B in 97.7% of clouds). Geometry is the
primary evidence — colour alone is not sufficient on its own, because id 5's saturation varies
between clouds (0.125 in `2-142.ply`, 0.574 in `2-102.ply`, where the pot is evidently dirty or
terracotta), whereas the hollow-ring geometry holds across the whole census.

Also CONFIRMED, as a by-product of the same tables: **id 1 is the stem** (radius r99 0.031 vs the
cloud's ~0.3, i.e. on the plant axis; vertical span 0.55 of the cloud; planarity 0.0016, i.e.
a line not a surface; plane normal perpendicular to up) and **id 2 is the leaves** (66.6% of all
points, green in 96.9% of clouds, broad and mid-to-high).

INFERRED, not confirmed: **id 3 = tassel and id 4 = fruit (ear)**. Supporting evidence — id 3 sits
at the apex (z_rel median 0.894) with a bright tan colour (138, 112, 56; brown-like in 100% of its
clouds), and id 4 sits mid-stem (z_rel median 0.399) at small radius (0.089). Both appear only on
mature plants: clouds containing id 3 have median z extent 1.85 m vs 0.79 m without it, and
clouds containing id 4 have median 2.05 m vs 0.81 m without it. This is consistent with
reproductive organs, but no colour/height test distinguishes "tassel" from "fruit" by name on its
own — the assignment rests on maize morphology (tassel at the apex, ear on the stem), not on the
data alone.

**Also worth recording: the ids do NOT follow the order the paper lists the classes in.** The
paper's list is soil, pot, stem, leaves, tassel, fruit, which would imply 1 = pot and 2 = stem.
The data contradicts that directly (id 1 is a central vertical line, id 2 is 66% of the points,
id 5 is the hollow ring). Do not derive any id→organ mapping from the paper's listing order.

Consequence for the plan: **soil/pot removal for Crops3D maize = drop `scalar_sf` ∈ {0, 5}.**
Note that this removes a median ~12% (soil) + ~24% (pot) of points on clouds that have both, and
that 9/225 clouds have neither id, so the step must tolerate their absence.

One incidental observation relevant to the block's confound hypothesis (not a Stage 1 question,
recorded because it was measured): pot presence correlates with growth stage — clouds **with**
id 5 have median z extent 0.82 m, clouds **without** it 1.11 m, and the clouds carrying
reproductive organs (ids 3/4, median 1.85–2.05 m) are the tall ones. So "pot present" is not
independent of plant size within Crops3D maize.

---

## Check 2 — Pheno4D: which label id is soil

### Method

Up axis was determined **without using labels**, so that the soil identification is not circular:
for each axis and each sign, take the 15% of points nearest that extreme, fit a plane, and score
by (normal alignment with that axis) / (planarity). A real ground surface is the only thing in
these clouds that is both broad and almost perfectly flat.

Then, per annotated scan, per label column, per id: point count and fraction, mean height along
the detected up axis (both relative to the cloud's own range and in raw source units), planarity,
plane-normal alignment with up, and horizontal spread (p95 of distance from the id's own
horizontal median). Sample: 6 annotated tomato scans and 6 annotated maize scans, spread across
plants and dates; maize analysed on **both** label columns (col 4 and col 5 of the file).

### Up-axis result (label-free)

All 12 scans: up axis = **+z**. The flat sheet at the low-z end has planarity 0.00005–0.00125 and
normal alignment with z of 0.9998–1.0000 in every scan. Axis extents are x 125–167, y 99–151,
z 25–459 (source units, i.e. mm) — so z is not even the largest extent on the young tomato scans
(z 25–37 vs x/y ~135), and the flat-sheet test, not the extent, is what identifies it.

### Evidence per id group (6 scans per species; h_rel 0 = bottom, 1 = top)

| species | label col | id group | scans | total pts | mean h_rel | mean planarity | mean normal ∥ up | mean horiz spread p95 |
|---|---|---|---|---|---|---|---|---|
| Tomato | 4 | id 0 | 6 | 10,960,341 | **0.121** | **0.00126** | **1.000** | **58.09** |
| Tomato | 4 | id 1 | 6 | 250,453 | 0.549 | 0.01972 | 0.222 | 10.06 |
| Tomato | 4 | id ≥2 | 6 | 1,938,146 | 0.799 | 0.01921 | 0.839 | 9.29 |
| Maize | 4 | id 0 | 6 | 3,679,933 | **0.028** | **0.00130** | **1.000** | **57.47** |
| Maize | 4 | id 1 | 6 | 1,584,276 | 0.561 | 0.00458 | 0.233 | 35.40 |
| Maize | 4 | id ≥2 | 6 | 1,100,188 | 0.411 | 0.01141 | 0.443 | 24.09 |
| Maize | 5 | id 0 | 6 | 3,680,846 | **0.028** | **0.00130** | **1.000** | **57.47** |
| Maize | 5 | id 1 | 6 | 542,597 | 0.281 | 0.01425 | 0.103 | 16.64 |
| Maize | 5 | id ≥2 | 6 | 2,140,954 | 0.601 | 0.00683 | 0.136 | 27.97 |

Per-scan, id 0 is the lowest id in **every one of the 12 scans**, by a wide margin in raw units:
its mean z is −1.6 to −5.7 (tomato) and −4.4 to −5.2 (maize), while every other id's mean z is
positive, from +6.7 up to +217. Plane-normal alignment with z is 0.999–1.000 for id 0 in all 12
scans, and no other id in any scan exceeds 0.99.

Shape of the id-0 point set (8 scans): its p01→p99 z span is only **4.7–10.3 mm** while the
plant extends 25–459 mm, planarity 0.00054–0.00357, and hollowness 0.40–0.64 — i.e. a **filled
flat disc**, not a ring. Very few non-soil points lie below id 0's 99th-percentile height
(0–4,015 of 0.7–2.0 M points; the one exception, `T02_0321_a.txt` with 64,314, is a late scan
where the plant drapes down to the soil).

### Maize's two label columns

| file | n points | n id 0 col 4 | n id 0 col 5 | IoU of the two id-0 sets | points disagreeing on any id | frac |
|---|---|---|---|---|---|---|
| M01_0313_a | 1,554,168 | 1,322,046 | 1,321,721 | 0.99961 | 205,634 | 0.1323 |
| M02_0315_a | 1,441,265 | 913,041 | 913,285 | 0.99971 | 214,906 | 0.1491 |
| M03_0317_a | 668,618 | 465,210 | 465,701 | 0.99864 | 184,071 | 0.2753 |
| M04_0319_a | 933,060 | 420,792 | 420,576 | 0.99915 | 474,985 | 0.5091 |
| M05_0321_a | 658,827 | 258,244 | 258,434 | 0.99795 | 392,872 | 0.5963 |
| M06_0324_a | 1,108,459 | 300,600 | 301,129 | 0.99765 | 780,264 | 0.7039 |
| M07_0325_a | 1,214,597 | 69,930 | 69,994 | 0.99874 | 1,129,934 | 0.9303 |

### Finding

**CONFIRMED by the data: label id 0 is soil in Pheno4D — for tomato (the 4th column) and for
maize in BOTH label columns (4th and 5th).** It is the lowest, flattest (planarity ~0.001),
perfectly horizontal (normal ∥ z = 1.000 in 12/12 scans), broadest (horizontal spread ~57 units
vs 9–35 for plant organs) and in most scans the largest point set, and it forms a thin filled
layer only 5–10 mm thick.

**CONFIRMED, and a usefully specific result: the choice of maize label column is irrelevant for
soil removal, while mattering a great deal for everything else.** The two columns' id-0 sets agree
with IoU 0.998–0.9996, but the columns disagree on 13%–93% of all points overall — exactly what
two different leaf-segmentation conventions (the leaf-collar and leaf-tip methods) sharing one
soil label would look like.

Consistency with the existing code: `adapters/dataset.py::pheno4d_collapse_organ_labels` maps raw
0→0 ("soil"), 1→1 ("stem"), ≥2→2 ("leaf") and uses maize's **column 0 of the label block** (the
file's 4th column), with its docstring flagging both the id→organ mapping and the column choice as
unverified inferences. This check **upgrades the soil part of that to CONFIRMED and removes the
column-choice caveat as far as soil is concerned**; the stem/leaf part of that mapping was not
tested here and remains an inference. `adapters/infer_segmentation.py` treats id 0 as soil on the
*Crops3D* side, which is a separate label space — nothing here changes it, and its own documented
finding that `TOMATO_SOIL_CROPS3D_ID = 1` for Crops3D tomato is addressed by check 3 below.

Consequence for the plan: **soil removal on annotated Pheno4D scans = drop label id 0**, from
either maize column. Note the pot wall is **not** separately labelled in Pheno4D and id 0 is not
hollow, so dropping id 0 removes the soil surface but leaves whatever pot structure is in the
cloud — relevant when the RANSAC cut for unannotated scans is tuned against these labels
(remaining Stage 1 box): the labels define "soil", not "soil + pot".

---

## Check 3 — Crops3D tomato: no soil points, and is there a flat ground layer anyway?

### Method

1. Census of `scalar_sf` over **all 83** tomato clouds.
2. Per-id colour/height/geometry on 15 clouds (same statistics as check 1).
3. Label-free flat-ground test: for each of the 6 axis/sign extremes of each of the 15 clouds,
   fit a plane to the 15% of points nearest that extreme and record planarity and normal
   alignment — 90 extremes tested.
4. A calibrated density-spike test, with Crops3D maize and Pheno4D as **positive controls**: the
   maximum fraction of a cloud falling inside a slab of thickness 2% of an axis' extent, searched
   over both ends of all three axes within the outer 25% of each axis. A ground surface is a thin
   slab holding a large share of the points; this metric is what detects one without labels.

### Evidence

Census over all 83 clouds — only three ids exist: id 0 in 83/83 clouds (14.45 M pts), id 1 in
83/83 (72.27 M), id 2 in 10/83 (2.73 M). **No cloud contains any id > 2** (0 occurrences).

Per-id, over the 15-cloud sample: no id is a thin bottom layer. Every id spans most of the cloud's
height (id 0: z_rel p05 0.04–0.31 → p95 0.69–0.93; id 1: p05 0.10–0.38 → p95 0.75–0.94), none is
flat (planarity 0.034–0.239 for ids 0/1, vs 0.0058 for maize's soil and ~0.001 for Pheno4D's), and
every id is green: mean RGB has G>R and G>B in essentially every cloud (e.g. id 0 (110, 129, 92),
id 1 (116, 137, 117), id 2 (89, 115, 64)), with none showing soil's brown R>G>B ordering.

Label-free flat-ground test, 90 extremes across 15 clouds: planarity min 0.01015, median 0.03382,
max 0.20958; normal alignment min 0.232, median 0.985, max 1.000. **Extremes meeting the soil
signature (planarity < 0.005 AND alignment > 0.95): 0 of 90.** For reference, that threshold is
loose: Pheno4D soil scores 0.0005–0.0036 and Crops3D maize soil ~0.006. A deliberately *loose*
version (planarity < 0.02 AND alignment > 0.90) does catch 18 of 90, but every one of those slabs
contains **both** plant ids (85 of 90 slabs are `[0, 1]`, 5 are `[0, 1, 2]`) — none is a
single-id surface, and a 15% slab is thin relative to the cloud, so some apparent flatness is an
artefact of the slab itself rather than evidence of a plane.

Calibrated density-spike test (max fraction of cloud in a 2%-thick slab at an axis end):

| dataset | species | clouds | min | median | max |
|---|---|---|---|---|---|
| Crops3D | Maize (soil present) | 15 | 0.0665 | 0.1254 | 0.4390 |
| Crops3D | **Tomato** | 14 | **0.0240** | **0.0416** | **0.0753** |
| Pheno4D | Tomato (soil present) | 4 | 0.0839 | 0.3322 | 0.6442 |
| Pheno4D | Maize (soil present) | 4 | 0.1820 | 0.4524 | 0.5285 |

Every tomato cloud scores below every Pheno4D scan and below the maize median; the single
overlap is maize's minimum (0.0665, a cloud whose labelled soil is only 1.8% of points) against
tomato's maximum (0.0753), so this metric separates the two cleanly in practice but not with a
margin — which is why it is reported as supporting, not primary, evidence.

### Finding

**CONFIRMED by the labels: Crops3D tomato has no soil class.** The label space contains exactly
three ids across all 83 clouds, none of which has a soil signature: both ubiquitous ids span the
full height of the cloud, are non-planar (planarity 0.03–0.24), and are green rather than brown.
There is no id that could be dropped as "soil" even if one wanted to.

**CONFIRMED, separately: no unlabelled flat ground-like layer is present either.** 0 of 90
axis-extreme slabs meet the soil planarity/orientation signature that both Crops3D maize and
Pheno4D soil meet comfortably, and the density-spike metric puts all 14 tested tomato clouds
below every soil-bearing control cloud. So the absence of a soil *label* is not hiding an
unlabelled soil *surface* — there is genuinely no ground plane in these scans.

INFERRED, not confirmed (and not needed by the plan, recorded because the tables show it): the
three tomato ids look like **0 = stem, 1 = leaves, 2 = fruit** — id 1 is 70–90% of points and
reaches highest, id 0 is 13–30% and sits lower on average, and id 2 appears in only 10/83 clouds
at low height (z_rel median 0.21–0.32), consistent with fruit on a minority of scans. Note this
**contradicts the `TOMATO_SOIL_CROPS3D_ID = 1` constant in `adapters/infer_segmentation.py`**,
which assigns id 1 to "soil". That file's own docstring already explains why: the value was
derived from a model-transfer confusion matrix against *Pheno4D* labels, not from Crops3D's own
labels, and it explicitly suspects it was "picking up on generalization behavior rather than
correctly relabeling" given Crops3D tomato appears to have no soil. This check confirms that
suspicion from the Crops3D side: id 1 cannot be soil, because no tomato id is. **Not changed
here** — it is a Block A–D file and outside this task's scope — but it should not be used as a
source of truth for Crops3D tomato ids, and it is worth a decision before any later block reuses
it.

Consequence for the plan: **soil/pot removal for Crops3D tomato is a no-op**, as section 6
anticipated. This is itself part of the confound the block is testing: after soil/pot removal,
maize clouds lose a median ~36% of their points and tomato clouds lose none.

---

The remaining three Stage 1 boxes (up-axis from renders, Crops3D tomato units/scale, RANSAC soil
cut) are covered in **Part B** below, done in a later pass.

### Reproducing Part A's numbers

The diagnostics live in the session scratchpad, not in `scripts/`, because that task was scoped to
change only this note and the plan's checklist:
`scan_ids.py` (id census), `maize_id_evidence.py` (per-id stats + up-axis vote),
`maize_pot_test.py` (stratified sample + paired id0/id5 comparison),
`maize_pot_census.py` (all-156-cloud confirmation), `maize_pot_xsec.py` (ASCII cross-sections),
`pheno4d_soil.py` (label-free up axis + per-id stats), `pheno4d_soil_detail.py` (id-0 profile +
column agreement), `tomato_soil.py` (tomato census + flat-ground test), `ground_spike.py`
(calibrated density-spike test with positive controls). They should be folded into
`scripts/inspect_crops3d_sf.py` or a new numbered script if these checks need to be re-run later.

---

## Part B: up-axis, units/scale, and the RANSAC soil cut

All numbers in this part likewise come from command output on the read-only raw data (nothing in
`/home/apurba/plant3d-data` was written or moved). Diagnostics again live in the session
scratchpad except for `scripts/08_soil_removal_check.py`, which is a real, reusable repo file per
plan section 8 and the only script this part adds to the repo. Renders were saved to
`results/pp_checks/`.

### Check 4 — Up-axis for Crops3D (Pheno4D already confirmed +z in check 2)

#### Method

For **maize**, check 1 already confirmed id 0 = soil; here that is used as a direct anchor: for
every cloud with a usable soil point set, fit soil's best-fit plane via PCA and test (a) whether
its normal is closest to the z axis, (b) whether it is actually flat (planarity), and (c) whether
it sits at/below the **plant body's own z-range** (stem id 1 + leaves id 2 only — not the whole
cloud, see the note below on why).

For **tomato**, there is no soil/pot anchor (check 3), so three independent shape-based proxies
were computed on all 83 clouds instead: (1) which raw axis has the largest bounding-box extent,
(2) which raw axis the whole-cloud's top principal component is closest to, and (3) which raw
axis the inferred-stem (id 0, INFERRED not confirmed — check 3) point set's own principal axis is
closest to, plus how line-like id 0 actually is (elongation = PC1/PC2 variance ratio). Finally, 5
clouds per species per dataset (20 total) were rendered to `results/pp_checks/` for direct visual
inspection, deliberately including the tomato clouds the shape proxies most disagreed on.

**A correction made while computing the maize check**: an initial version measured soil's position
against the *whole cloud's* z-range and flagged 88/214 clouds as deviant. Investigating the worst
cases showed this was wrong — in every case inspected, the pot (id 5) legitimately extends a few
mm *below* the soil surface (physically expected: a pot's base sits below the soil it holds), so
the whole-cloud z-minimum is often the pot, not an up-axis problem. Re-measuring soil's position
against the plant body's own z-range (excluding soil/pot ids) removes this artifact; this is the
version reported below.

#### Evidence — Crops3D maize (CONFIRMED)

All 225 clouds; 214 have both a usable soil point set and a plant body (stem+leaves); 11 have
no `scalar_sf` id 0 at all (`1-106.ply, 2-122.ply, 2-124.ply, 2-127.ply, 2-134.ply, 2-136.ply,
2-140.ply, 2-148.ply, 2-149.ply, 2-22.ply, 2-35.ply`) and are excluded, not counted as deviant.

| statistic (214 usable clouds) | min | median | max |
|---|---|---|---|
| soil-plane normal \|cos\| with z | 0.2466 | 0.9981 | 1.0000 |
| soil position vs. plant body's own z-range (0 = at the plant's lowest point) | −0.229 | −0.013 | 0.207 |
| planarity (smallest PCA eigenvalue share; small = flat) | 0.00049 | 0.00492 | 0.17844 |

Soil's plane normal is best-aligned with z (over x and y) in **213/214** clouds. Applying three
deviation criteria together (z-alignment < 0.9, OR planarity > 0.05, OR soil sitting > 10% up into
the plant's own height range) flags **8/214** clouds. All 8 have below-median soil point counts
(0th–57th percentile of the 214-cloud distribution; one, `3-26.ply`, has only 84 soil points),
consistent with noisy small-sample PCA rather than a real axis problem — except one,
`3-61.ply` (z-alignment 0.247, planarity 0.178), whose soil point set is genuinely
near-isotropic (extent 0.182 × 0.199 × 0.202 m, not a flat disc).

**`3-61.ply` was rendered to check this directly** (`results/pp_checks/crops3d_maize_3-61.png`):
the plant is a completely ordinary, upright maize plant with the pot clearly visible at the
bottom and soil inside it — z is unambiguously up. The soil label in this one file is evidently
noisy or partly occluded (e.g. mostly hidden under the pot rim), not evidence of a different
coordinate convention. A normal render (`crops3d_maize_1-1.png`) shows the same thing even more
clearly: a flat brown soil disc at the very bottom of the z range.

**Finding: CONFIRMED — the Crops3D maize soil disc is a horizontal plane at the bottom along +z
in effectively all 225 clouds.** Zero clouds show an actual up-axis deviation once the one
flagged candidate is checked against its render; the flag was a soil-labelling/sampling artifact
in that single file, not a coordinate-frame issue.

#### Evidence — Crops3D tomato (NOT reliably quantifiable from shape alone; render evidence is reassuring but partial)

Shape-based proxies, all 83 clouds:

| proxy | z picked | x picked | y picked | z-rate |
|---|---|---|---|---|
| largest bounding-box extent | 43 | 35 | 5 | 51.8% |
| whole-cloud top-PCA-component axis | 55 | 23 | 5 | 66.3% |
| inferred-stem (id 0) PCA axis | 61 | 19 | 3 | 73.5% |

The three proxies **disagree with each other substantially**: bbox vs. whole-cloud-PCA agree on
only 59/83 (71.1%); stem-PCA vs. bbox agree on 60/83 (72.3%). 62/83 clouds are **near-isotropic**
(longest axis within 15% of z's own extent) — i.e. most tomato clouds simply have no strongly
dominant spatial axis for a shape heuristic to lock onto. Checking whether the same physical plant
(grouped by shared filename prefix, e.g. all `9-6_*.ply` files) agrees with itself: **11 of 12
multi-file groups show a MIXED bbox-axis across files of presumably the same plant/session** — only
one group is internally consistent.

The inferred-stem (id 0) point set is genuinely line-like in most clouds (elongation > 3 in
74/83, never "blobby" — elongation never drops below 1.5), so it is not a meaningless signal, but
its alignment confidence is weak and variable (`stem_align` 0.64–0.97, i.e. rarely a single crisp
direction, plausibly branching/multi-angle-merge artifacts) and it disagrees with z in 22/83
clouds.

**Direct rendering tells a different, more reassuring story.** 5 tomato clouds were rendered,
deliberately choosing the ones the heuristics flagged most strongly as "not z": `8-22_20-28.ply`
(bbox axis x), `9-6_35-22.ply` and `10-5_30-52.ply` (bbox axis y — `10-5_30-52.ply` is an almost
horizontal trailing vine, extent x 0.77/y 1.18/z 0.62), plus two unflagged controls
(`9-3_1-1.ply`, `8-31_30-4.ply`). **In all 5 renders, z visually points up and the plant's base
sits at the low end of z** — including the two most extreme "sideways-looking" cases by the
bbox-extent numbers. What is actually happening in those two: the *plant* sprawls/trails
horizontally (a real growth habit, visible in the render), not the *coordinate frame* — the
bbox-extent and whole-cloud-PCA heuristics are measuring plant shape, which is a poor proxy for
"which axis is up" specifically for bushy or trailing tomato plants, independent of whether the
per-file coordinate convention is actually consistent.

**This revises part A's earlier, stronger claim** ("Crops3D tomato's up-axis looks genuinely
inconsistent across clouds") **down to: the shape-based heuristics are demonstrably unreliable for
this species, but the one direct check available (rendering) found no actual up-axis
inconsistency in the 5 clouds checked, including the ones heuristics disagreed on most.** This is
NOT the same as confirming all 83 clouds are z-up — only 5/83 were visually verified, and the
heuristics cannot be used to rule out the other 78 given they are contradicted by direct
inspection in 2 of the 5 cases checked.

**Finding: INFERRED, not confirmed either way.** Evidence leans toward "z is consistently up for
Crops3D tomato, and the dataset-wide column convention probably does hold," but this is a 5-cloud
visual sample against 83, and per the task's own instruction, no fix is picked here.

#### Three options for Crops3D tomato up-axis (as requested — not decided here)

**Option A — Trust the dataset-wide +z convention, no per-cloud correction.**
Evidence for: all 5 rendered clouds (including the 2 most strongly flagged by shape heuristics)
show z pointing up; Maize and Pheno4D both use a single stable +z convention with no evidence of
per-file variation, so a priori a single SfM/export pipeline producing one convention per species
is at least as likely as one that varies file-by-file. Evidence against: only 5/83 clouds
verified; if even a handful of the 78 unchecked files are genuinely mis-oriented, nothing in this
pipeline would ever detect it (no soil/pot anchor exists for this species, unlike maize). Effect
of being wrong: for any actually-mis-oriented file, Z-axis rotation (G-R) augmentation would spin
the plant around the wrong physical axis (e.g. around its horizontal canopy spread instead of its
vertical stem), producing physically implausible augmented samples silently mixed into training;
the height-relative voxel size (plant height / 200, section 6) would be computed along the wrong
axis for that file, under- or over-estimating its true height and introducing a voxel-size
artifact correlated with whichever files are affected — precisely the kind of per-species,
per-file confound this whole block exists to eliminate, reintroduced quietly.

**Option B — Per-cloud geometric realignment (whole-cloud PCA or inferred-stem direction forced to +z).**
Evidence for: the inferred-stem (id 0) direction is a genuine, usably line-like signal in most
clouds (74/83 with elongation > 3). Evidence against, concretely demonstrated by this check's own
data: this would be **actively wrong** for exactly the plants that most need to stay as they are
— `9-6_35-22.ply` and `10-5_30-52.ply` are both confirmed-by-render to already be correctly
z-up, yet their shape-based principal axis points sideways (because the plant sprawls/trails);
forcing realignment to the shape axis would rotate an already-correct cloud onto its side. The
stem-PCA axis itself is also not reliably crisp (`stem_align` as low as 0.64). Effect of being
wrong: same two failure modes as Option A (corrupted Z-rotation augmentation, corrupted
height-relative voxel size) but inflicted on a **much larger, concretely identified set of
clouds** (at least the 22/83 where stem-PCA disagrees with z, almost certainly including some of
the already-correct sprawling/bushy ones) rather than an unknown minority — this option's own
failure mode is demonstrated, not hypothetical, in this dataset.

**Option C — Targeted manual/visual verification before deciding (expand the render check).**
Render and eye-check the clouds the heuristics most disagree on (e.g. the ~24 clouds where
bbox-axis and whole-cloud-PCA axis disagree with each other, or the 22 where stem-PCA disagrees
with z) before committing to either A or B, then hand-correct only the files a human actually
confirms are wrong (if any). Evidence for: this is the only option that doesn't risk silently
reintroducing the exact confound this block is testing for, and the 5-cloud sample so far found
zero true positives among exactly the cases this would prioritize checking — suggesting the
needed manual pass may be short. Evidence against: real, if modest, one-time labour cost (tens of
renders to eye-check, not hundreds); does not scale automatically if a future batch of tomato
clouds is added later without being re-checked.

### Check 5 — Crops3D tomato units and scale

#### Method

Bounding-box z-extent (maize, Pheno4D — both have a confirmed +z axis) or max-extent-over-any-axis
(tomato, no confirmed axis — both are reported) over every cloud in each dataset/species.

#### Evidence

| dataset/species | n | min | p25 | median | p75 | max | unit |
|---|---|---|---|---|---|---|---|
| Crops3D Maize (z-extent) | 225 | 0.208 | 0.693 | 0.831 | 0.982 | 2.392 | m (CONFIRMED — FARO TLS, stated acquisition method) |
| Crops3D Tomato (z-extent) | 83 | 0.171 | 0.304 | 0.413 | 0.595 | 1.768 | unknown |
| Crops3D Tomato (max extent, any axis) | 83 | 0.184 | 0.350 | 0.485 | 0.666 | 1.768 | unknown |
| Pheno4D Tomato (z-extent, all 140 scans) | 140 | 24.5 | 47.1 | 71.5 | 100.9 | 179.1 | mm (CONFIRMED — laser triangulation arm, stated in paper) |
| Pheno4D Maize (z-extent, all 83 scans) | 83 | 34.5 | 117.6 | 229.0 | 323.2 | 469.3 | mm (CONFIRMED) |

If Crops3D tomato's raw values are metres: median z-extent 413 mm, max 1768 mm, vs. Pheno4D
tomato's median 72 mm and max 179 mm (ratio 5.8× at the median). **This does not contradict a
metric reading**: Pheno4D is explicitly a young/early-growth-stage study (per CLAUDE.md, its own
multi-temporal design tracks early growth), so its own max of 179 mm (18 cm) is a seedling/young
plant, while a greenhouse "crop dataset" paper plausibly includes more mature specimens — mature
greenhouse tomato plants commonly reach 0.3–2 m, which comfortably covers Crops3D tomato's
0.17–1.77 m range as metres.

A second, independent check: median nearest-neighbour point spacing, raw units, on a sample of
clouds. Crops3D maize (CONFIRMED metres, FARO TLS, several-metre field range): median spacing
1.48 mm (range 0.92–2.05 mm across 10 clouds). Crops3D tomato (unit unknown, close-range Canon
SfM, greenhouse): median spacing, taken as metres, 1.67 mm (range 0.83–2.65 mm across 9 clouds)
— essentially the **same numeric range** as maize's confirmed-metric spacing (ratio 1.13×).

**This spacing comparison is INCONCLUSIVE, not confirmatory, and is reported as such.** The
reasoning cuts both ways: if both datasets were in the same real-world unit, close-range SfM
photogrammetry of a potted plant would normally be expected to produce *finer* spacing than a
multi-metre-range field TLS scan, not statistically indistinguishable spacing — so the near-equal
numbers are mildly surprising. But Crops3D's own release also ships a separate, deliberately
uniformly-downsampled `Crops3D_10k` variant alongside the base `Crops3D` release, which suggests
the base release itself may already reflect a controlled target point density across species
rather than raw native sensor resolution — which would fully explain the near-equal spacing
without needing to invoke wrong units. No way to resolve this further from the data alone was
found in this session.

**Finding: CONFIRMED — Crops3D maize is in metres (stated acquisition method, plausible plant
sizes); Pheno4D is in mm (stated in the paper, plausible seedling/young-plant sizes).**
**Crops3D tomato: INFERRED, not confirmed, and reported honestly as such.** Its raw values, read
as metres, are physically plausible for mature greenhouse tomato plants and are not contradicted
by comparison to Pheno4D (whose own range covers a different, younger growth window); the
point-spacing check is inconclusive in either direction. No independent calibration reference
(a ruler, a marker of known size, ground-control points) was available in this environment to
settle it further — this is a genuine open gap, not a confirmed answer.

### Check 6 — RANSAC soil cut for unlabeled Pheno4D scans

**SUPERSEDED by Part C below.** Two problems with this approach, both resolved in Part C: (1) a
tilted-plane RANSAC fit on the lowest 15% of *points* fails specifically on late-growing-season
scans where drooping leaves/stem dominate that point quantile; (2) it used soil LABELS on
annotated (`_a`) scans to decide the cut, which — since Pheno4D is the target dataset — would put
target labels into a preprocessing step; this was corrected to a fully label-free method applied
identically to every scan. Kept below, unedited, as the record of how the problem was diagnosed;
superseded by `scripts/08_soil_removal_check.py`'s current (label-free, horizontal-slab) version
and Part C's numbers, not by this section.

Implemented in `scripts/08_soil_removal_check.py` (pure numpy RANSAC, no sklearn in this
environment): fits a plane on the lowest 15% of a scan's points by z (plan section 6's default),
orients it using Pheno4D's confirmed +z axis, then for a grid of margins reports soil-removed%
(fraction of label-0 points at/below plane−margin) and plant-points-lost% (fraction of label≥1
points incorrectly removed), on all 126 annotated scans (77 tomato + 49 maize).

#### Overall result at the plan's literal spec (bottom 15% window)

| margin (mm) | soil removed (pooled) | plant lost (pooled) | worst-scan soil removed | worst-scan plant lost |
|---|---|---|---|---|
| 0.0 | 10.89% | 3.968% | 0.00% | 74.91% |
| 1.0 | 67.82% | 4.412% | 0.02% | 80.47% |
| 2.0 | 88.89% | 4.596% | 0.18% | 84.38% |
| 3.0 | 92.27% | 4.760% | 0.38% | 86.90% |
| 4.0 | **95.02%** | **4.929%** | 0.56% | 89.01% |
| 5.0 | 97.12% | 5.128% | 0.75% | 91.19% |
| 10.0 | 99.76% | 6.265% | 2.00% | 99.74% |

**Finding, stated plainly per instructions: the plan's starting target (>95% soil removed AND
<1% plant lost) is NOT reached at any tested margin.** Margin=4.0 mm is the closest crossing of
the 95% soil-removed line, but at that margin pooled plant-loss is 4.93% — almost 5× the target.
Full per-margin table (0–15 mm) in `results/pp_checks/pheno4d_soil_removal_summary.csv`; full
per-scan, per-margin numbers in `results/pp_checks/pheno4d_soil_removal_annotated.csv`.

#### Root cause (identified, not just reported as a shortfall)

The pooled numbers are dragged down by a **small number of badly-failing scans**, not a uniform
shortfall. At margin=4mm, the median scan already looks good (Tomato median plant-lost 0.063%,
Maize median plant-lost 0.021% — both well under the 1% target), but the worst scans lose up to
89% of plant points. The 6 worst scans by plant-points-lost are concentrated in the **last few
days of the growing season** for their plant (`M03_0325_a`, `T05_0325_a`, `T06_0324_a`,
`T06_0325_a`, `T06_0319_a`, `T02_0325_a`).

Directly checking what fraction of each scan's bottom-15% candidate window is actually
plant vs. soil confirms why: in these 6 scans the window is **56–92% plant points** (drooping
leaves/stem reaching down to near-soil height), vs. **100% soil** in every early-season control
scan checked. RANSAC is doing exactly what it is told — fitting the dominant local plane in its
candidate set — but by late in the growing season that candidate set is dominated by drooping
plant material, not soil, so it fits a plane through leaves/stem instead.

Shrinking the candidate window (diagnostic only, not a change to the plan's spec) rescues 5 of
these 6: at a 1–2% window, candidate-set soil purity returns to 100% for `M03_0325_a`,
`T06_0324_a`, `T06_0325_a`, `T06_0319_a`, and mostly for `T05_0325_a`. **One scan, `T02_0325_a`,
is a genuinely different, more fundamental failure**: its single lowest point in the *entire*
cloud (z = −8.54) is labelled stem (id 1), while soil's own minimum is z = −5.38 — the stem
physically droops/sags more than 3mm **below** the soil surface in this scan. No candidate-window
size can fix this, because the true global z-minimum is plant material, not soil — a single flat
global plane cannot cleanly separate the two for this scan by construction.

Re-running with a 2% window (`results/pp_checks/bottom_frac_0.02/`, diagnostic, via
`--bottom-frac 0.02`) improves the achievable trade-off substantially but still does not clear
the target: margin=2.5mm gives 79.4% soil removed at 1.02% plant lost (closest crossing), margin
=5.0mm gives 94.1% soil removed at 1.22% plant lost. The worst-scan max-plant-lost stays high
(55.9–97.4% across margins) because `T02_0325_a`'s fundamental failure is unaffected by window
size. **This diagnostic is reported for Stage 2 to consider; it does not change this script's
default (`--bottom-frac 0.15`, matching the plan) and was not implemented as a silent fix.**

#### Pot check

Tests whether label id 0 (soil) hides a separate pot-wall ring (as Crops3D maize's id 5 does,
check 1), and whether a soil-plane cut would leave a pot rim behind.

| species | n | soil-radius / full-cloud-radius (coverage) | hollowness (low = ring) | outer-edge height delta vs. centre |
|---|---|---|---|---|
| Tomato | 77 | median 0.884 | median 0.495 | median −0.91 mm |
| Maize | 49 | median 0.675 | median 0.444 | median −2.81 mm |

No ring signature (hollowness is well above the ~0.0–0.08 range that identified Crops3D maize's
actual pot wall in check 1) and the outer edge of the soil label is *lower*, not higher, than its
centre in both species — the opposite of what a raised pot rim would look like. Coverage below
100% (especially maize's 0.675 median) is consistent with later-growth scans where plant organs
drape outward past the soil's own footprint, not with a separate structure beyond it.

**Finding: CONFIRMED (geometrically; no independent pot label exists to check against directly)
— Pheno4D has no separately identifiable pot-rim label or structure.** Any pot material actually
present in these clouds is either not captured by the scanner, or indistinguishable from/folded
into the soil label (id 0) itself. A soil-plane cut removes it together with soil (there is no
separate rim structure for a cut to "leave behind").

#### Unannotated-scan sanity check (no ground truth — points-removed% only)

10 of 97 unannotated scans, at the margin closest to the pooled 1% plant-lost target (margin=0mm
at the 15% window, since pooled plant-lost was already 3.97% at margin 0): points-removed ranged
2.50%–54.68% (median 6.05%), with one clear outlier (`M04_0318.txt`, 54.68% removed — plausibly
the same late-season drooping failure mode found above, not separately verified since this scan
has no labels). At the 2% window, the sample instead ranged 1.04%–79.43% (median 33.04%),
reflecting the much smaller/tighter candidate set's higher per-scan sensitivity to exactly how
much of that thin slice happens to be soil in each individual scan. Full tables in
`results/pp_checks/pheno4d_soil_removal_unannotated_sanity.csv` (and the `bottom_frac_0.02/`
subfolder for the 2% variant). This is a sanity check only, as instructed — there is no ground
truth on unannotated scans to validate against.

### Reproducing Part B's numbers

`scripts/08_soil_removal_check.py` (repo file, real CLI, documented above) is the one script this
part adds to the repo — **since updated in Part C**, see that section for the current algorithm;
this entry is about Part B's now-superseded RANSAC numbers specifically. Everything else lives in
the session scratchpad:
`maize_upaxis_full2.py` (maize soil-plane census, plant-body-relative, all 225 clouds),
`tomato_upaxis_full.py` (tomato bbox/PCA axis census, all 83), `tomato_stem_axis.py` (tomato
stem-PCA axis + elongation, all 83), `render_clouds.py` (the 20 renders in `results/pp_checks/`),
`units_scale.py` (bounding-box extent tables), `point_spacing.py` (nearest-neighbour spacing
comparison), `pheno4d_pot_check.py`/`pheno4d_pot_check2.py` (early pot-geometry exploration,
superseded by `08_soil_removal_check.py`'s own `pot_rim_check()`), `pheno4d_failure_diag.py` and
`pheno4d_bottomfrac_check.py` (root-cause diagnostics for the worst-failing scans).

---

## Part C: tomato up-axis montage, a label-free soil cut, and the plan correction

Per explicit user instruction. All raw-data reads remain read-only. Two repo changes this part:
`scripts/08_soil_removal_check.py` was rewritten in place (same filename, new algorithm — the
RANSAC version is kept, clearly marked superseded, in check 6 above rather than deleted), and
`docs/PREPROCESSING_COMPARISON_PLAN.md` section 6 was corrected (see below).

### Task 1 — Tomato up-axis montage (no decision made — for the user to look at)

**File: `results/pp_checks/tomato_axis_montage.png`** (36 panels, 6×6 grid, ~3.1 MB).

Selection: the 24 Crops3D tomato clouds Part B's three shape heuristics (bbox-extent axis,
whole-cloud PCA axis, inferred-stem PCA axis) disagreed on most — ranked by number of distinct
axes named across the three proxies, then by how many named something other than z — plus 12
random clouds (seed=0) from the remaining 59, as an unbiased control set. Exact lists in
`tomato_disagree_24.json`/`tomato_random_12.json` (session scratchpad).

Each panel: a 2D side-view scatter of (whichever of x/y has the larger extent in that cloud) vs.
z, points coloured by their own RGB, a black arrow marked "+z" in the corner showing the raw
dataset's z direction, and the filename printed above. Red-bordered panels are the 24
heuristic-disagreement clouds; grey-bordered panels are the 12 random controls. **No judgement
about whether any panel looks correctly oriented is made in this note or anywhere else in this
pass** — per instruction, that call is left to the user.

### Task 2 — Label-free horizontal-slab soil cut, re-evaluated on all 126 annotated scans

#### Method

Replaces check 6's tilted-plane RANSAC entirely (that section is kept above, marked superseded,
for the record of how the problem was diagnosed). New method in `find_soil_slab()` in
`scripts/08_soil_removal_check.py`, fully label-free:

1. Restrict the search to the bottom 30% of the scan's **physical z-range** (`--slab-search-frac`,
   default 0.3) — not the bottom 30% of *points*. This is the key fix: it is a geometric fact
   about the scene (soil sits below the plant it supports) rather than a point-count quantile, so
   it cannot be swamped by a large number of drooping-leaf points the way the old "lowest 15% of
   points" criterion was.
2. Slide a 3mm-thick horizontal window (`--slab-window-mm`) through that restricted range in 1mm
   steps (`--slab-bin-width-mm`). Score each window by (point count in it) × (lateral footprint =
   number of occupied 2mm×2mm x,y grid cells in it, `--slab-cell-size-mm`, converted to mm²).
   Soil wins this score because it is uniquely both dense (a solid surface) *and* wide (spans
   close to the whole pot diameter); a drooping leaf cluster can be locally dense but is narrower,
   and any sparse low structure is wide but not dense.
3. z0 = the top edge of the best-scoring window. Cut: remove points with z ≤ z0 + margin.

Labels are used **only** to score the result (soil-removed%, plant-points-lost%) on the 126
annotated scans — never to find z0, on those scans or any other. The same detector runs
identically on annotated and unannotated scans.

#### Validation against the superseded method's own known failures

The method was prototyped and checked against the 6 scans check 6 identified as worst-failing
(`M03_0325_a`, `T05_0325_a`, `T06_0324_a`, `T06_0325_a`, `T06_0319_a`, `T02_0325_a`) before being
written into the repo script. An unrestricted whole-range version of the density×footprint search
(no bottom-30% restriction) still picked the wrong slab on 4 of these 6 — a dense, moderately wide
leaf-canopy layer well above the true soil can outscore it once the search isn't restricted
in height at all. Restricting to the bottom 30% of the physical z-range fixed all but one (see
below), confirmed on this 9-scan set and then on a broader 30-scan random sample at
`--slab-search-frac` ∈ {0.2, 0.3, 0.4} — **30/30 matched the true soil location at every tested
value**, so 0.3 was kept as the default without needing to tune it further against labels (the
value was fixed before running the full 126-scan evaluation below, not selected by it).

#### A genuine label-quality finding, not a detector failure

Investigating the one scan the new detector still disagreed with the labels on
(`T02_0325_a.txt`) found that this scan's own label 0 is not confined to a thin bottom layer at
all — it spans **77.6%** of the scan's own full z-range (points labeled 0 appear from z=−5.4 all
the way up to z≈99, with a secondary mass around z=73–84), against a **6.8% median and 28.1%
second-highest** across all other 125 annotated scans (checked directly; full distribution in
`pheno4d_failure_diag.py`/the broader per-file scan in the session scratchpad). **This is a
genuine, isolated label-quality issue in this one file, confirmed by direct inspection of the
label histogram, not a shortcoming of the geometric method**: the detector itself still finds a
physically sensible slab for this scan (z=[−6.88,−3.88], high density 11,897, large footprint
7240mm² — consistent with every other scan's real soil) — it is label 0's own claim that soil
extends up to z≈99 that is implausible, not the detector's answer.

#### Full result on all 126 annotated scans

| margin (mm) | soil removed (pooled) | plant lost (pooled) | worst-scan soil removed | worst-scan plant lost |
|---|---|---|---|---|
| 0.0 | 89.96% | 0.288% | 0.05% | 8.59% |
| 1.0 | 92.90% | 0.327% | 0.25% | 8.88% |
| 1.5 | 94.13% | 0.350% | 0.34% | 8.96% |
| **2.0** | **95.32%** | **0.377%** | 0.43% | 9.00% |
| 2.5 | 96.39% | 0.412% | 0.51% | 9.02% |
| 3.0 | 97.23% | 0.454% | 0.61% | 9.04% |
| 5.0 | 99.04% | 0.655% | 0.95% | 11.11% |
| 10.0 | 99.83% | 1.288% | 2.75% | 22.70% |

Full table (0–15mm) in `results/pp_checks/pheno4d_soil_removal_summary.csv`; full per-scan,
per-margin numbers in `results/pp_checks/pheno4d_soil_removal_annotated.csv`; the detector's
own found-slab-vs-true-soil validation in `results/pp_checks/pheno4d_slab_detector_validation.csv`
(125/126 match, the one mismatch being `T02_0325_a.txt` above).

**Finding, stated plainly: the plan's starting target (>95% soil removed AND <1% plant lost) IS
reached, at margin = 2.0mm (95.32% soil removed, 0.377% plant lost, pooled across all 126
scans).** This is a large improvement over the superseded RANSAC method, whose best pooled
result (margin=4mm) was 95.02% soil removed at 4.93% plant lost — roughly 13× worse on the
plant-loss side, and never actually crossed the <1% bar at any margin.

Excluding `T02_0325_a.txt` (the one scan with its own demonstrably anomalous label, above) makes
the result stronger still: 95.48% soil removed, **0.154%** plant lost at margin=2.0mm — included
here for completeness, not to inflate the headline number, since the reported 95.32%/0.377% above
already includes it and already clears the target.

#### Worst scans (margin = 2.0mm, by plant-points-lost)

| file | species | soil removed | plant lost | n_plant |
|---|---|---|---|---|
| T02_0325_a.txt | Tomato | 0.43% | 9.003% | 3,238,019 |
| T02_0321_a.txt | Tomato | 99.73% | 4.635% | 1,497,805 |
| M05_0313_a.txt | Maize | 99.77% | 3.117% | 81,287 |
| T05_0305_a.txt | Tomato | 97.05% | 2.810% | 128,318 |
| T02_0307_a.txt | Tomato | 99.81% | 2.802% | 99,118 |
| T02_0305_a.txt | Tomato | 99.75% | 1.466% | 48,352 |
| T05_0309_a.txt | Tomato | 98.87% | 1.440% | 293,882 |
| T05_0307_a.txt | Tomato | 97.80% | 1.327% | 286,656 |
| T07_0307_a.txt | Tomato | 99.55% | 1.036% | 177,072 |
| M02_0313_a.txt | Maize | 99.63% | 0.868% | 345,065 |

Only **9/126** scans exceed 1% plant loss at this margin (8 tomato, 1 maize); the median scan is
far better (Tomato median 0.032%, Maize median 0.015%). `T02_0325_a` is the one-off broken-label
case above. The rest of the worst-10 list is dominated by **early-season (day 0305–0313),
small-plant scans** — `n_plant` as low as 48,352–345,065, an order of magnitude below the
dataset's typical scan. This is a boundary-ambiguity effect, not a cutting error: a seedling this
small touches the soil right at ground level, so any soil cut inevitably clips a few points at
the interface, and that small ABSOLUTE number becomes a large PERCENTAGE of a small denominator.
Checked directly: log(n_plant) vs. plant_lost_frac correlation across the 125 scans (excluding
`T02_0325_a`) is −0.274 — a real but moderate tendency for smaller/younger plants to lose a
higher fraction, consistent with this explanation without being overwhelmingly driven by it alone.

#### Remaining plant loss at the target-reaching margin

At margin=2.0mm (the point the target is first reached): **0.377% of all plant points, pooled,
are lost** — concentrated in the 9 scans above (nearly all of the pooled loss), with the broad
majority of the other 117 scans losing a small fraction of a percent or nothing at all. This is
the honest residual: the method does not perfectly separate soil from plant on every single scan
(two structurally different reasons — one corrupted label, several genuine seedling-stage boundary
cases) but comfortably clears the plan's aggregate target.

#### Pot check (unchanged from Part B, re-confirmed)

Same result as Part B's check 6: Tomato coverage_frac median 0.884, hollowness median 0.495, rim
height delta median −0.91mm; Maize coverage_frac median 0.675, hollowness median 0.444, rim height
delta median −2.81mm. No ring signature, no raised outer edge — **CONFIRMED (geometrically): no
separate pot-rim label or structure exists in Pheno4D**; the geometric cut removes any pot
material together with soil, with nothing structurally distinct left behind to "leave behind."

#### Unannotated scans (now a real preview, not just a sanity check)

Since the method never used labels on annotated scans either, running it on unannotated scans is
no longer "sanity-only" — it is the actual preprocessing step. Sample of 10 of 97, at margin=7mm
(closest pooled plant-lost% to the 1% target in the table above — chosen automatically by the
script, the same way the superseded version chose its sanity-check margin): points removed ranged
1.74%–91.59% (median 38.10%). This wide range reflects real growth-stage variation (how much of
each individual plant's own cloud the soil disc makes up, which shrinks as the canopy grows) —
not instability in the detector, which `pheno4d_slab_detector_validation.csv` already showed is
accurate on 125/126 labeled scans. Full table in
`results/pp_checks/pheno4d_soil_removal_unannotated_sanity.csv`.

### Task 3 — Plan correction (`docs/PREPROCESSING_COMPARISON_PLAN.md` section 6)

Changed the Pheno4D soil-removal definition: the geometric cut (above) now applies to **every**
Pheno4D scan, annotated or not, with soil labels used only to validate it — never to compute it,
including on the `_a` scans, which the plan previously treated differently. Rationale recorded in
the plan itself: Pheno4D is the target dataset, and using its own labels inside a preprocessing
step — even restricted to the `_a` subset — would leak target supervision into a step the rest of
the pipeline treats as unsupervised. The Crops3D maize rule (drop soil id 0 and pot id 5, using
Crops3D's own source labels — fine, since Crops3D is the labeled source dataset) and the Crops3D
tomato note (nothing to remove) are unchanged, now stated as CONFIRMED per Part A/B's checks
rather than "to be identified."

### Reproducing Part C's numbers

`scripts/08_soil_removal_check.py` was rewritten in place (new `find_soil_slab()` function
replaces `fit_plane_ransac()`/`signed_height()`; CLI flags renamed from `--ransac-*`/
`--bottom-frac` to `--slab-*`). Session scratchpad: `tomato_axis_montage.py` (the montage),
`slab_prototype.py`/`slab_prototype2.py` (unrestricted vs. bottom-30%-restricted search,
validated against the 6 known-bad scans), `slab_broad_validate.py` (30-scan random validation
across `--slab-search-frac` ∈ {0.2, 0.3, 0.4}), plus the label-anomaly investigation reusing
`pheno4d_failure_diag.py`'s approach on `T02_0325_a.txt` specifically.
