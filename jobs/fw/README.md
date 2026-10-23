# Full workflow rerun (FW) on the soil-kept preprocessing (P4a): job set

Status (2026-10-21): **rounds 1 to 4 done** (all seeds OK; C1 = reused `PP_P4a`). Results:
`step_notes/FW_1_Results.md` (PointNet++, DGCNN) and `step_notes/FW_2_Results.md` (KPConv).
**Block D was run as job 359157** (3-task array, seeds 1, 2, 3, all finished; elapsed 32:48,
19:14 and 36:42 per sacct); results in `step_notes/FW_3_Results.md`.
Every A/B/C row is retrained with seeds 1, 2, 3 on
`data/preprocessed_variants/P4a/preprocessed_manifest.csv` (soil kept; best baseline in Block PP,
see `step_notes/PP_2_Results.md` and `PP_3_Results.md`). Results go to `results/FW_<row>_s<seed>/`.
No existing result folder starts with `FW_`, so there is no clash. For Block D see "Block D: what
was run" below.

All six split CSVs the rows use (Crops3D train/val, Pheno4D held-out, adaptation pool, oracle
train/val) resolve 100% against the P4a manifest (checked 2026-10-14: 263, 45, 63, 160, 72, 18 rows).

## Files

- `fw_common.sh`: shared body, sourced by every job file. Per seed it skips if `run.log` already has
  the FINAL target line, moves an unfinished `run.log` aside, and continues to the next seed on
  failure (same rules as `jobs/pp_compare.sbatch`).
- `fw_<row>.sbatch`, 18 files: one job per row with seeds 1, 2, 3 run one after another. The
  exception is `fw_B3b.sbatch`, a 3-task array (one seed per task), because one seed takes about 6 h.
  Arguments are copied from each row's original job file; only `--manifest`, `--seed` and the
  result name differ.
- Dry run (prints each command, trains nothing): `FW_DRY_RUN=1 bash jobs/fw/fw_A1.sbatch`.
- Logs: `results/logs/fw_<row>_<jobid>.log` (B3b: `fw_B3b_<jobid>_<task>.log`).

## The 24 rows

Time per seed = wall-clock span of the original single-seed run (from its `run.log`). Where that
run had a GPU-contention episode, the median epoch time × 100 is used instead (marked *).
Every A/B/C row needs 1 GPU (A100); D rows ran on CPU.

| Row | Script | Model | DA | Aug | `--manifest` | Time / seed | 3 seeds | Job time limit |
|---|---|---|---|---|---|---|---|---|
| A1 | `train_a1_pointnet2_da0.py` | PointNet++ | DA-0 | ALL | added | 0.11 h | 0.3 h | 12 h |
| A2 | `train_a2_pointnet2_da_a.py` | PointNet++ | DA-A | ALL | added | 0.18 h | 0.5 h | 12 h |
| A3 | `train_a3_pointnet2_da_s.py` | PointNet++ | DA-S | ALL | added | 0.57 h | 1.7 h | 12 h |
| A4 | `train_a4_pointnet2_da_a_ln.py` | PointNet++ | DA-A | L-N | added | 0.2 h* | 0.6 h | 12 h |
| A5 | `train_a5_pointnet2_da_o.py` | PointNet++ | DA-O | ALL | added | 0.11 h | 0.3 h | 12 h |
| A6 | `train_a6_pointnet2_da_d.py` | PointNet++ | DA-D | ALL | added | 0.20 h | 0.6 h | 12 h |
| B1 | `train_b1_kpconv_da0.py` | KPConv | DA-0 | ALL | added | 2.2 h | 6.7 h | 60 h |
| B2 | `train_b2_kpconv_da_a.py` | KPConv | DA-A | ALL | added | 2.7 h | 8.1 h | 60 h |
| B3 | `train_b3_kpconv_da_d.py` | KPConv | DA-D | ALL | added | 3.5 h* | 10.5 h | 60 h |
| B3b | `train_b3b_kpconv_da_s.py` | KPConv | DA-S | ALL | added | 6.2 h | 18.7 h | 60 h per seed (array) |
| B4 | `train_b4_kpconv_da0_ld.py` | KPConv | DA-0 | L-D | added | 2.1 h | 6.4 h | 60 h |
| B5 | `train_b5_kpconv_da_o.py` | KPConv | DA-O | ALL | added | 0.7 h | 2.1 h | 60 h |
| C1 | `train_c1_dgcnn_da0.py` | DGCNN | DA-0 | ALL | already had it | 0.14 h | 0.4 h | 12 h |
| C2 | `train_c2_dgcnn_da_a.py` | DGCNN | DA-A | ALL | added | 0.25 h* | 0.8 h | 12 h |
| C3 | `train_c3_dgcnn_da_s.py` | DGCNN | DA-S | ALL | added | 0.6 h* | 1.8 h | 12 h |
| C4 | `train_c4_dgcnn_da_a_ln.py` | DGCNN | DA-A | L-N | added | 0.22 h | 0.7 h | 12 h |
| C5 | `train_c5_dgcnn_da_o.py` | DGCNN | DA-O | ALL | added | 0.39 h | 1.2 h | 12 h |
| C6 | `train_c6_dgcnn_da_d.py` | DGCNN | DA-D | ALL | added | 0.25 h | 0.8 h | 12 h |
| D1 | `train_d1_fusion_baseline.py` | MLP head on frozen B3b features | (B3b: DA-S) | – | not applicable | < 1 min (CPU) | – | job 359157: 3 seeds, all finished |
| D2 | `train_d2_fusion_growth_curves.py` | same + growth-curve params | (B3b) | – | not applicable | ~1 min (CPU) | – | job 359157: 3 seeds, all finished |
| D3 | `train_d3_fusion_temporal.py` | same + temporal info | (B3b) | – | not applicable | < 1 min (CPU) | – | job 359157: 3 seeds, all finished |
| D4 | `train_d4_fusion_previous_stage.py` | same + previous stage | (B3b) | – | not applicable | < 1 min (CPU) | – | job 359157: 3 seeds, all finished |
| D5 | `train_d5_fusion_physics_logistic.py` | D4 + Logistic ODE loss | (B3b) | – | not applicable | ~2 min (CPU) | – | job 359157: 3 seeds, all finished |
| D6 | `train_d6_fusion_physics_gompertz.py` | D4 + Gompertz ODE loss | (B3b) | – | not applicable | ~1 min (CPU) | – | job 359157: 3 seeds, all finished |

The 24 rows are the strategy table's 21 A–D rows plus the three added rows B3b, A6 and C6.
Block E (optional) is not included.

**GPU-hour estimate (A–C, 3 seeds each): about 62 GPU-h** (Block A 4.1, Block B 52.4, Block C
5.6). This assumes no contention. The cluster has shown real contention twice on KPConv rows (B1's
first attempt, and B3 at 10.2 h against 3.4 h of epoch time), so a realistic upper bound is about
2× for Block B. Block D adds minutes of CPU, plus well under 1 GPU-h of B3b inference for its
prerequisites (an estimate: segmentation and feature extraction are single forward passes).

## Submission plan (QOS `dgx`: at most 5 submitted and 4 running jobs per user; each array task counts)

Cheapest rows first. Submit a round once the previous round's jobs have finished.

| Round | Jobs | GPU-h (nominal) | Wall-clock (4 running at once) |
|---|---|---|---|
| 1 | A1, A5, A2, A4, A6 | ~2.4 | ~1 h |
| 2 | C4, C2, C6, C5, A3 | ~5.2 | ~2 h |
| 3 | C3, B5, B4, B1, B2 | ~25 | ~10 h (B2 starts when the first job ends, then runs ~8 h) |
| 4 | B3, B3b (array: 3 jobs) | ~29 | ~10.5 h (B3 longest) |
| (optional) | C1 | ~0.4 | — |

C1 is optional because it repeats `results/PP_P4a_s{1,2,3}` exactly: same script, arguments, seeds
and manifest. `PP_P4a` can be reused as FW C1, or C1 can be added as the 5th job of round 4.
Block D was run after round 4 as job 359157 (3 seeds, all finished; see "Block D: what was run" below).

## Block D: what was run (job 359157)

`fw_D.sbatch`, submitted as one 3-task array, job **359157** (one seed per task; 2026-10-21, tasks
took 32, 19 and 36 min; every step `OK` in `results/logs/fw_D_359157_<task>.log`). Task `<seed>` did:

1. **Features (GPU):** `adapters/extract_b3b_features.py --checkpoint results/FW_B3b_s<seed>/model.pt
   --manifest data/preprocessed_variants/P4a/preprocessed_manifest.csv` extracted the frozen
   256-number pooled feature of that seed's FW B3b backbone (KPConv, DA-S, ALL, trained on P4a) for
   all 223 Pheno4D scans, into `data/features_fw/P4a_B3b_s<seed>/` (log: `extract.log` there).
2. **D rows (CPU):** D1, D2, D3, D4, D5, the D5 control, D6 and the D6 control, each with
   `--features_dir data/features_fw/P4a_B3b_s<seed> --seed <seed>` and otherwise the original
   arguments. The controls are the same scripts with `--lambda_phys 0 --lambda_mono 0`. D5, D6 and
   both controls warm-start from the same seed's `results/FW_D4_s<seed>/model.pt` (`--init_from_d4`).

Results: `results/FW_D<n>_s<seed>/` (controls: `results/FW_D5_ablation_s<seed>/`,
`results/FW_D6_ablation_s<seed>/`). Summary: `step_notes/FW_3_Results.md`.

How the four earlier open decisions were settled: `extract_b3b_features.py` gained `--manifest` (it already
had `--out_dir`), and D1–D6 gained `--features_dir` (commit 5c7907d); Block D seed `<seed>` uses FW B3b
seed `<seed>`; the trait and growth **targets were kept fixed** (every D script still reads the
original `data/traits/*.csv`; `infer_segmentation.py` was not rerun), so only the input features
changed.

The original Block D files were **not overwritten**: `data/features/Pheno4D/`, `data/traits/` and
`results/D*/` are unchanged (no diff in git).

## Open flags for the rerun

- **`TOMATO_SOIL_CROPS3D_ID = 1` in `adapters/infer_segmentation.py`, now renamed
  `TOMATO_SOIL_PROXY_CROPS3D_ID`** (value and behaviour unchanged). The old name and comment
  contradicted the finding that Crops3D tomato has no soil: id 1 is a plant label (the most
  frequent Crops3D tomato id, most likely leaf; inferred from label counts), used as a soil
  *proxy* only because B3b's id-1 predictions empirically covered Pheno4D soil. Corrected
  2026-10-20: it is read **only** by `remap_mode="geometric"`
  (`run_inference_on_batch_tomato_geometric`, a validate-only experiment). `--mode full`, which
  produced the D0 segmentation, uses `height_split`, which never reads it. So it does **not**
  affect the D1–D6 trait targets (the earlier note here said it did), and it does not affect any
  A–C row.
- **Soil cue (Block PP).** P4a keeps soil and pot. `PP_3_Results.md` found that early-season clouds
  are close to species-blind under both P4 and P4a, so FW A–C target accuracy may partly reflect
  the soil cue rather than plant shape.
- **C1 = PP_P4a** (see above).
- **A4 is DA-A/L-N, not the table's DA-A/L-D**: same deviation as the original A4, kept for
  comparability with C4.
- **B3b, A6 and C6 are additions** beyond the original table, not replacements.
- **KPConv `neighborhood_limits`** are recalibrated at the start of every run. P4a clouds keep soil,
  so the limits will differ from the old runs. This is expected; the calibration exists to avoid
  B1's original out-of-memory failure.
- **GPU contention** on `dgx`: 60 h per Block B job (CLAUDE.md). B3b is split per seed for that
  reason.
- **Oracle rows (A5, B5, C5)** train on P4a Pheno4D clouds, which include pot/soil. Their
  segmentation uses the inferred Pheno4D label collapse (`pheno4d_collapse_organ_labels`, id 0 =
  soil).
- **CPU vs GPU precision.** The `--manifest` check below was run on CPU. Numbers on GPU differ
  slightly, so FW runs will not reproduce the old GPU run.logs exactly. That is not expected anyway,
  since the data changes.

## `--manifest` change and its check

17 scripts gained `--manifest` (default `None`), the same 7 lines C1 already has: one argument, plus
one `if` that replaces the manifest path only when the option is given. Without the option, the path
is the same hard-coded `data/preprocessed/preprocessed_manifest.csv`; the only visible difference is
`manifest=None` in the logged argument line.

The check was run on the login node CPU (seed 1, each row's own arguments). "Unedited" = a git
worktree of commit 7d4aada; "edited" = this commit. Logged lines were compared after removing
timestamps, the argument line (`Namespace(...)`), "Using CPU" and pure timing lines. Two kinds of
check were used:

- **1-epoch check** (A1–A6, B1–B3): unedited and edited script, no `--manifest`, 1 full epoch.
  Every logged line compared, including the first-epoch metrics and the FINAL lines.
- **Start-up check** (B3b, B4, B5, C2–C6): a full CPU epoch of these rows is too heavy for the
  login node, so each run was killed as soon as the last line before training was printed
  (data counts, class weights, KPConv calibration and batches per epoch). No run reached a training
  batch. Four runs per row: unedited; edited without `--manifest`; edited with `--manifest` set
  explicitly to the default path `data/preprocessed/preprocessed_manifest.csv` (this exercises the
  new override code); edited with `--manifest` set to P4a.

| Row | Check | Unedited vs edited (no option) | Edited: option = default path | Edited: option = P4a |
|---|---|---|---|---|
| A1 | 1 epoch | identical, all 29 lines | not run | not run |
| A2 | 1 epoch | identical, all 31 lines | not run | not run |
| A3 | 1 epoch | identical, all 31 lines | not run | not run |
| A4 | 1 epoch | identical, all 32 lines | not run | not run |
| A5 | 1 epoch | identical, all 33 lines | not run | not run |
| A6 | 1 epoch | identical, all 31 lines | not run | not run |
| B1 | 1 epoch | pre-training lines identical (7); epoch metrics differ (KPConv run-to-run noise, see below) | not run | not run |
| B2 | 1 epoch | pre-training lines identical (9); epoch metrics differ (same reason) | not run | not run |
| B3 | 1 epoch | pre-training lines identical (9); the unedited run was then killed by the system (exit 137, out of memory on the shared login node) during training | not run | not run |
| B3b | start-up | identical (9 lines) | identical to no option | starts; counts 263 / 45 / 63 |
| B4 | start-up | identical (7 lines) | identical to no option | starts; counts 263 / 45 / 63 |
| B5 | start-up | identical (8 lines) | identical to no option | starts; counts 72 / 18 / 63 / 36 |
| C1 | – | not checked here: unchanged in this commit (already had `--manifest`, used by `PP_P4a`) | – | – |
| C2 | start-up | identical (8 lines) | identical to no option | starts; counts 263 / 45 / 63 |
| C3 | start-up | identical (8 lines) | identical to no option | starts; counts 263 / 45 / 63 |
| C4 | start-up | identical (9 lines) | identical to no option | starts; counts 263 / 45 / 63 |
| C5 | start-up | identical (7 lines) | identical to no option | starts; counts 72 / 18 / 63 / 36 |
| C6 | start-up | identical (8 lines) | identical to no option | starts; counts 263 / 45 / 63 |

With the P4a manifest the data counts are unchanged and the segmentation class weights change
slightly. For example, C2 tomato label counts are [59711, 228589, 2516] by default and
[59742, 228498, 2576] with P4a. So the option does switch the cache. KPConv calibration under P4a:
B3b combined [499, 45, 32] (default [499, 45, 31]); B4 [499, 45, 31] (same); B5 [499, 36, 30]
(default [499, 37, 30]).

**Why B1/B2 epoch metrics cannot match.** KPConv runs are not repeatable even with the same seed and
unchanged code. Two runs of the *unedited* B1 script gave different first-epoch numbers (train cls
loss 0.3491 vs 0.3438, seg loss 2.2085 vs 2.1907) and different FINAL target accuracy (0.9048 vs
0.9206). The most likely cause is `batch_grid_subsampling` in `../KPConv-PyTorch/datasets/common.py`,
which by default (`random_grid_orient=True`) rotates the subsampling grid at random for every batch
element. Within a single unedited B1 run, the same checkpoint also scored differently at epoch 0
and at the FINAL evaluation (source val cls acc 0.9333 vs 0.9556). For the KPConv rows the check is therefore
limited to the pre-training lines, which match in every case.

Not covered: no edited script was run for a full epoch with `--manifest` P4a. The `C1` path that
the FW jobs use has already run on P4a in `PP_P4a_s{1,2,3}`; the other 17 rows first train on P4a
in the FW jobs themselves.
