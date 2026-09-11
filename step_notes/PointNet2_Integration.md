# PointNet++ Backbone Integration

## What this is, in plain language

This project compares three different families of neural network architecture ("backbones")
for reading 3D plant point clouds: PointNet++, KPConv, and DGCNN. Each backbone needs to be
plugged into the project's training pipeline before it can be used in any experiment. This
file covers the work to plug in the **second** of the three, PointNet++ — a well-known
architecture that processes a point cloud by repeatedly grouping nearby points and summarizing
each group, at increasing scale, to build up a description of the whole shape.

The project's reference codebase (a published research repo the project builds on, see
`CLAUDE.md` → "Reference codebase") only comes with two of the three backbones already built
in (PointNet, DGCNN). PointNet++ had to be brought in separately from a different, trusted
external source and adapted to fit the same training pipeline.

**Why it matters:** without this, none of Block A of the project's 24-row experiment plan
(all the PointNet++ rows) could be trained at all.

## What was done and how

1. **Chose and vendored a reference implementation.** Used `yanx27/Pointnet_Pointnet2_pytorch`,
   a widely-used community implementation, cloned as an unmodified sibling folder next to this
   repo (`../Pointnet_Pointnet2_pytorch`), pinned to a specific commit
   (`eb64fe0b4c24055559cea26299cb485dcb43d8dd`) so it can't silently change later. This follows
   the exact same pattern already used for the project's other external dependency,
   `DefRec_and_PCM` — treated like an installed library, never modified, never checked into
   this repo.

2. **Checked it would actually work in this project's software environment before trusting
   it.** The project's other candidate backbone, KPConv, is known to need specially-compiled
   GPU code, which is often a source of setup pain. Before assuming PointNet++ would "just
   work," inspected its core operations (farthest-point sampling, ball-query neighborhood
   grouping, feature propagation) and confirmed they're written in plain PyTorch — no special
   compiled extension needed. Verified this concretely by running those operations directly
   against a plant point cloud shaped exactly like this project's real data (4096 points), on
   CPU, under the exact PyTorch/NumPy versions this project uses (torch 2.6.0+cu124, numpy
   2.2.6) — no errors.

3. **Hit and fixed one real integration bug.** The vendored repo has a folder called `models/`
   with no package marker file, and its own example scripts import from it as
   `from models.pointnet2_utils import ...`. This name collided with this project's own file
   `adapters/models.py` — once both were on Python's import search path, a plain `models`
   import resolved to the wrong one, causing an `'models' is not a package` error. Fixed by
   adding the vendored repo's `models/` subfolder directly onto the import path and importing
   its `pointnet2_utils` submodule by its unique name instead, sidestepping the collision
   entirely (mirrors a style the vendored repo's own internal scripts already use).

4. **Built the adapter.** Wrote `adapters/models_pointnet2.py::PointNet2_ClsSeg`, which
   combines the vendored repo's building blocks (a 3-level set-abstraction encoder for the
   whole-plant classification feature, and a 3-level feature-propagation decoder for
   per-point features) with the same per-species segmentation head design already used for
   the DGCNN backbone. The result exposes the exact same output shape
   (`{"cls": ..., "seg_feat": ...}`) as the DGCNN adapter, so the rest of the training code
   doesn't need to know or care which backbone is underneath.

5. **Wrote the training script and cluster job.** `adapters/train_a1_pointnet2_da0.py` mirrors
   the already-working DGCNN training script (`train_c1_dgcnn_da0.py`) almost line-for-line —
   only the backbone is different. `jobs/a1_pointnet2_da0.sbatch` submits it to the cluster's
   GPU queue.

6. **Verified before handing off to the GPU cluster.** Ran a full 1-epoch trial on real
   Crops3D/Pheno4D data on the laptop's CPU (slow, but correctness — not speed — was the goal)
   and confirmed sane, non-zero classification and segmentation metrics with no shape or
   gradient errors, before submitting the real run to the cluster.

## Technical specifics

- **Files added:** `adapters/models_pointnet2.py`, `adapters/train_a1_pointnet2_da0.py`,
  `jobs/a1_pointnet2_da0.sbatch`
- **External dependency:** `yanx27/Pointnet_Pointnet2_pytorch`, cloned to
  `../Pointnet_Pointnet2_pytorch` (sibling of this repo), pinned at commit
  `eb64fe0b4c24055559cea26299cb485dcb43d8dd`, unmodified, imported via `POINTNET2_ROOT` env
  var + `sys.path` manipulation (see `adapters/models_pointnet2.py` docstring for the exact
  mechanism)
- **Architecture composed:** `PointNetSetAbstraction` (×3, SSG variant, mirroring
  `pointnet2_cls_ssg.py`) for the pooled global feature + `PointNetFeaturePropagation` (×3,
  mirroring `pointnet2_sem_seg.py`, not `pointnet2_part_seg_ssg.py` — that variant needs a
  ShapeNet category label input this project has no equivalent for) for full-resolution
  per-point features, combined with `PointSegDA.Models.segmentation`-style per-species heads
  (same design already used by `DGCNN_ClsSeg`)
- **Key decision — import collision workaround:** add the vendored repo's `models/` subfolder
  itself to `sys.path`, then `import pointnet2_utils` directly (unique name) instead of
  `from models.pointnet2_utils import ...` (colliding name). See
  `adapters/models_pointnet2.py` docstring for full detail.
- **Verification:** 1-epoch CPU smoke test, real Crops3D/Pheno4D data, tensor shape
  `(B, 3, 4096)`, torch 2.6.0+cu124, numpy 2.2.6 — passed, no errors, non-zero sane metrics.
- **Commit:** `d563efb` — "Integrate PointNet++ backbone for row A1 (DA-0, ALL)"

## Result

Integration complete and verified. **Pass** — no open issues. DANN/DefRec (needed for rows
A2/A3/A4) are explicitly not wired up yet; only the DA-0 (no-adaptation) path was built, since
that's what row A1 needs.

## What's next

Hand off to actual GPU training — see `step_notes/A1_PointNet2_DA0.md` for what happened when
this adapter was run for real on the cluster.

---
**2026-09-11 (afternoon):** Integration completed and committed (`d563efb`). Handed off to the
cluster to train row A1 for real (job 308286).
