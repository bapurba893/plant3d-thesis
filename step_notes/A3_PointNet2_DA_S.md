# Row A3: PointNet++, Self-Supervised Domain Adaptation (DA-S), All Augmentations

## What this is, in plain language

A3 is the DGCNN-track's C3 row, ported to the PointNet++ backbone: same self-supervised
deformation-reconstruction method (DefRec-style — deform a region of the point cloud, train the
model to reconstruct it via Chamfer distance, on BOTH source and target domains per CLAUDE.md's
DA-S definition), same augmentation mix, same data/protocol — only the backbone changes.

**Why this row, specifically, before Block B:** per explicit user instruction, C3 was DGCNN's
least-bad domain-adaptation result so far — its full-run mean target accuracy (0.703) sits closer
to C1's no-adaptation baseline (0.791, an 8.8-point gap) than either DA-A variant did (C2: 15.8
points; C4: 17.8 points — see `step_notes/C4_DGCNN_DA_A_LN.md`). A3 tests whether "DA-S is the
closest of the three methods tried so far" is a genuine, backbone-general property of the method,
or just something true of DGCNN specifically — the same kind of cross-backbone question A2 already
asked of DA-A (see `step_notes/A2_PointNet2_DA_A.md`: DA-A's underperformance direction replicated
on PointNet++ but the magnitude was much smaller, because A1's own baseline was already unstable).
Answering this before adding a third backbone (KPConv, Block B) matters more right now because the
current two-backbone picture is still incomplete on this specific question.

## Status: submitted to the cluster (2026-09-12)

## Design decisions

1. **DefRec's core machinery (`DefRec_and_PCM.DefRec.deform_input`/`.calc_loss`,
   `utils.pc_utils`) is fully backbone-agnostic** — it operates on raw point tensors and reads a
   generic `logits['DefRec']` key, never touching the backbone's internals directly. Reused
   unchanged from C3, including the chunked-Chamfer-distance OOM workaround
   (`DEFREC_CHUNK=8`/`defrec_chunked_backward`) — PointNet2 produces the same dense `(B, 4096,
   ...)` per-point tensors DGCNN does, so the same memory pressure that OOM'd an 80GB A100 on
   C3's first attempt applies here too; no reason to expect otherwise, so the fix was carried
   over proactively rather than waiting to hit the same OOM.
2. **One real gap needed filling: PointNet2_ClsSeg had no reconstruction head at all** (A1/A2
   never needed one). Unlike A2's adversarial machinery (which had to be built from scratch,
   backbone-agnostically, in `adapters/dann.py`), PointNet++'s DefRec support only needed
   composing ONE existing reference-repo class it hadn't used yet:
   `PointDA.Models.RegionReconstruction` — the exact same class `DGCNN_ClsSeg` already uses via
   its `DGCNN` base class, a generic per-point Conv1d stack over an arbitrary input feature
   width (identical in kind to `PointSegDA.Models.segmentation`, already reused for both
   backbones' seg heads). Added to `adapters/models_pointnet2.py::PointNet2_ClsSeg`:
   - `self.DefRec = RegionReconstruction(args, seg_input_size)` in `__init__` (same
     `seg_input_size=1152` already used for the seg heads).
   - `forward(..., activate_DefRec=True)` now sets `logits["DefRec"] = self.DefRec(logits["seg_feat"])`
     — reusing the already-computed `seg_feat` tensor (per-point features concatenated with the
     pooled global feature) as DefRec's input, since that's exactly DefRec's expected input shape.
     This is a minor efficiency improvement over DGCNN's own `forward`, which builds an identical
     tensor a second time from scratch for `DefRec_input` — not a behavioral difference, just
     avoiding a redundant duplicate computation this repo's composition doesn't need to repeat.
   - No reference-repo code modified — same "adapt this repo" pattern as everything else in this
     project.
3. **`adapters/train_a3_pointnet2_da_s.py` is a near-line-for-line copy of
   `train_c3_dgcnn_da_s.py`**, with `DGCNN_ClsSeg`/`model_args.model="dgcnn"` swapped for
   `PointNet2_ClsSeg`/plain `model_args` (same swap pattern A1 used for C1, A2 used for C2) — same
   data, same loss architecture (Kendall over cls/seg/defrec), same DefRec hyperparameters
   (`DefRec_dist=volume_based_voxels`, `num_regions=3`, `DefRec_weight=1.0` neutral), same
   model-selection protocol.
4. **Comparison baseline is A1** (`results/A1_pointnet2_da0_clsseg/run.log`), the same way C3
   compares against C1 and A2 compares against A1 — all four rows (A1/A2/A3, C1) share the
   identical `pheno4d_heldout_eval.csv` target test set/protocol.
5. **Full-trajectory analysis from the start**, per explicit user instruction, same methodology
   as A2/C2/C3/C4.

## What was done and how

1. Added `RegionReconstruction` import + `self.DefRec` + the `activate_DefRec=True` branch to
   `adapters/models_pointnet2.py::PointNet2_ClsSeg` (see design decision 2). Purely additive —
   A1/A2 never set `activate_DefRec=True`, so they're unaffected.
2. Wrote `adapters/train_a3_pointnet2_da_s.py` (see design decision 3).
3. Added `jobs/a3_pointnet2_da_s.sbatch` — same cluster settings as A1/A2/C3 (dgx partition/qos,
   1 GPU, 4h wall time), both `DEFREC_ROOT` and `POINTNET2_ROOT` exported (needed for the
   PointNet++ backbone, same as A1/A2's jobs).
4. **CPU smoke test** (1 epoch, batch_size=8, real data, `--gpu -1`): ran end-to-end in ~9.5
   minutes (slower than A1/A2's smoke tests, expected — Chamfer distance is the most expensive
   op in this whole project's training loops, same as C3's own smoke test) with no shape/gradient
   errors. `RegionReconstruction` composed correctly (no shape mismatch feeding it PointNet2's
   1152-wide `seg_feat`), all losses non-zero and in sane single-epoch ranges (`defrec loss`
   substantially larger than `cls`/`seg` loss, expected — same pattern C3 showed). Deleted
   `results/_smoketest_a3/` afterward.
5. Added `jobs/a3_pointnet2_da_s.sbatch`.

---
**2026-09-12:** Submitted `jobs/a3_pointnet2_da_s.sbatch` to the `dgx` partition — job **308789**.
Queue was empty beforehand. Awaiting completion.
