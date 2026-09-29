# A6: PointNet++, DA-D (Deep CORAL) — added row

Not in CLAUDE.md's original strategy table (Block A is A1-A5: DA-0/DA-A/DA-S/DA-A-LN/DA-O — no
DA-D slot). Added per explicit user instruction, alongside its DGCNN counterpart (C6), same
category of addition as B3b was to Block B. Numbered A6 (next available slot).

## Motivating question

Shared with C6 (see that file for the full framing): does DA-D's large KPConv win (B3, +17.9
pts) generalize across backbones, or was it KPConv-specific? PointNet++'s own story differs from
DGCNN's in a relevant way: A2 (DA-A) hurts mildly (and turns out to be partly an
augmentation-mix artifact — A4 nearly closes the gap just by narrowing to L-N), while A3 (DA-S)
is this backbone's BEST non-Oracle result (+20.1 pts over A1) — so PointNet++ has already shown
one cooperative-loss win. A6 tests whether a second, mechanistically distinct cooperative method
also helps, which would strengthen "cooperative losses help this backbone" as a pattern rather
than something specific to DefRec's reconstruction mechanism.

## Implementation

`adapters/train_a6_pointnet2_da_d.py`: byte-for-byte the same plumbing as
`train_a2_pointnet2_da_a.py` (`PlantClsSegDataset`/`PlantSpeciesDataset`, `PointNet2_ClsSeg`,
same split) with C6's/B3's DA-D loss machinery substituted for A2's DANN machinery — identical
pattern to how C6 was built, itself mirroring how A2 originally ported C2's DANN machinery onto
A1's plumbing. `coral_loss` unchanged (already backbone-agnostic).

CPU smoke test (1 epoch, real data) passed cleanly before submitting — sane non-degenerate
numbers, no shape/gradient errors, coral loss values in the same tiny-but-nonzero range as C6's
and B3's own epoch-0 readings.

## Status

Submitted to the cluster via `jobs/a6_pointnet2_da_d.sbatch` (2026-09-30), `--time=04:00:00`
matching A1-A5's own budget. Full-trajectory analysis against A1's own DA-0 baseline required
before drawing conclusions, per explicit user instruction — same discipline as A2/A3/A4's own
trajectory analyses. Results pending.
