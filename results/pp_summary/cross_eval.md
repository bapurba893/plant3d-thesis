**Block PP soil-cue test: cross-evaluation** (no retraining; protocol-selected checkpoints `results/PP_<ID>_s<seed>/model.pt`; target = Pheno4D held-out, 63 clouds; mean ± std over seeds 1,2,3)

Soil-kept models (P0, P1, P4a) evaluated on the soil-removed target cache (P4); soil-removed models (P3, P4) on the soil-kept target cache (P4a). Change = cross minus the same model's own-cache result, averaged over seeds (per-seed values in the last column).

| Model pipeline | Model soil | Evaluated on cache | Target acc | Target bal. acc | Δ acc vs own | Δ bal. acc vs own | Δ acc per seed (s1 / s2 / s3) |
|---|---|---|---|---|---|---|---|
| P0 | kept | P0 (own) | 0.862 ± 0.102 | 0.842 ± 0.087 | reference | reference | – |
| P0 | kept | P4 | 0.619 ± 0.057 | 0.700 ± 0.045 | -0.243 | -0.142 | -0.190 / -0.270 / -0.270 |
| P1 | kept | P1 (own) | 0.915 ± 0.018 | 0.884 ± 0.025 | reference | reference | – |
| P1 | kept | P4 | 0.735 ± 0.081 | 0.792 ± 0.064 | -0.180 | -0.092 | -0.079 / -0.190 / -0.270 |
| P4a | kept | P4a (own) | 0.884 ± 0.037 | 0.865 ± 0.025 | reference | reference | – |
| P4a | kept | P4 | 0.651 ± 0.097 | 0.725 ± 0.076 | -0.233 | -0.140 | -0.238 / -0.143 / -0.317 |
| P3 | removed | P3 (own) | 0.730 ± 0.069 | 0.787 ± 0.054 | reference | reference | – |
| P3 | removed | P4a | 0.905 ± 0.032 | 0.870 ± 0.043 | +0.175 | +0.082 | +0.095 / +0.286 / +0.143 |
| P4 | removed | P4 (own) | 0.730 ± 0.069 | 0.787 ± 0.054 | reference | reference | – |
| P4 | removed | P4a | 0.905 ± 0.032 | 0.870 ± 0.043 | +0.175 | +0.082 | +0.095 / +0.286 / +0.143 |

Majority baseline (always Tomato): 40/63 = 0.635. Own-cache rows reproduce each run.log's FINAL target line (checked per run; see cross_eval_runs.csv, column `own_matches_log`). P3, P4 and P4c caches are identical in data (step_notes/PP_2_Results.md caveat 3.1), so P3-model-on-P4-cache equals P3 on its own cache and is not listed as a cross run.
