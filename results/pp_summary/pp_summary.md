**Block PP: C1 baseline (DGCNN, DA-0) per preprocessing pipeline** (mean ± std over seeds; target = Pheno4D held-out, 63 clouds)

| Pipeline | Seeds | Full-run mean target acc | Full-run mean target bal. acc | Selected-checkpoint target acc | Source seg mIoU, tomato | Source seg mIoU, maize* (not comparable across pipelines (absent classes scored 0)) | Decision vs P0 |
|---|---|---|---|---|---|---|---|
| P0 | 1,2,3 | 0.845 ± 0.048 | 0.826 ± 0.042 | 0.862 ± 0.102 | 0.308 ± 0.018 | 0.321 ± 0.019 | reference |
| P1 | 1,2,3 | 0.864 ± 0.009 | 0.839 ± 0.006 | 0.915 ± 0.018 | 0.307 ± 0.026 | 0.325 ± 0.029 | beats P0 |
| P2 | 1,2,3 | 0.717 ± 0.010 | 0.777 ± 0.008 | 0.772 ± 0.024 | 0.317 ± 0.024 | 0.177 ± 0.013 | no clear difference |
| P3 | 1,2,3 | 0.703 ± 0.014 | 0.766 ± 0.010 | 0.730 ± 0.069 | 0.322 ± 0.011 | 0.191 ± 0.020 | no clear difference |
| P4 | 1,2,3 | 0.703 ± 0.014 | 0.766 ± 0.010 | 0.730 ± 0.069 | 0.322 ± 0.011 | 0.191 ± 0.020 | no clear difference |
| P4a | 1,2,3 | 0.863 ± 0.015 | 0.846 ± 0.014 | 0.884 ± 0.037 | 0.333 ± 0.022 | 0.345 ± 0.004 | beats P0 |
| P4b | 1,2,3 | 0.678 ± 0.006 | 0.747 ± 0.005 | 0.688 ± 0.072 | 0.295 ± 0.042 | 0.188 ± 0.017 | no clear difference |
| P4c | 1,2,3 | 0.703 ± 0.014 | 0.766 ± 0.010 | 0.730 ± 0.069 | 0.322 ± 0.011 | 0.191 ± 0.020 | no clear difference |
| P4d | 1,2,3 | 0.571 ± 0.021 | 0.662 ± 0.017 | 0.561 ± 0.033 | 0.298 ± 0.022 | 0.149 ± 0.017 | no clear difference |
| *majority baseline (40/63)* | – | 0.635 | – | – | – | – | – |
| *existing P0 single-seed C1* | – | 0.791 | – | – | – | – | – |

Decision rule: a pipeline beats P0 only if all 3 of its seeds have a higher full-run mean target accuracy than P0's mean (0.845, over 3 complete seed(s)). Selected checkpoint = lowest source validation loss; target labels never used for selection. Std is shown only when there are 2 or more seeds.

\* Maize mIoU: not comparable across pipelines (absent classes scored 0). Soil-removal pipelines (P2, P3, P4, P4b-P4d) have no Maize class 0 or 5 points, so their Maize mIoU is capped at 4/6; P0, P1 and P4a are not.
