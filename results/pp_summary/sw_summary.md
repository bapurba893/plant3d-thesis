**Block PP: learned soil weight vs. soil kept (P4a) and soil removed (P4)** (DGCNN, DA-0; target = Pheno4D held-out, 63 clouds)

| Version | Metric | Seed 1 | Seed 2 | Seed 3 | Mean ± std |
|---|---|---|---|---|---|
| P4a (soil kept) | Full-run mean target acc (primary) | 0.847 | 0.866 | 0.877 | 0.863 ± 0.015 |
| P4a (soil kept) | Full-run mean target bal. acc | 0.840 | 0.836 | 0.862 | 0.846 ± 0.014 |
| P4a (soil kept) | Selected-checkpoint target acc | 0.841 | 0.905 | 0.905 | 0.884 ± 0.037 |
| P4 (soil removed) | Full-run mean target acc (primary) | 0.715 | 0.688 | 0.707 | 0.703 ± 0.014 |
| P4 (soil removed) | Full-run mean target bal. acc | 0.774 | 0.754 | 0.769 | 0.766 ± 0.010 |
| P4 (soil removed) | Selected-checkpoint target acc | 0.778 | 0.651 | 0.762 | 0.730 ± 0.069 |
| SW_scalar (learned soil weight, scalar) | Full-run mean target acc (primary) | 0.814 | 0.872 | 0.865 | 0.850 ± 0.032 |
| SW_scalar (learned soil weight, scalar) | Full-run mean target bal. acc | 0.807 | 0.848 | 0.849 | 0.835 ± 0.024 |
| SW_scalar (learned soil weight, scalar) | Selected-checkpoint target acc | 0.825 | 0.921 | 0.889 | 0.878 ± 0.048 |
| SW_features (learned soil weight, features) | Full-run mean target acc (primary) | 0.870 | 0.857 | 0.866 | 0.864 ± 0.006 |
| SW_features (learned soil weight, features) | Full-run mean target bal. acc | 0.842 | 0.831 | 0.859 | 0.844 ± 0.014 |
| SW_features (learned soil weight, features) | Selected-checkpoint target acc | 0.905 | 0.905 | 0.936 | 0.915 ± 0.018 |

Full-run mean = mean over the 100 epochs of the per-epoch target accuracy. Selected checkpoint = lowest source validation loss; target labels are never used for selection. Std = sample std over the 3 seeds. Maize mIoU is not reported: it is not comparable between soil-kept and soil-removed versions.

**Decision rule:** a learned version beats P4a only if all 3 of its seeds exceed the P4a 3-seed mean full-run accuracy (0.8632).

- SW_scalar beats P4a: NO -- 2/3 seeds above the P4a 3-seed mean 0.8632 (s1=0.8138, s2=0.8724, s3=0.8651)
- SW_features beats P4a: NO -- 2/3 seeds above the P4a 3-seed mean 0.8632 (s1=0.8695, s2=0.8570, s3=0.8657)

**Soil weight, last 10 epochs** (raw = bounded weight in [0.2, 5.0]; normalised = after scaling to mean 1 over all maize points per batch; means averaged over the epochs, min/max over the epochs; at-bound fractions = share of soil points within 0.048 of 0.2 or 5.0)

| Version | Seed | Raw mean | Raw min | Raw max | Norm. mean | Norm. min | Norm. max | Plant mean | Frac. at 0.2 | Frac. at 5.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| SW_scalar | 1 | 0.8623 | 0.8623 | 0.8625 | 0.8987 | 0.8845 | 0.9129 | 1.0406 | 0.0000 | 0.0000 |
| SW_scalar | 2 | 0.8601 | 0.8600 | 0.8602 | 0.8969 | 0.8794 | 0.9090 | 1.0414 | 0.0000 | 0.0000 |
| SW_scalar | 3 | 0.8629 | 0.8628 | 0.8630 | 0.8984 | 0.8881 | 0.9109 | 1.0402 | 0.0000 | 0.0000 |
| SW_features | 1 | 0.2565 | 0.2009 | 0.3948 | 0.3287 | 0.2348 | 0.5457 | 1.2691 | 0.4514 | 0.0000 |
| SW_features | 2 | 0.2546 | 0.2005 | 0.3487 | 0.3263 | 0.2272 | 0.4723 | 1.2704 | 0.4289 | 0.0000 |
| SW_features | 3 | 0.2371 | 0.2006 | 0.3557 | 0.3042 | 0.2395 | 0.4683 | 1.2754 | 0.7557 | 0.0000 |

Bounds: YES -- soil points at a bound in: SW_features s1 (lo=0.4514, hi=0.0000); SW_features s2 (lo=0.4289, hi=0.0000); SW_features s3 (lo=0.7557, hi=0.0000).

**Missing / warnings:**

- none -- all 12 runs complete, all metrics computed
