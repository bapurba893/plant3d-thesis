# Shortcut probe (docs/PREPROCESSING_COMPARISON_PLAN.md section 7, item 5)

Logistic regression on non-shape features, trained on Crops3D train (species label), scored on Crops3D val (source) and Pheno4D held-out (target). `all3` = n_raw + nn_spacing + flatness; `cache_only` = nn_spacing + flatness (no manifest-derived raw point count).

pipeline features     src_acc  src_auc  tgt_acc  tgt_auc
--------------------------------------------------------
P0       all3          0.9778   0.9949   0.6508   0.9978
P0       cache_only    0.9778   0.9823   0.9206   0.9022
P1       all3          0.9778   1.0000   0.6667   0.9967
P1       cache_only    0.9778   0.9975   0.9206   0.9446
P2       all3          0.9778   1.0000   0.6508   0.9957
P2       cache_only    0.9556   0.9899   0.5397   0.8500
P3       all3          0.9778   1.0000   0.7619   0.9989
P3       cache_only    0.9778   1.0000   0.7460   0.9174
P4       all3          0.9778   1.0000   0.7619   0.9989
P4       cache_only    0.9778   1.0000   0.7460   0.9174
P4a      all3          0.9778   0.9975   0.6667   0.9978
P4a      cache_only    0.9778   0.9899   0.9048   0.8913
P4b      all3          0.9778   1.0000   0.7302   0.9946
P4b      cache_only    0.9778   0.9975   0.6349   0.8815
P4c      all3          0.9778   1.0000   0.7619   0.9989
P4c      cache_only    0.9778   1.0000   0.7460   0.9174
P4d      all3          1.0000   1.0000   0.7460   0.9946
P4d      cache_only    0.9778   0.9975   0.5714   0.9065
