# FW target accuracy: final epoch, best epoch, validation-selected, full-run mean

Written by `scripts/18_fw_epoch_accuracy.py` from the 54 FW run.logs (C1 = `PP_P4a_s<seed>`). Target = Pheno4D held-out, 63 clouds. SD = sample SD over seeds.

**Best epoch uses the target test labels to choose the epoch. It is an upper bound (an oracle choice of epoch), not a result any method could report.** Validation-selected = the checkpoint at the lowest source (Oracle rows: Oracle) validation loss, chosen without target labels; this is the honest single-checkpoint number. Full-run mean = mean over all epochs.

PointNet++: PointNet++ eval is not deterministic (random farthest-point-sampling start), so the FINAL re-evaluation of the selected checkpoint can differ by a cloud or two from that epoch's logged accuracy; see step_notes/PointNet2_Eval_Nondeterminism.md. The FINAL value is used, as in all earlier summaries.

## Per row (mean ± SD over 3 seeds)

| Row | Backbone / DA / Aug | Final epoch | Best epoch (uses target labels: upper bound) | Validation-selected | Full-run mean | Best − selected | Seeds |
|---|---|---|---|---|---|---|---|
| A1 | PointNet++ / DA-0 / ALL | 0.4286 ± 0.0550 | 0.9471 ± 0.0243 | 0.4233 ± 0.0330 | 0.4968 ± 0.0202 | 0.5238 | 3 |
| A2 | PointNet++ / DA-A / ALL | 0.5027 ± 0.1475 | 0.9418 ± 0.0330 | 0.5132 ± 0.1191 | 0.5651 ± 0.0674 | 0.4286 | 3 |
| A3 | PointNet++ / DA-S / ALL | 0.6614 ± 0.2737 | 0.9629 ± 0.0367 | 0.7725 ± 0.2586 | 0.7313 ± 0.1024 | 0.1904 | 3 |
| A4 | PointNet++ / DA-A / L-N | 0.5132 ± 0.1283 | 0.9048 ± 0.0728 | 0.5185 ± 0.1081 | 0.5436 ± 0.0387 | 0.3862 | 3 |
| A5 (Oracle) | PointNet++ / DA-O / ALL | 0.9894 ± 0.0092 | 1.0000 ± 0.0000 | 0.9894 ± 0.0092 | 0.9275 ± 0.0113 | 0.0106 | 3 |
| A6 | PointNet++ / DA-D / ALL | 0.7725 ± 0.1350 | 0.9471 ± 0.0330 | 0.7725 ± 0.0783 | 0.6961 ± 0.0699 | 0.1746 | 3 |
| B1 | KPConv / DA-0 / ALL | 0.6666 ± 0.0420 | 0.9365 ± 0.0000 | 0.7196 ± 0.0601 | 0.6729 ± 0.0107 | 0.2169 | 3 |
| B2 | KPConv / DA-A / ALL | 0.6613 ± 0.0733 | 0.9365 ± 0.0000 | 0.6720 ± 0.0242 | 0.6844 ± 0.0631 | 0.2645 | 3 |
| B3 | KPConv / DA-D / ALL | 0.6349 ± 0.0824 | 0.9312 ± 0.0092 | 0.6455 ± 0.0799 | 0.7020 ± 0.0196 | 0.2857 | 3 |
| B3b | KPConv / DA-S / ALL | 0.7672 ± 0.2247 | 0.9418 ± 0.0092 | 0.7302 ± 0.2136 | 0.8316 ± 0.1053 | 0.2116 | 3 |
| B4 | KPConv / DA-0 / L-D | 0.8519 ± 0.0661 | 0.9524 ± 0.0159 | 0.6297 ± 0.1860 | 0.7866 ± 0.0402 | 0.3227 | 3 |
| B5 (Oracle) | KPConv / DA-O / ALL | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9947 ± 0.0092 | 0.9543 ± 0.0059 | 0.0053 | 3 |
| C1 | DGCNN / DA-0 / ALL | 0.8942 ± 0.0184 | 0.9524 ± 0.0275 | 0.8836 ± 0.0367 | 0.8632 ± 0.0155 | 0.0688 | 3 |
| C2 | DGCNN / DA-A / ALL | 0.4656 ± 0.0399 | 0.9418 ± 0.0243 | 0.7989 ± 0.1972 | 0.6429 ± 0.0471 | 0.1429 | 3 |
| C3 | DGCNN / DA-S / ALL | 0.9312 ± 0.0331 | 0.9683 ± 0.0000 | 0.9312 ± 0.0242 | 0.8525 ± 0.0121 | 0.0371 | 3 |
| C4 | DGCNN / DA-A / L-N | 0.5661 ± 0.0601 | 0.9100 ± 0.0330 | 0.6508 ± 0.0840 | 0.6317 ± 0.0279 | 0.2593 | 3 |
| C5 (Oracle) | DGCNN / DA-O / ALL | 0.9524 ± 0.0420 | 1.0000 ± 0.0000 | 0.9524 ± 0.0420 | 0.8914 ± 0.0044 | 0.0476 | 3 |
| C6 | DGCNN / DA-D / ALL | 0.9206 ± 0.0420 | 0.9577 ± 0.0184 | 0.9365 ± 0.0420 | 0.8687 ± 0.0410 | 0.0212 | 3 |

## Per run

| Run | Final epoch (e) | Best epoch, target labels (e) | Validation-selected (e) | Full-run mean | Warnings |
|---|---|---|---|---|---|
| results/FW_A1_s1 | 0.3968 (99) | 0.9683 (13) | 0.4127 (97) | 0.5014 |  |
| results/FW_A1_s2 | 0.3968 (99) | 0.9206 (13) | 0.3968 (81) | 0.5143 |  |
| results/FW_A1_s3 | 0.4921 (99) | 0.9524 (24) | 0.4603 (92) | 0.4748 | FINAL target acc 0.4603 != epoch 92's logged 0.4762 (PointNet++ eval nondeterminism) |
| results/FW_A2_s1 | 0.6667 (99) | 0.9524 (19) | 0.6349 (66) | 0.6333 |  |
| results/FW_A2_s2 | 0.3810 (99) | 0.9683 (12) | 0.3968 (40) | 0.4986 |  |
| results/FW_A2_s3 | 0.4603 (99) | 0.9048 (36) | 0.5079 (54) | 0.5635 | FINAL target acc 0.5079 != epoch 54's logged 0.4921 (PointNet++ eval nondeterminism) |
| results/FW_A3_s1 | 0.7143 (99) | 0.9841 (11) | 0.8889 (77) | 0.7932 | FINAL target acc 0.8889 != epoch 77's logged 0.9206 (PointNet++ eval nondeterminism) |
| results/FW_A3_s2 | 0.3651 (99) | 0.9206 (67) | 0.4762 (88) | 0.6130 | FINAL target acc 0.4762 != epoch 88's logged 0.4444 (PointNet++ eval nondeterminism) |
| results/FW_A3_s3 | 0.9048 (99) | 0.9841 (24, 2 epochs tie) | 0.9524 (82) | 0.7876 | FINAL target acc 0.9524 != epoch 82's logged 0.9365 (PointNet++ eval nondeterminism) |
| results/FW_A4_s1 | 0.3651 (99) | 0.9683 (28, 2 epochs tie) | 0.3968 (41) | 0.5000 |  |
| results/FW_A4_s2 | 0.5873 (99) | 0.9206 (26) | 0.6032 (51) | 0.5738 |  |
| results/FW_A4_s3 | 0.5873 (99) | 0.8254 (57) | 0.5556 (54) | 0.5570 | FINAL target acc 0.5556 != epoch 54's logged 0.5397 (PointNet++ eval nondeterminism) |
| results/FW_A5_s1 | 0.9841 (99) | 1.0000 (10, 21 epochs tie) | 0.9841 (93) | 0.9314 | FINAL target acc 0.9841 != epoch 93's logged 1.0000 (PointNet++ eval nondeterminism) |
| results/FW_A5_s2 | 0.9841 (99) | 1.0000 (8, 36 epochs tie) | 0.9841 (95) | 0.9363 |  |
| results/FW_A5_s3 | 1.0000 (99) | 1.0000 (31, 60 epochs tie) | 1.0000 (95) | 0.9148 |  |
| results/FW_A6_s1 | 0.6190 (99) | 0.9206 (5) | 0.6825 (97) | 0.6495 | FINAL target acc 0.6825 != epoch 97's logged 0.6508 (PointNet++ eval nondeterminism) |
| results/FW_A6_s2 | 0.8730 (99) | 0.9841 (29) | 0.8254 (72) | 0.7765 | FINAL target acc 0.8254 != epoch 72's logged 0.7778 (PointNet++ eval nondeterminism) |
| results/FW_A6_s3 | 0.8254 (99) | 0.9365 (23, 2 epochs tie) | 0.8095 (84) | 0.6624 |  |
| results/FW_B1_s1 | 0.6984 (99) | 0.9365 (7, 4 epochs tie) | 0.6508 (96) | 0.6637 | FINAL target acc 0.6508 != epoch 96's logged 0.7143 |
| results/FW_B1_s2 | 0.6825 (99) | 0.9365 (4, 9 epochs tie) | 0.7619 (86) | 0.6846 | FINAL target acc 0.7619 != epoch 86's logged 0.7302 |
| results/FW_B1_s3 | 0.6190 (99) | 0.9365 (7, 8 epochs tie) | 0.7460 (84) | 0.6703 | FINAL target acc 0.7460 != epoch 84's logged 0.7143 |
| results/FW_B2_s1 | 0.7460 (99) | 0.9365 (4, 3 epochs tie) | 0.6984 (60) | 0.7275 |  |
| results/FW_B2_s2 | 0.6190 (99) | 0.9365 (6, 2 epochs tie) | 0.6667 (55) | 0.7138 | FINAL target acc 0.6667 != epoch 55's logged 0.6825 |
| results/FW_B2_s3 | 0.6190 (99) | 0.9365 (3, 2 epochs tie) | 0.6508 (72) | 0.6121 | FINAL target acc 0.6508 != epoch 72's logged 0.6190 |
| results/FW_B3_s1 | 0.6825 (99) | 0.9206 (2, 4 epochs tie) | 0.7302 (94) | 0.6913 | FINAL target acc 0.7302 != epoch 94's logged 0.7143 |
| results/FW_B3_s2 | 0.5397 (99) | 0.9365 (5, 3 epochs tie) | 0.5714 (93) | 0.6902 | FINAL target acc 0.5714 != epoch 93's logged 0.5397 |
| results/FW_B3_s3 | 0.6825 (99) | 0.9365 (2, 2 epochs tie) | 0.6349 (83) | 0.7246 | logged best epoch 83 != argmin of logged val total loss 87 (4-dp rounding tie?) |
| results/FW_B3b_s1 | 0.8889 (99) | 0.9365 (4, 33 epochs tie) | 0.7937 (92) | 0.8754 | FINAL target acc 0.7937 != epoch 92's logged 0.7460 |
| results/FW_B3b_s2 | 0.9048 (99) | 0.9365 (6, 18 epochs tie) | 0.9048 (96) | 0.9079 |  |
| results/FW_B3b_s3 | 0.5079 (99) | 0.9524 (10) | 0.4921 (67) | 0.7114 | FINAL target acc 0.4921 != epoch 67's logged 0.4444 |
| results/FW_B4_s1 | 0.8730 (99) | 0.9365 (11, 4 epochs tie) | 0.8413 (82) | 0.8302 | FINAL target acc 0.8413 != epoch 82's logged 0.9206 |
| results/FW_B4_s2 | 0.7778 (99) | 0.9524 (60) | 0.4921 (31) | 0.7508 |  |
| results/FW_B4_s3 | 0.9048 (99) | 0.9683 (13, 3 epochs tie) | 0.5556 (44) | 0.7789 | FINAL target acc 0.5556 != epoch 44's logged 0.5714 |
| results/FW_B5_s1 | 1.0000 (99) | 1.0000 (17, 52 epochs tie) | 0.9841 (74) | 0.9603 | FINAL target acc 0.9841 != epoch 74's logged 1.0000 |
| results/FW_B5_s2 | 1.0000 (99) | 1.0000 (15, 66 epochs tie) | 1.0000 (88) | 0.9486 |  |
| results/FW_B5_s3 | 1.0000 (99) | 1.0000 (16, 73 epochs tie) | 1.0000 (69) | 0.9541 |  |
| results/PP_P4a_s1 | 0.8730 (99) | 0.9683 (6) | 0.8413 (95) | 0.8467 |  |
| results/PP_P4a_s2 | 0.9048 (99) | 0.9206 (5, 2 epochs tie) | 0.9048 (99) | 0.8656 |  |
| results/PP_P4a_s3 | 0.9048 (99) | 0.9683 (38, 2 epochs tie) | 0.9048 (92) | 0.8773 |  |
| results/FW_C2_s1 | 0.4603 (99) | 0.9683 (6) | 0.5714 (67) | 0.6057 |  |
| results/FW_C2_s2 | 0.4286 (99) | 0.9206 (7, 3 epochs tie) | 0.9206 (62) | 0.6270 |  |
| results/FW_C2_s3 | 0.5079 (99) | 0.9365 (7) | 0.9048 (70) | 0.6959 |  |
| results/FW_C3_s1 | 0.9048 (99) | 0.9683 (45) | 0.9048 (99) | 0.8406 |  |
| results/FW_C3_s2 | 0.9683 (99) | 0.9683 (99) | 0.9365 (79) | 0.8649 |  |
| results/FW_C3_s3 | 0.9206 (99) | 0.9683 (52) | 0.9524 (85) | 0.8519 |  |
| results/FW_C4_s1 | 0.5397 (99) | 0.9206 (9) | 0.5873 (55) | 0.6192 |  |
| results/FW_C4_s2 | 0.5238 (99) | 0.8730 (52) | 0.6190 (58) | 0.6122 |  |
| results/FW_C4_s3 | 0.6349 (99) | 0.9365 (12) | 0.7460 (32) | 0.6636 |  |
| results/FW_C5_s1 | 1.0000 (99) | 1.0000 (19, 12 epochs tie) | 1.0000 (97) | 0.8902 |  |
| results/FW_C5_s2 | 0.9365 (99) | 1.0000 (45, 3 epochs tie) | 0.9365 (99) | 0.8878 |  |
| results/FW_C5_s3 | 0.9206 (99) | 1.0000 (36, 5 epochs tie) | 0.9206 (99) | 0.8963 |  |
| results/FW_C6_s1 | 0.9524 (99) | 0.9683 (36, 5 epochs tie) | 0.9683 (98) | 0.8754 |  |
| results/FW_C6_s2 | 0.8730 (99) | 0.9365 (2, 9 epochs tie) | 0.8889 (95) | 0.8248 |  |
| results/FW_C6_s3 | 0.9365 (99) | 0.9683 (45, 2 epochs tie) | 0.9524 (97) | 0.9059 |  |
