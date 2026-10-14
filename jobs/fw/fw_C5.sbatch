#!/bin/bash
#SBATCH --job-name=fw_C5
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_C5_%j.log

# Full-workflow rerun, row C5 (DGCNN, DA-O, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/c5_dgcnn_da_o.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_C5_s<seed>/. Seeds 1, 2, 3 run one after another.

ROW=C5
SCRIPT=train_c5_dgcnn_da_o.py
ARGS=(--batch_size 8 --test_batch_size 16 --lr 1e-3 --wd 5e-5 --dropout 0.5)
NEEDS_KPCONV=0

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
