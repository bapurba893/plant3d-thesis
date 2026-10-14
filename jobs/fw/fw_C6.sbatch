#!/bin/bash
#SBATCH --job-name=fw_C6
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_C6_%j.log

# Full-workflow rerun, row C6 (DGCNN, DA-D, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/c6_dgcnn_da_d.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_C6_s<seed>/. Seeds 1, 2, 3 run one after another.

ROW=C6
SCRIPT=train_c6_dgcnn_da_d.py
ARGS=(--batch_size 32 --lr 1e-3 --wd 5e-5 --dropout 0.5)
NEEDS_KPCONV=0

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
