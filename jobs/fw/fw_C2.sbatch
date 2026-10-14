#!/bin/bash
#SBATCH --job-name=fw_C2
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_C2_%j.log

# Full-workflow rerun, row C2 (DGCNN, DA-A, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/c2_dgcnn_da_a.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_C2_s<seed>/. Seeds 1, 2, 3 run one after another.

ROW=C2
SCRIPT=train_c2_dgcnn_da_a.py
ARGS=(--batch_size 32 --lr 1e-3 --wd 5e-5 --dropout 0.5 --disc_dropout 0.3 --gamma 10.0 --lambda_ent 0.1)
NEEDS_KPCONV=0

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
