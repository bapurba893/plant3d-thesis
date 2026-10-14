#!/bin/bash
#SBATCH --job-name=fw_A2
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_A2_%j.log

# Full-workflow rerun, row A2 (PointNet++, DA-A, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/a2_pointnet2_da_a.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_A2_s<seed>/. Seeds 1, 2, 3 run one after another.

ROW=A2
SCRIPT=train_a2_pointnet2_da_a.py
ARGS=(--batch_size 32 --lr 1e-3 --wd 5e-5 --dropout 0.5 --disc_dropout 0.3 --gamma 10.0 --lambda_ent 0.1)
NEEDS_KPCONV=0

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
