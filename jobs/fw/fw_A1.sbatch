#!/bin/bash
#SBATCH --job-name=fw_A1
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_A1_%j.log

# Full-workflow rerun, row A1 (PointNet++, DA-0, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/a1_pointnet2_da0.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_A1_s<seed>/. Seeds 1, 2, 3 run one after another.

ROW=A1
SCRIPT=train_a1_pointnet2_da0.py
ARGS=(--batch_size 32 --lr 1e-3 --wd 5e-5 --dropout 0.5)
NEEDS_KPCONV=0

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
