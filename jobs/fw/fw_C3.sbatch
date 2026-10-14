#!/bin/bash
#SBATCH --job-name=fw_C3
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_C3_%j.log

# Full-workflow rerun, row C3 (DGCNN, DA-S, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/c3_dgcnn_da_s.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_C3_s<seed>/. Seeds 1, 2, 3 run one after another.

ROW=C3
SCRIPT=train_c3_dgcnn_da_s.py
ARGS=(--batch_size 32 --lr 1e-3 --wd 5e-5 --dropout 0.5 --DefRec_dist volume_based_voxels --num_regions 3 --DefRec_weight 1.0)
NEEDS_KPCONV=0

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
