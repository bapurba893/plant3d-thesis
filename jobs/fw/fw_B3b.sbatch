#!/bin/bash
#SBATCH --job-name=fw_B3b
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --array=1-3
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_B3b_%A_%a.log

# Full-workflow rerun, row B3b (KPConv, DA-S, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/b3b_kpconv_da_s.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_B3b_s<seed>/. One seed per array task (1-3).
# NOTE: Split into a 3-task array (one seed per task): ~6 h per seed nominal, 3 in a row would leave too little margin in 60 h under GPU contention.
# Block B: 60 h wall-clock budget per job (CLAUDE.md, Backbones section).

ROW=B3b
SCRIPT=train_b3b_kpconv_da_s.py
ARGS=(--batch_size 16 --lr 1e-3 --wd 5e-5 --dropout 0.5 --DefRec_dist volume_based_voxels --num_regions 3 --DefRec_weight 1.0 --num_workers 0 --verbose_batches)
NEEDS_KPCONV=1

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
