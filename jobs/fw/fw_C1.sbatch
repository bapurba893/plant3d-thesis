#!/bin/bash
#SBATCH --job-name=fw_C1
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_C1_%j.log

# Full-workflow rerun, row C1 (DGCNN, DA-0, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/c1_dgcnn_da0_clsseg.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_C1_s<seed>/. Seeds 1, 2, 3 run one after another.
# NOTE: Same script, arguments, seeds and manifest as results/PP_P4a_s{1,2,3} (Block PP) -- this row repeats those runs; it can be skipped and PP_P4a reused.

ROW=C1
SCRIPT=train_c1_dgcnn_da0.py
ARGS=(--batch_size 32 --lr 1e-3 --wd 5e-5 --dropout 0.5)
NEEDS_KPCONV=0

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
