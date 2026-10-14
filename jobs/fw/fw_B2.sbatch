#!/bin/bash
#SBATCH --job-name=fw_B2
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_B2_%j.log

# Full-workflow rerun, row B2 (KPConv, DA-A, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/b2_kpconv_da_a.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_B2_s<seed>/. Seeds 1, 2, 3 run one after another.
# Block B: 60 h wall-clock budget per job (CLAUDE.md, Backbones section).

ROW=B2
SCRIPT=train_b2_kpconv_da_a.py
ARGS=(--batch_size 16 --lr 1e-3 --wd 5e-5 --dropout 0.5 --disc_dropout 0.3 --gamma 10.0 --lambda_ent 0.1 --num_workers 0 --verbose_batches)
NEEDS_KPCONV=1

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
