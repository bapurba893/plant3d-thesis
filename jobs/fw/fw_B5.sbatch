#!/bin/bash
#SBATCH --job-name=fw_B5
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_B5_%j.log

# Full-workflow rerun, row B5 (KPConv, DA-O, ALL) on the soil-kept P4a preprocessing.
# Arguments copied from jobs/b5_kpconv_da_o.sbatch; only --manifest, --seed and the result name differ.
# Results: results/FW_B5_s<seed>/. Seeds 1, 2, 3 run one after another.
# Block B: 60 h wall-clock budget per job (CLAUDE.md, Backbones section).

ROW=B5
SCRIPT=train_b5_kpconv_da_o.py
ARGS=(--batch_size 8 --test_batch_size 16 --lr 1e-3 --wd 5e-5 --dropout 0.5 --num_workers 0 --verbose_batches)
NEEDS_KPCONV=1

source /home/pearl/25m0301/plant3d-thesis/jobs/fw/fw_common.sh
