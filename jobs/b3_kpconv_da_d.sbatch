#!/bin/bash
#SBATCH --job-name=b3_kpconv_da_d
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=60:00:00
# Standing Block B (KPConv) convention per CLAUDE.md's Backbones section. This row (DA-D/CORAL)
# needs no deformed-batch rebuilds (unlike B3b/DA-S) -- structurally closer to B1/B2's cost
# shape, so expected to land near their real runtimes (2-3h) rather than B3b's ~5h, but the
# 60h budget stays the standing policy regardless.
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/b3_kpconv_da_d_%j.log

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM   # unmodified upstream clone, used as a library only
                                                   # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
KPCONV_ROOT=/home/pearl/25m0301/KPConv-PyTorch    # cloned as a sibling repo (commit d19c575d3fa9fcfd5a74845b5b27aac7e50472c7)
export DEFREC_ROOT
export KPCONV_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"

cd "$REPO_ROOT"
python -u adapters/train_b3_kpconv_da_d.py \
    --exp_name B3_kpconv_da_d \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --wd 5e-5 \
    --dropout 0.5 \
    --num_workers 0 \
    --verbose_batches
