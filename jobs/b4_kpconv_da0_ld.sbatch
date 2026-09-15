#!/bin/bash
#SBATCH --job-name=b4_kpconv_da0_ld
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=60:00:00
# Standing Block B (KPConv) convention per CLAUDE.md's Backbones section. This row (DA-0, no
# adaptation machinery at all) is structurally identical to B1's cost shape -- expected to land
# near B1/B2's real runtimes (2-3h) unless it hits the same kind of shared-GPU contention B1 and
# B3 both did at various points; the 60h budget stays the standing policy regardless.
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/b4_kpconv_da0_ld_%j.log

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
python -u adapters/train_b4_kpconv_da0_ld.py \
    --exp_name B4_kpconv_da0_ld \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --wd 5e-5 \
    --dropout 0.5 \
    --num_workers 0 \
    --verbose_batches
