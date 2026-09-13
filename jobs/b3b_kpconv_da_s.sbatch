#!/bin/bash
#SBATCH --job-name=b3b_kpconv_da_s
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=60:00:00
# Standing Block B (KPConv) convention per CLAUDE.md's Backbones section. This row (DA-S) is
# expected to need MORE collate calls per step than B1/B2 (deformed-batch rebuilds, see
# adapters/train_b3b_kpconv_da_s.py's docstring) -- the 60h budget already has real margin for
# this even at several times B1/B2's per-step cost, given how fast B1 (2h15m) and B2 (2h44m)
# both finished once they landed on uncontended nodes.
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/b3b_kpconv_da_s_%j.log

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
python -u adapters/train_b3b_kpconv_da_s.py \
    --exp_name B3b_kpconv_da_s \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --wd 5e-5 \
    --dropout 0.5 \
    --DefRec_dist volume_based_voxels \
    --num_regions 3 \
    --DefRec_weight 1.0 \
    --num_workers 0 \
    --verbose_batches
