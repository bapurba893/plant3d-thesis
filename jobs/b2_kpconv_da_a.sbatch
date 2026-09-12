#!/bin/bash
#SBATCH --job-name=b2_kpconv_da_a
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=60:00:00
# Standing Block B (KPConv) convention per CLAUDE.md's Backbones section, applied from the
# start (not rediscovered per row): KPConv's collate is CPU-bound/single-threaded, and B1 hit
# real shared-GPU contention on this partition (job 308956 was on pace for ~48h before being
# killed and resubmitted at 60h) -- 60h stays the budget for every Block B row regardless of
# how fast any individual run turns out to be (B1 itself finished in 2h15m once it landed on an
# uncontended node -- see step_notes/B1_KPConv_DA0.md).
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/b2_kpconv_da_a_%j.log

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
python -u adapters/train_b2_kpconv_da_a.py \
    --exp_name B2_kpconv_da_a \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --wd 5e-5 \
    --dropout 0.5 \
    --disc_dropout 0.3 \
    --gamma 10.0 \
    --lambda_ent 0.1 \
    --num_workers 0 \
    --verbose_batches
