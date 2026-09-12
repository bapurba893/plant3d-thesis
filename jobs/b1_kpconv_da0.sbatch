#!/bin/bash
#SBATCH --job-name=b1_kpconv_da0
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
# KPConv's collate (grid subsampling + radius neighbor search, via the compiled C++
# extensions) is CPU-bound and single-threaded (no OpenMP in the vendored code, and
# multi-worker DataLoader parallelism showed signs of a deadlock/severe slowdown when
# tried -- reverted, see step_notes/B1_KPConv_DA0.md). Unlike every other row's ~13min-1h04m
# full 100-epoch runtime, this one is expected to take on the order of 10-15+ hours --
# NOT a bug, a real, backbone-specific cost. dgx partition/QOS has no MaxWall limit
# (confirmed via `scontrol show partition dgx`: MaxTime=6-00:00:00) -- the previous 4h
# budget here was just this project's own convention copied from the other backbones,
# not a cluster policy, so raising it is safe.
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/b1_kpconv_da0_%j.log

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM   # unmodified upstream clone, used as a library only
                                                   # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
KPCONV_ROOT=/home/pearl/25m0301/KPConv-PyTorch    # cloned as a sibling repo (commit d19c575d3fa9fcfd5a74845b5b27aac7e50472c7);
                                                   # cpp_wrappers/*/setup.py and wrapper.cpp carry minimal
                                                   # compatibility patches for numpy>=2/modern setuptools --
                                                   # see step_notes/B1_KPConv_DA0.md
export DEFREC_ROOT
export KPCONV_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"

cd "$REPO_ROOT"
python -u adapters/train_b1_kpconv_da0.py \
    --exp_name B1_kpconv_da0 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --wd 5e-5 \
    --dropout 0.5 \
    --num_workers 0 \
    --verbose_batches
