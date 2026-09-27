#!/bin/bash
#SBATCH --job-name=d1_fusion_baseline
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# D1 trains only a small regression head (TraitFusionHead, a direct 256->7 linear layer,
# 1,799 params -- see step_notes/D1_Fusion_Baseline.md) on top of B3b's FROZEN, precomputed
# pooled features (data/features/Pheno4D/, cached once by adapters/extract_b3b_features.py).
# No KPConv collate, no point-cloud processing at all in this training loop -- unlike every
# Block B row, this one runs in well under a minute even on CPU. --gres=gpu:1 kept only for
# project convention; --time=01:00:00 is generous headroom, not a real constraint.
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/d1_fusion_baseline_%j.log

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
cd "$REPO_ROOT"

python -u adapters/train_d1_fusion_baseline.py --gpu 0
