#!/bin/bash
#SBATCH --job-name=d6_fusion_physics_gompertz
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# D6, the Gompertz counterpart of D5 (jobs/d5_fusion_physics_logistic.sbatch). Same profile as
# D5's real run: warm-started from D4's checkpoint, ~59-row train table, CPU (train_d6_fusion_
# physics_gompertz.py defaults to --gpu -1); --gres=gpu:1 kept only for project convention;
# --time=01:00:00 generous headroom. Submitted via sbatch so it survives an interactive session
# disconnecting, matching D5's established practice.
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/d6_fusion_physics_gompertz_%j.log

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
cd "$REPO_ROOT"

python -u adapters/train_d6_fusion_physics_gompertz.py --gpu -1
