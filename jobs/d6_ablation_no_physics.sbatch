#!/bin/bash
#SBATCH --job-name=d6_ablation_no_physics
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# Control ablation for D6, the Gompertz counterpart of jobs/d5_ablation_no_physics.sbatch (see
# step_notes/D5_D6_Physics_Loss.md's D5 ablation section for why this is needed: D5's raw R^2
# gains were mostly a warm-start/extra-training confound, not the physics loss itself, confirmed
# by an identical run with lambda_phys=lambda_mono=0). IDENTICAL to D6 in every respect (same
# warm start from D4, same 200 epochs, same cosine schedule, same seed) except
# --lambda_phys 0 --lambda_mono 0, isolating the physics/monotonicity loss's own contribution
# from the extra-training effect. Submitted via sbatch, matching D5/D6's established practice.
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/d6_ablation_no_physics_%j.log

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
cd "$REPO_ROOT"

python -u adapters/train_d6_fusion_physics_gompertz.py --gpu -1 \
    --exp_name D6_ablation_no_physics \
    --lambda_phys 0 --lambda_mono 0
