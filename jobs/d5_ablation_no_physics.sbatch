#!/bin/bash
#SBATCH --job-name=d5_ablation_no_physics
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# Control ablation for D5 (see step_notes/D5_D6_Physics_Loss.md's "Real D5 training run" section,
# 2026-09-29): D5's R^2 improved over D4 on ALL 7 terms, including 5 terms L_phys/L_mono's
# gradients cannot reach at all (PerTermFusionHeadD4's heads are fully independent per-term
# nn.Linear layers, no shared trunk) -- strong evidence the improvement is actually coming from
# D5's warm-start-from-D4 + fresh 200-epoch cosine schedule (extra training), not the physics
# constraint itself. This run is IDENTICAL to D5 in every respect (same warm start, same 200
# epochs, same cosine schedule, same seed, same lr/wd/dropout/lambda_corr/lr_ode/lr_K) except
# --lambda_phys 0 --lambda_mono 0, which zeroes ONLY the physics/monotonicity loss contribution
# while keeping run_physics_mono_step's extra per-epoch forward pass (and its RNG/dropout
# consumption) in the training loop unchanged -- the minimal-diff way to isolate the "extra
# training" confound from the physics constraint's own effect. Submitted via sbatch, matching D5,
# so it survives an interactive session disconnecting.
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/d5_ablation_no_physics_%j.log

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
cd "$REPO_ROOT"

python -u adapters/train_d5_fusion_physics_logistic.py --gpu -1 \
    --exp_name D5_ablation_no_physics \
    --lambda_phys 0 --lambda_mono 0
