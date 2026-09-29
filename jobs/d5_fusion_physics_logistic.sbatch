#!/bin/bash
#SBATCH --job-name=d5_fusion_physics_logistic
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# D5 reuses D4's exact architecture/dataset (PerTermFusionHeadD4 warm-started from D4's own
# checkpoint) on the same small (~59-row train) table -- D4's own CPU run (results/
# D4_fusion_previous_stage/run.log) took 27s for 200 epochs; D5 adds a once-per-epoch full-table
# physics/monotonicity pass on the same small table (negligible extra cost). --gres=gpu:1 kept
# only for project convention (train_d5_fusion_physics_logistic.py defaults to --gpu -1/CPU, same
# as D1-D4); --time=01:00:00 is generous headroom, not a real constraint. Submitted via sbatch
# (rather than run directly, like D1-D4 were) specifically so this row survives an interactive
# session disconnecting mid-run -- explicit user instruction given repeated connection drops
# during D5's physics-loss debugging session (2026-09-29).
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/d5_fusion_physics_logistic_%j.log

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
cd "$REPO_ROOT"

python -u adapters/train_d5_fusion_physics_logistic.py --gpu -1
