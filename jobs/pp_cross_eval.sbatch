#!/bin/bash
#SBATCH --job-name=pp_cross_eval
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/pp_cross_eval_%j.log

# Block PP soil-cue test (step_notes/PP_2_Results.md caveat 3.4): evaluate the already-trained
# PP checkpoints on Pheno4D held-out clouds from the OTHER soil treatment's cache. No training.
#   soil-kept models (P0, P1, P4a) x seeds 1-3 -> own cache + P4 cache (soil removed)
#   soil-removed models (P3, P4)  x seeds 1-3 -> own cache + P4a cache (soil kept)
# One single job (dgx QOS: MaxSubmitPU=5), 30 evaluations of 63 clouds each.
#
# Runs on the GPU on purpose: the logged "FINAL target" numbers were produced on an A100 with
# PyTorch's default cuDNN TF32 convolutions, and a CPU (full fp32) evaluation of PP_P4a_s1 differs
# from its log by one borderline cloud (0.8571 vs 0.8413). The script first re-checks PP_P4a_s1 on
# its own cache against 0.8413 and writes nothing if that fails; every own-cache run is also
# checked against its run.log (column own_matches_log in cross_eval_runs.csv).
#
# Writes results/pp_summary/cross_eval.{md,csv} and cross_eval_runs.csv.

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM  # unmodified upstream clone, used as a library only
                                                 # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
export DEFREC_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"

cd /home/pearl/25m0301/plant3d-thesis
python scripts/11_cross_eval.py --mode cross --gpu 0
