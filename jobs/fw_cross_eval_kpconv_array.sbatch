#!/bin/bash
#SBATCH --job-name=fw_xeval_kp
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=1-3
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_cross_eval_kpconv_%A_%a.log

# FW soil-cue test, stage 2b (KPConv) split into 3 independent array tasks so one slow row cannot
# hold up the others. Same evaluation as jobs/fw_cross_eval.sbatch (scripts/16_fw_cross_eval.py,
# no training); only the row split, guard placement and output folders differ:
#   task 1: B1 B2   (guard on B1 seed 1 first; if it fails, only this task stops)
#   task 2: B3 B3b
#   task 3: B4 B5
# Each task writes to its own subfolder of results/fw_cross_eval/stage2b_kpconv_v2/, since the
# script writes fixed file names per --out_dir.
#   sbatch jobs/fw_cross_eval_kpconv_array.sbatch

BASE=results/fw_cross_eval/stage2b_kpconv_v2
case "$SLURM_ARRAY_TASK_ID" in
  1) ROWS="B1 B2";  OUT=$BASE/task1_B1_B2;  GUARD_ARGS="--guard B1_s1" ;;
  2) ROWS="B3 B3b"; OUT=$BASE/task2_B3_B3b; GUARD_ARGS="" ;;
  3) ROWS="B4 B5";  OUT=$BASE/task3_B4_B5;  GUARD_ARGS="" ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM   # unmodified upstream clone, used as a library only
                                                  # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
export DEFREC_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"
KPCONV_ROOT=/home/pearl/25m0301/KPConv-PyTorch    # sibling repo (commit d19c575d3fa9fcfd5a74845b5b27aac7e50472c7)
export KPCONV_ROOT

cd /home/pearl/25m0301/plant3d-thesis
python -u scripts/16_fw_cross_eval.py --rows $ROWS --gpu 0 --out_dir "$OUT" $GUARD_ARGS
