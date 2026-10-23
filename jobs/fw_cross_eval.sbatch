#!/bin/bash
#SBATCH --job-name=fw_cross_eval
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_cross_eval_%j.log

# FW soil-cue test: evaluate the P4a-trained FW checkpoints (results/FW_<row>_s<seed>/model.pt;
# C1 = results/PP_P4a_s<seed>/model.pt) on the Pheno4D held-out clouds of the P4a cache (own) and
# the P4 cache (soil removed). No training. See scripts/16_fw_cross_eval.py.
#
# Rows and output folder are set from the environment, so one job file serves both stages:
#   Stage 1 (DGCNN only):
#     sbatch --export=ALL,FW_XEVAL_ROWS="C1 C2 C3 C4 C5 C6",FW_XEVAL_OUT=results/fw_cross_eval/stage1_dgcnn jobs/fw_cross_eval.sbatch
#   Stage 2a (PointNet++):
#     sbatch --time=01:00:00 --export=ALL,FW_XEVAL_ROWS="A1 A2 A3 A4 A5 A6",FW_XEVAL_OUT=results/fw_cross_eval/stage2a_pointnet2 jobs/fw_cross_eval.sbatch
#   Stage 2b (KPConv, rows in this order, guard on B1 seed 1 first):
#     sbatch --dependency=afterok:<2a job> --export=ALL,FW_XEVAL_ROWS="B1 B2 B3 B3b B4 B5",FW_XEVAL_OUT=results/fw_cross_eval/stage2b_kpconv,FW_XEVAL_GUARD=B1_s1 jobs/fw_cross_eval.sbatch
#   All 18 rows (default):
#     sbatch jobs/fw_cross_eval.sbatch
# FW_XEVAL_GUARD (optional, e.g. B1_s1): evaluate that run on its own cache first and stop (exit 2)
# if its run.log FINAL target accuracy is not within the repeat min-max +- 1 cloud.
# Outputs are rewritten after every run; an output folder that already has results is refused.
#
# Runs on the GPU on purpose: the logged FINAL target numbers were produced on an A100 (TF32
# convolutions); a CPU evaluation can differ by a borderline cloud (jobs/pp_cross_eval.sbatch).

ROWS=${FW_XEVAL_ROWS:-"A1 A2 A3 A4 A5 A6 B1 B2 B3 B3b B4 B5 C1 C2 C3 C4 C5 C6"}
OUT=${FW_XEVAL_OUT:-results/fw_cross_eval}
GUARD_ARGS=${FW_XEVAL_GUARD:+--guard $FW_XEVAL_GUARD}

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
