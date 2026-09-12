#!/bin/bash
#SBATCH --job-name=c5_dgcnn_da_o
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/c5_dgcnn_da_o_%j.log

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM  # unmodified upstream clone, used as a library only
                                                 # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
export DEFREC_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"

cd "$REPO_ROOT"
python adapters/train_c5_dgcnn_da_o.py \
    --exp_name C5_dgcnn_da_o \
    --epochs 100 \
    --batch_size 8 \
    --test_batch_size 16 \
    --lr 1e-3 \
    --wd 5e-5 \
    --dropout 0.5
