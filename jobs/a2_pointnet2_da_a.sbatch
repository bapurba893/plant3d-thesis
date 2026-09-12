#!/bin/bash
#SBATCH --job-name=a2_pointnet2_da_a
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/a2_pointnet2_da_a_%j.log

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM              # unmodified upstream clone, used as a library only
                                                              # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
POINTNET2_ROOT=/home/pearl/25m0301/Pointnet_Pointnet2_pytorch  # unmodified upstream clone, used as a library only
                                                                # (commit eb64fe0b4c24055559cea26299cb485dcb43d8dd)
export DEFREC_ROOT
export POINTNET2_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"

cd "$REPO_ROOT"
python adapters/train_a2_pointnet2_da_a.py \
    --exp_name A2_pointnet2_da_a \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --wd 5e-5 \
    --dropout 0.5 \
    --disc_dropout 0.3 \
    --gamma 10.0 \
    --lambda_ent 0.1
