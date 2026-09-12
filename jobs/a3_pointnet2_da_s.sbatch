#!/bin/bash
#SBATCH --job-name=a3_pointnet2_da_s
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/a3_pointnet2_da_s_%j.log

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
python adapters/train_a3_pointnet2_da_s.py \
    --exp_name A3_pointnet2_da_s \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --wd 5e-5 \
    --dropout 0.5 \
    --DefRec_dist volume_based_voxels \
    --num_regions 3 \
    --DefRec_weight 1.0
