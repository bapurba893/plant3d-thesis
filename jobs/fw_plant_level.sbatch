#!/bin/bash
#SBATCH --job-name=fw_plant
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=1-6
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_plant_level_%A_%a.log

# Plant-level evaluation: re-evaluate the 54 selected FW checkpoints (18 rows x 3 seeds) with
# scripts/16_fw_cross_eval.py, which now also writes one line per held-out cloud
# (fw_cross_eval_clouds.csv: run, seed, cloud id, plant id, species, true/predicted label).
# No training. PointNet++ and KPConv: 3 evaluation repeats per cache; DGCNN: 1.
# Each task writes to its own folder (the script uses fixed file names per --out_dir):
#   task 1: A1 A2 A3   (guard on A1 seed 1 first)
#   task 2: A4 A5 A6
#   task 3: B1 B2      (guard on B1 seed 1 first)
#   task 4: B3 B3b
#   task 5: B4 B5
#   task 6: C1 C2 C3 C4 C5 C6
# Then, on the login node (CPU, no model): python scripts/17_fw_plant_level.py --clouds
#   results/fw_plant_level/<CACHE_TAG>/task*/fw_cross_eval_clouds.csv
#
# Environment overrides (sbatch --export=ALL,...):
#   PL_CACHES  target caches for --cache (default "P4a P4"; e.g. "P4a P4s" for a new cache)
#   PL_TAG     results subfolder name (default: the caches joined by "_", e.g. P4a_P4)
#   PL_ROWS / PL_OUT  evaluate only these rows into this folder (single-row test), e.g.
#     sbatch --array=3 --export=ALL,PL_ROWS=B1,PL_OUT=results/fw_plant_level/test_B1 jobs/fw_plant_level.sbatch
#
# Full run:  sbatch jobs/fw_plant_level.sbatch

CACHES=${PL_CACHES:-"P4a P4"}
TAG=${PL_TAG:-$(echo $CACHES | tr ' ' '_')}
BASE=results/fw_plant_level/$TAG
case "$SLURM_ARRAY_TASK_ID" in
  1) ROWS="A1 A2 A3";          GUARD_ARGS="--guard A1_s1" ;;
  2) ROWS="A4 A5 A6";          GUARD_ARGS="" ;;
  3) ROWS="B1 B2";             GUARD_ARGS="--guard B1_s1" ;;
  4) ROWS="B3 B3b";            GUARD_ARGS="" ;;
  5) ROWS="B4 B5";             GUARD_ARGS="" ;;
  6) ROWS="C1 C2 C3 C4 C5 C6"; GUARD_ARGS="" ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
OUT=$BASE/task${SLURM_ARRAY_TASK_ID}_$(echo $ROWS | tr ' ' '_')
if [ -n "$PL_ROWS" ]; then
    ROWS=$PL_ROWS
    OUT=${PL_OUT:?PL_OUT must be set together with PL_ROWS}
    GUARD_ARGS=""
fi
# the guard checks the own cache (P4a); skip it when P4a is not evaluated
[[ " $CACHES " == *" P4a "* ]] || GUARD_ARGS=""

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM   # unmodified upstream clone, used as a library only
                                                  # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
export DEFREC_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"
KPCONV_ROOT=/home/pearl/25m0301/KPConv-PyTorch    # sibling repo (commit d19c575d3fa9fcfd5a74845b5b27aac7e50472c7)
export KPCONV_ROOT
POINTNET2_ROOT=/home/pearl/25m0301/Pointnet_Pointnet2_pytorch  # sibling repo (commit eb64fe0b4c24055559cea26299cb485dcb43d8dd)
export POINTNET2_ROOT

cd /home/pearl/25m0301/plant3d-thesis
echo "task $SLURM_ARRAY_TASK_ID: rows $ROWS; caches $CACHES; out $OUT; $(date '+%F %T')"
python -u scripts/16_fw_cross_eval.py --rows $ROWS --cache $CACHES --gpu 0 --out_dir "$OUT" $GUARD_ARGS
rc=$?
echo "task $SLURM_ARRAY_TASK_ID finished with exit code $rc; $(date '+%F %T')"
exit $rc
