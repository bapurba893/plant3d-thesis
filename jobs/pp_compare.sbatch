#!/bin/bash
#SBATCH --job-name=pp_compare
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-26
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/pp_compare_%A_%a.log

# Block PP (docs/PREPROCESSING_COMPARISON_PLAN.md): row C1 (DGCNN, DA-0, ALL, L_cls+L_seg)
# retrained once per preprocessing pipeline x seed, all other settings fixed per plan section 4.
#
# Array index -> (pipeline, seed):   idx = pipeline_position * 3 + seed_position
#   0-2   P0   (seeds 1,2,3)          default cache, data/preprocessed/
#   3-5   P1   6-8   P2   9-11  P3   12-14 P4
#   15-17 P4a  18-20 P4b  21-23 P4c  24-26 P4d
#
# Submitting only part of the array: --array on the command line overrides the directive above.
#   P1-P4 only (12 tasks):        sbatch --array=3-14  jobs/pp_compare.sbatch
#   P4 ablations only (12 tasks): sbatch --array=15-26 jobs/pp_compare.sbatch
#   P0 seeds 2,3 only:            sbatch --array=1-2   jobs/pp_compare.sbatch
#   Mixed lists also work:        sbatch --array=1-2,3-14 jobs/pp_compare.sbatch
#   Limit concurrency with %N:    sbatch --array=3-14%4 jobs/pp_compare.sbatch
# Dry run (prints the mapping, trains nothing, no Slurm needed):
#   PP_DRY_RUN=1 bash jobs/pp_compare.sbatch
#
# Each task writes results/PP_<ID>_s<seed>/{run.log,model.pt} (via --exp_name) and its Slurm
# stdout to results/logs/pp_compare_<jobid>_<taskid>.log.

# TO BE CONFIRMED: file name of each variant's manifest inside data/preprocessed_variants/<ID>/.
# The laptop is still generating these caches; plan section 8 names it preprocessed_manifest.csv.
VARIANT_MANIFEST_NAME=preprocessed_manifest.csv

PIPELINES=(P0 P1 P2 P3 P4 P4a P4b P4c P4d)
SEEDS=(1 2 3)

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis

pp_task() {  # $1 = array index -> sets PIPELINE, SEED, EXP_NAME, MANIFEST_ARGS
    local idx=$1
    PIPELINE=${PIPELINES[$((idx / ${#SEEDS[@]}))]}
    SEED=${SEEDS[$((idx % ${#SEEDS[@]}))]}
    EXP_NAME=PP_${PIPELINE}_s${SEED}
    if [ "$PIPELINE" = "P0" ]; then
        MANIFEST_ARGS=()  # P0 = existing cache, script's default manifest path
    else
        MANIFEST_ARGS=(--manifest "data/preprocessed_variants/${PIPELINE}/${VARIANT_MANIFEST_NAME}")
    fi
}

if [ -n "$PP_DRY_RUN" ]; then
    printf "%-6s %-9s %-5s %-10s %s\n" index pipeline seed exp_name manifest
    for ((i = 0; i < ${#PIPELINES[@]} * ${#SEEDS[@]}; i++)); do
        pp_task "$i"
        printf "%-6s %-9s %-5s %-10s %s\n" "$i" "$PIPELINE" "$SEED" "$EXP_NAME" \
            "${MANIFEST_ARGS[1]:-(default) data/preprocessed/preprocessed_manifest.csv}"
    done
    exit 0
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: run via sbatch (job array), or set PP_DRY_RUN=1 to print the mapping" >&2
    exit 1
fi
pp_task "$SLURM_ARRAY_TASK_ID"
echo "array task $SLURM_ARRAY_TASK_ID -> pipeline=$PIPELINE seed=$SEED exp_name=$EXP_NAME"

# Fail fast instead of training on the wrong data.
if [ "$PIPELINE" != "P0" ] && [ ! -f "$REPO_ROOT/${MANIFEST_ARGS[1]}" ]; then
    echo "ERROR: manifest not found: $REPO_ROOT/${MANIFEST_ARGS[1]}" >&2
    exit 1
fi
# run.log is opened in append mode; refuse to mix two runs into one log.
if [ -e "$REPO_ROOT/results/$EXP_NAME/run.log" ]; then
    echo "ERROR: $REPO_ROOT/results/$EXP_NAME/run.log already exists; move it aside first" >&2
    exit 1
fi

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM  # unmodified upstream clone, used as a library only
                                                 # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
export DEFREC_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"

cd "$REPO_ROOT"
python adapters/train_c1_dgcnn_da0.py \
    --exp_name "$EXP_NAME" \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --wd 5e-5 \
    --dropout 0.5 \
    --seed "$SEED" \
    "${MANIFEST_ARGS[@]}"
