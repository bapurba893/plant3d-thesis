#!/bin/bash
#SBATCH --job-name=pp_softw
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/pp_softweight_%A_%a.log

# Block PP soil-weight experiment: row C1 (DGCNN, DA-0, ALL, L_cls+L_seg) on the soil-kept P4a
# cache, with a LEARNED weight on Crops3D maize soil/pot points' L_wCE term
# (adapters/train_c1_dgcnn_da0_softw.py, adapters/soil_weight.py). All other settings identical
# to jobs/pp_compare.sbatch, so results compare directly with results/PP_P4a_s{1,2,3}
# (which is the --soil_weight none case, so it is not rerun here).
#
# One array task = one SETTING; the task runs seeds 1, 2, 3 one after another (same layout as
# pp_compare.sbatch, for the dgx QOS job limit).
#
# Array index -> setting:
#   0 learned_scalar   1 learned_features
#
# Submit:   sbatch jobs/pp_softweight.sbatch
# Dry run (prints the mapping and each seed's status, trains nothing):
#   PP_DRY_RUN=1 bash jobs/pp_softweight.sbatch
#
# Per seed: results/PP_SW_<setting>_s<seed>/{run.log,model.pt,soil_weight.pt}. Same skip /
# move-aside / continue-on-failure rules as pp_compare.sbatch.

SETTINGS=(learned_scalar learned_features)
SEEDS=(1 2 3)
MANIFEST=data/preprocessed_variants/P4a/preprocessed_manifest.csv

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis

seed_done() {  # $1 = exp_name; true if its run.log has the FINAL target line
    grep -q "FINAL target (Pheno4D held-out) test accuracy" "$REPO_ROOT/results/$1/run.log" 2>/dev/null
}

if [ -n "$PP_DRY_RUN" ]; then
    printf "%-6s %-17s %-40s %s\n" index setting "status (s1 s2 s3)" manifest
    for ((i = 0; i < ${#SETTINGS[@]}; i++)); do
        status=""
        for SEED in "${SEEDS[@]}"; do
            if seed_done "PP_SW_${SETTINGS[$i]}_s${SEED}"; then status+="s$SEED=done(skip) "
            else status+="s$SEED=run "; fi
        done
        printf "%-6s %-17s %-40s %s\n" "$i" "${SETTINGS[$i]}" "$status" "$MANIFEST"
    done
    exit 0
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: run via sbatch (job array), or set PP_DRY_RUN=1 to print the mapping" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SETTINGS[@]}" ]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID out of range 0-$((${#SETTINGS[@]} - 1))" >&2
    exit 1
fi
SETTING=${SETTINGS[$SLURM_ARRAY_TASK_ID]}
echo "array task $SLURM_ARRAY_TASK_ID -> soil_weight=$SETTING seeds=${SEEDS[*]}"

if [ ! -f "$REPO_ROOT/$MANIFEST" ]; then
    echo "ERROR: manifest not found: $REPO_ROOT/$MANIFEST" >&2
    exit 1
fi

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM  # unmodified upstream clone, used as a library only
                                                 # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
export DEFREC_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"

cd "$REPO_ROOT"
FAILED=()
for SEED in "${SEEDS[@]}"; do
    EXP_NAME=PP_SW_${SETTING}_s${SEED}
    RUN_LOG="$REPO_ROOT/results/$EXP_NAME/run.log"
    echo "===== $(date '+%F %T') $EXP_NAME: start ====="
    if seed_done "$EXP_NAME"; then
        echo "===== $EXP_NAME: SKIPPED, run.log already has the FINAL target line ====="
        continue
    fi
    if [ -e "$RUN_LOG" ]; then
        aside="$RUN_LOG.incomplete_$(date +%Y%m%d_%H%M%S)"
        mv "$RUN_LOG" "$aside"
        echo "===== $EXP_NAME: unfinished run.log moved aside to $aside ====="
    fi
    python adapters/train_c1_dgcnn_da0_softw.py \
        --exp_name "$EXP_NAME" \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-3 \
        --wd 5e-5 \
        --dropout 0.5 \
        --seed "$SEED" \
        --manifest "$MANIFEST" \
        --soil_weight "$SETTING"
    rc=$?
    if [ $rc -eq 0 ] && seed_done "$EXP_NAME"; then
        echo "===== $(date '+%F %T') $EXP_NAME: OK ====="
    else
        echo "===== $(date '+%F %T') $EXP_NAME: FAILED (exit code $rc) -- continuing with next seed =====" >&2
        FAILED+=("$EXP_NAME")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "===== task $SLURM_ARRAY_TASK_ID ($SETTING) done, FAILED seeds: ${FAILED[*]} =====" >&2
    exit 1
fi
echo "===== task $SLURM_ARRAY_TASK_ID ($SETTING) done, all seeds OK or skipped ====="
