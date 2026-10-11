#!/bin/bash
#SBATCH --job-name=pp_compare
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/pp_compare_%A_%a.log

# Block PP (docs/PREPROCESSING_COMPARISON_PLAN.md): row C1 (DGCNN, DA-0, ALL, L_cls+L_seg)
# retrained once per preprocessing pipeline x seed, all other settings fixed per plan section 4.
#
# One array task = one PIPELINE; the task runs seeds 1, 2, 3 one after another (about 11 min
# each, so ~35 min per task). This layout exists because the dgx QOS allows at most 5
# submitted jobs per user (MaxSubmitPU=5, MaxJobsPU=4), and every array task counts as a job.
#
# Array index -> pipeline:
#   0 P0 (default cache, data/preprocessed/)   1 P1   2 P2   3 P3   4 P4
#   5 P4a   6 P4b   7 P4c   8 P4d
#
# Submit in two rounds (5-job limit):
#   round 1: sbatch --array=0-4 jobs/pp_compare.sbatch
#   round 2: sbatch --array=5-8 jobs/pp_compare.sbatch   (after round 1 has left the queue)
# Dry run (prints the mapping and each seed's current status, trains nothing):
#   PP_DRY_RUN=1 bash jobs/pp_compare.sbatch
#
# Per seed:
#   - results/PP_<ID>_s<seed>/run.log already has the "FINAL target" line -> seed skipped
#     (already done, e.g. the P4 seed-1 test run, job 347914_12).
#   - run.log exists but is unfinished (earlier crash/timeout) -> moved aside to
#     run.log.incomplete_<timestamp> so the new run does not append into it.
#   - a failing seed is logged and the loop moves on to the next seed; the task exits
#     non-zero at the end if any seed failed, so sacct shows it as FAILED.
#
# Each run writes results/PP_<ID>_s<seed>/{run.log,model.pt} (via --exp_name); the task's
# Slurm stdout (all 3 seeds) goes to results/logs/pp_compare_<jobid>_<index>.log.

# Variant manifest file name inside data/preprocessed_variants/<ID>/ (confirmed 2026-10-11:
# all 8 variant caches ship preprocessed_manifest.csv).
VARIANT_MANIFEST_NAME=preprocessed_manifest.csv

PIPELINES=(P0 P1 P2 P3 P4 P4a P4b P4c P4d)
SEEDS=(1 2 3)

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis

pp_pipeline() {  # $1 = array index -> sets PIPELINE, MANIFEST_ARGS
    PIPELINE=${PIPELINES[$1]}
    if [ "$PIPELINE" = "P0" ]; then
        MANIFEST_ARGS=()  # P0 = existing cache, script's default manifest path
    else
        MANIFEST_ARGS=(--manifest "data/preprocessed_variants/${PIPELINE}/${VARIANT_MANIFEST_NAME}")
    fi
}

seed_done() {  # $1 = exp_name; true if its run.log has the FINAL target line
    grep -q "FINAL target (Pheno4D held-out) test accuracy" "$REPO_ROOT/results/$1/run.log" 2>/dev/null
}

if [ -n "$PP_DRY_RUN" ]; then
    printf "%-6s %-9s %-7s %-40s %s\n" index pipeline seeds "status (s1 s2 s3)" manifest
    for ((i = 0; i < ${#PIPELINES[@]}; i++)); do
        pp_pipeline "$i"
        status=""
        for SEED in "${SEEDS[@]}"; do
            if seed_done "PP_${PIPELINE}_s${SEED}"; then status+="s$SEED=done(skip) "
            else status+="s$SEED=run "; fi
        done
        printf "%-6s %-9s %-7s %-40s %s\n" "$i" "$PIPELINE" "1,2,3" "$status" \
            "${MANIFEST_ARGS[1]:-(default) data/preprocessed/preprocessed_manifest.csv}"
    done
    exit 0
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: run via sbatch (job array), or set PP_DRY_RUN=1 to print the mapping" >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#PIPELINES[@]}" ]; then
    echo "ERROR: array index $SLURM_ARRAY_TASK_ID out of range 0-$((${#PIPELINES[@]} - 1))" >&2
    exit 1
fi
pp_pipeline "$SLURM_ARRAY_TASK_ID"
echo "array task $SLURM_ARRAY_TASK_ID -> pipeline=$PIPELINE seeds=${SEEDS[*]}"

# Fail fast instead of training on the wrong data.
if [ "$PIPELINE" != "P0" ] && [ ! -f "$REPO_ROOT/${MANIFEST_ARGS[1]}" ]; then
    echo "ERROR: manifest not found: $REPO_ROOT/${MANIFEST_ARGS[1]}" >&2
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
    EXP_NAME=PP_${PIPELINE}_s${SEED}
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
    python adapters/train_c1_dgcnn_da0.py \
        --exp_name "$EXP_NAME" \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-3 \
        --wd 5e-5 \
        --dropout 0.5 \
        --seed "$SEED" \
        "${MANIFEST_ARGS[@]}"
    rc=$?
    if [ $rc -eq 0 ] && seed_done "$EXP_NAME"; then
        echo "===== $(date '+%F %T') $EXP_NAME: OK ====="
    else
        echo "===== $(date '+%F %T') $EXP_NAME: FAILED (exit code $rc) -- continuing with next seed =====" >&2
        FAILED+=("$EXP_NAME")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "===== task $SLURM_ARRAY_TASK_ID ($PIPELINE) done, FAILED seeds: ${FAILED[*]} =====" >&2
    exit 1
fi
echo "===== task $SLURM_ARRAY_TASK_ID ($PIPELINE) done, all seeds OK or skipped ====="
