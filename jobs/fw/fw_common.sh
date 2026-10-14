#!/bin/bash
# Shared body for the full-workflow (FW) rerun job files in jobs/fw/ -- sourced, not run directly.
#
# The caller sets, before sourcing:
#   ROW        row id, e.g. A1 (result folders are results/FW_<ROW>_s<seed>)
#   SCRIPT     training script under adapters/
#   ARGS       bash array of the row's own arguments (copied from that row's original job file)
#   SEEDS      bash array of seeds to run one after another (default 1 2 3); in a job array
#              (B3b) the seed is the array task id instead
#   NEEDS_KPCONV  1 for Block B rows (exports KPCONV_ROOT)
#
# Every run uses the soil-kept preprocessing (P4a) via --manifest. Per seed: skip if its run.log
# already has the FINAL target line; move an unfinished run.log aside; continue with the next
# seed on failure (same rules as jobs/pp_compare.sbatch / pp_softweight.sbatch).
#
# Dry run (prints what would run, trains nothing):  FW_DRY_RUN=1 bash jobs/fw/fw_<row>.sbatch

MANIFEST=data/preprocessed_variants/P4a/preprocessed_manifest.csv
REPO_ROOT=/home/pearl/25m0301/plant3d-thesis

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SEEDS=("$SLURM_ARRAY_TASK_ID")
elif [ -z "${SEEDS+x}" ]; then
    SEEDS=(1 2 3)
fi

seed_done() {  # $1 = exp_name; true if its run.log has the FINAL target line
    grep -q "FINAL target (Pheno4D held-out) test accuracy" "$REPO_ROOT/results/$1/run.log" 2>/dev/null
}

if [ -n "$FW_DRY_RUN" ]; then
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME=FW_${ROW}_s${SEED}
        if seed_done "$EXP_NAME"; then st="done(skip)"; else st="run"; fi
        echo "$EXP_NAME [$st]: python -u adapters/$SCRIPT --exp_name $EXP_NAME --epochs 100" \
             "${ARGS[*]} --seed $SEED --manifest $MANIFEST"
    done
    exit 0
fi

if [ ! -f "$REPO_ROOT/$MANIFEST" ]; then
    echo "ERROR: manifest not found: $REPO_ROOT/$MANIFEST" >&2
    exit 1
fi

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

DEFREC_ROOT=/home/pearl/25m0301/DefRec_and_PCM   # unmodified upstream clone, used as a library only
                                                  # (commit 5cb7b797099b0d44abcd00860669a5f95812ae22)
export DEFREC_ROOT
export PYTHONPATH="$DEFREC_ROOT:$PYTHONPATH"
if [ "$NEEDS_KPCONV" = 1 ]; then
    KPCONV_ROOT=/home/pearl/25m0301/KPConv-PyTorch  # sibling repo (commit d19c575d3fa9fcfd5a74845b5b27aac7e50472c7)
    export KPCONV_ROOT
fi

cd "$REPO_ROOT"
FAILED=()
for SEED in "${SEEDS[@]}"; do
    EXP_NAME=FW_${ROW}_s${SEED}
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
    python -u "adapters/$SCRIPT" \
        --exp_name "$EXP_NAME" \
        --epochs 100 \
        "${ARGS[@]}" \
        --seed "$SEED" \
        --manifest "$MANIFEST"
    rc=$?
    if [ $rc -eq 0 ] && seed_done "$EXP_NAME"; then
        echo "===== $(date '+%F %T') $EXP_NAME: OK ====="
    else
        echo "===== $(date '+%F %T') $EXP_NAME: FAILED (exit code $rc) -- continuing =====" >&2
        FAILED+=("$EXP_NAME")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "===== $ROW done, FAILED: ${FAILED[*]} =====" >&2
    exit 1
fi
echo "===== $ROW done, all seeds OK or skipped ====="
