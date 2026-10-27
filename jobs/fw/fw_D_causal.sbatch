#!/bin/bash
#SBATCH --job-name=fw_D_causal
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1-3
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_D_causal_%A_%a.log

# Block D rerun with CAUSAL growth-curve inputs (step_notes/FW_5_Results.md): D2, D3, D4, D5,
# D5 ablation, D6, D6 ablation. Each scan's growth-curve params are fitted on its plant's strictly
# earlier scans only (data/traits/pheno4d_growth_curves_causal.csv); scans without a fit get the
# per-species training-plant mean plus a 0/1 fit-available column; D5/D6 start their ODE params
# from the training plants' mean only. D1 has no growth-curve input and is not rerun.
# One seed per array task (1-3).
#
# Inputs reused unchanged from job 359157: the FW B3b features data/features_fw/P4a_B3b_s<seed>/
# (no feature extraction here -- the job stops for a seed whose extract.log is not complete).
# data/traits/pheno4d_growth_curves_causal.csv must exist before submitting (written once by
# `python scripts/growth_curves.py --causal`, not by this job, so the three tasks never write it).
#
# Results: results/FW_D_causal_s<seed>/<row>/ (row = D2, D3, D4, D5, D5_ablation, D6,
# D6_ablation). Job 359157's results (results/FW_D<n>_s<seed>/, FW_D5/D6_ablation_s<seed>/) are
# never written. D5/D6 and their ablations warm-start from results/FW_D_causal_s<seed>/D4/model.pt.
# The GPU is requested only because the dgx partition is the one this account uses; every step
# here runs on the CPU (as in 359157).
#
# Skip/resume rules (as fw_D.sbatch): a row is skipped if its run.log already has the
# "=== FINAL test metrics" line; an unfinished run.log is moved aside; on failure the job continues.
#
# Dry run (prints what would run, runs nothing):  FW_DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash jobs/fw/fw_D_causal.sbatch

REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
CAUSAL_CSV=data/traits/pheno4d_growth_curves_causal.csv
N_SCANS=223

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SEEDS=("$SLURM_ARRAY_TASK_ID")
else
    SEEDS=(1 2 3)
fi

# name|script|extra args (D5/D6 rows get --init_from_d4 added below)
D_ROWS=(
    "D2|train_d2_fusion_growth_curves.py|"
    "D3|train_d3_fusion_temporal.py|"
    "D4|train_d4_fusion_previous_stage.py|"
    "D5|train_d5_fusion_physics_logistic.py|"
    "D5_ablation|train_d5_fusion_physics_logistic.py|--lambda_phys 0 --lambda_mono 0"
    "D6|train_d6_fusion_physics_gompertz.py|"
    "D6_ablation|train_d6_fusion_physics_gompertz.py|--lambda_phys 0 --lambda_mono 0"
)

feat_dir() { echo "data/features_fw/P4a_B3b_s$1"; }
out_dir() { echo "results/FW_D_causal_s$1"; }
extract_done() {  # $1 = seed
    grep -q "wrote $N_SCANS feature .npz files" "$REPO_ROOT/$(feat_dir "$1")/extract.log" 2>/dev/null
}
d_done() {  # $1 = seed, $2 = row name
    grep -q "=== FINAL test metrics" "$REPO_ROOT/$(out_dir "$1")/$2/run.log" 2>/dev/null
}
needs_d4() { [[ "$1" == D5* || "$1" == D6* ]]; }

d_cmd() {  # $1 = seed, $2 = row name, $3 = script, $4 = extra args
    local cmd="python -u adapters/$3 --out_path $(out_dir "$1") --exp_name $2"
    cmd="$cmd --features_dir $(feat_dir "$1") --seed $1"
    [ -n "$4" ] && cmd="$cmd $4"
    needs_d4 "$2" && cmd="$cmd --init_from_d4 $(out_dir "$1")/D4/model.pt"
    echo "$cmd"
}

if [ -n "$FW_DRY_RUN" ]; then
    [ -f "$REPO_ROOT/$CAUSAL_CSV" ] || echo "WARNING: $CAUSAL_CSV not found"
    for SEED in "${SEEDS[@]}"; do
        extract_done "$SEED" || echo "WARNING: features for seed $SEED not complete: $(feat_dir "$SEED")"
        for row in "${D_ROWS[@]}"; do
            IFS='|' read -r name script extra <<< "$row"
            if d_done "$SEED" "$name"; then st="done(skip)"; else st="run"; fi
            echo "$(out_dir "$SEED")/$name [$st]: $(d_cmd "$SEED" "$name" "$script" "$extra")"
        done
    done
    exit 0
fi

if [ ! -f "$REPO_ROOT/$CAUSAL_CSV" ]; then
    echo "ERROR: $CAUSAL_CSV not found -- run: python scripts/growth_curves.py --causal" >&2
    exit 1
fi

source /home/pearl/25m0301/exit/etc/profile.d/conda.sh
conda activate plant3d

cd "$REPO_ROOT"
FAILED=()
for SEED in "${SEEDS[@]}"; do
    if ! extract_done "$SEED"; then
        echo "===== seed $SEED: FAILED, features not complete: $(feat_dir "$SEED") =====" >&2
        FAILED+=("FW_D_causal_s${SEED}_features")
        continue
    fi
    for row in "${D_ROWS[@]}"; do
        IFS='|' read -r name script extra <<< "$row"
        TAG="$(out_dir "$SEED")/$name"
        RUN_LOG="$REPO_ROOT/$TAG/run.log"
        echo "===== $(date '+%F %T') $TAG: start ====="
        if d_done "$SEED" "$name"; then
            echo "===== $TAG: SKIPPED, run.log already has the FINAL test line ====="
            continue
        fi
        if needs_d4 "$name" && ! d_done "$SEED" D4; then
            echo "===== $TAG: FAILED, $(out_dir "$SEED")/D4 did not finish (no warm start) =====" >&2
            FAILED+=("$TAG")
            continue
        fi
        if [ -e "$RUN_LOG" ]; then
            aside="$RUN_LOG.incomplete_$(date +%Y%m%d_%H%M%S)"
            mv "$RUN_LOG" "$aside"
            echo "===== $TAG: unfinished run.log moved aside to $aside ====="
        fi
        $(d_cmd "$SEED" "$name" "$script" "$extra")
        rc=$?
        if [ $rc -eq 0 ] && d_done "$SEED" "$name"; then
            echo "===== $(date '+%F %T') $TAG: OK ====="
        else
            echo "===== $(date '+%F %T') $TAG: FAILED (exit code $rc) -- continuing =====" >&2
            FAILED+=("$TAG")
        fi
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "===== D causal done, FAILED: ${FAILED[*]} =====" >&2
    exit 1
fi
echo "===== D causal done, all steps OK or skipped ====="
