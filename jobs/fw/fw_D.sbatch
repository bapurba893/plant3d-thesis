#!/bin/bash
#SBATCH --job-name=fw_D
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=1-3
#SBATCH --output=/home/pearl/25m0301/plant3d-thesis/results/logs/fw_D_%A_%a.log

# Full-workflow rerun, Block D (D1-D6 + the two no-physics ablations) on features from the FW
# B3b backbone (KPConv, DA-S, ALL, trained on the soil-kept P4a preprocessing).
# One seed per array task (1-3): backbone = results/FW_B3b_s<seed>/model.pt.
#
# Per seed:
#   1. GPU: extract B3b's frozen pooled features for all 223 Pheno4D scans, read through the P4a
#      manifest, into data/features_fw/P4a_B3b_s<seed>/ (adapters/extract_b3b_features.py).
#   2. CPU (as the original D runs): D1, D2, D3, D4, D5, D5 ablation, D6, D6 ablation, each with
#      --features_dir pointing to that folder, results in results/FW_D<n>_s<seed>/
#      (ablations: FW_D5_ablation_s<seed>, FW_D6_ablation_s<seed>). D5/D6 and their ablations
#      warm-start from this seed's results/FW_D4_s<seed>/model.pt via --init_from_d4; they are
#      skipped (counted as failed) if FW_D4 did not finish, because the scripts silently skip a
#      missing warm-start file.
#
# Trait targets stay FIXED: every D script still reads the original data/traits/*.csv (no
# --traits_dir; infer_segmentation.py is not rerun). The original Block D inputs
# (data/features/Pheno4D/, data/traits/) and results (results/D*/) are never written.
# D arguments = the originals' (all defaults; ablations add --lambda_phys 0 --lambda_mono 0), plus
# --seed <seed> as in every other FW row.
#
# Skip/resume rules (as jobs/fw/fw_common.sh): a step is skipped if already finished (features:
# extract.log has "wrote 223 feature .npz files"; D rows: run.log has the "=== FINAL test metrics"
# line); an unfinished log is moved aside; on failure the job continues with the next step.
#
# Dry run (prints what would run, runs nothing):  FW_DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash jobs/fw/fw_D.sbatch
# (without SLURM_ARRAY_TASK_ID the dry run lists all three seeds).

MANIFEST=data/preprocessed_variants/P4a/preprocessed_manifest.csv
REPO_ROOT=/home/pearl/25m0301/plant3d-thesis
N_SCANS=223

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SEEDS=("$SLURM_ARRAY_TASK_ID")
else
    SEEDS=(1 2 3)
fi

# name|script|extra args (D5/D6 rows get --init_from_d4 added below)
D_ROWS=(
    "D1|train_d1_fusion_baseline.py|"
    "D2|train_d2_fusion_growth_curves.py|"
    "D3|train_d3_fusion_temporal.py|"
    "D4|train_d4_fusion_previous_stage.py|"
    "D5|train_d5_fusion_physics_logistic.py|"
    "D5_ablation|train_d5_fusion_physics_logistic.py|--lambda_phys 0 --lambda_mono 0"
    "D6|train_d6_fusion_physics_gompertz.py|"
    "D6_ablation|train_d6_fusion_physics_gompertz.py|--lambda_phys 0 --lambda_mono 0"
)

feat_dir() { echo "data/features_fw/P4a_B3b_s$1"; }
ckpt() { echo "results/FW_B3b_s$1/model.pt"; }
extract_done() {  # $1 = seed
    grep -q "wrote $N_SCANS feature .npz files" "$REPO_ROOT/$(feat_dir "$1")/extract.log" 2>/dev/null
}
d_done() {  # $1 = exp_name
    grep -q "=== FINAL test metrics" "$REPO_ROOT/results/$1/run.log" 2>/dev/null
}
needs_d4() { [[ "$1" == D5* || "$1" == D6* ]]; }

extract_cmd() {  # $1 = seed
    echo "python -u adapters/extract_b3b_features.py --checkpoint $(ckpt "$1") --gpu 0" \
         "--manifest $MANIFEST --out_dir $(feat_dir "$1")"
}
d_cmd() {  # $1 = seed, $2 = row name, $3 = script, $4 = extra args
    local cmd="python -u adapters/$3 --exp_name FW_$2_s$1 --features_dir $(feat_dir "$1") --seed $1"
    [ -n "$4" ] && cmd="$cmd $4"
    needs_d4 "$2" && cmd="$cmd --init_from_d4 results/FW_D4_s$1/model.pt"
    echo "$cmd"
}

if [ -n "$FW_DRY_RUN" ]; then
    for SEED in "${SEEDS[@]}"; do
        if extract_done "$SEED"; then st="done(skip)"; else st="run"; fi
        echo "FW_D_s$SEED features [$st]: $(extract_cmd "$SEED")"
        [ -f "$REPO_ROOT/$(ckpt "$SEED")" ] || echo "  WARNING: checkpoint not found: $(ckpt "$SEED")"
        for row in "${D_ROWS[@]}"; do
            IFS='|' read -r name script extra <<< "$row"
            if d_done "FW_${name}_s$SEED"; then st="done(skip)"; else st="run"; fi
            echo "FW_${name}_s$SEED [$st]: $(d_cmd "$SEED" "$name" "$script" "$extra")"
        done
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
KPCONV_ROOT=/home/pearl/25m0301/KPConv-PyTorch    # sibling repo (commit d19c575d3fa9fcfd5a74845b5b27aac7e50472c7)
export KPCONV_ROOT

cd "$REPO_ROOT"
FAILED=()
for SEED in "${SEEDS[@]}"; do
    FEAT_DIR=$(feat_dir "$SEED")
    EXTRACT_LOG="$REPO_ROOT/$FEAT_DIR/extract.log"

    # ---- 1. feature extraction (GPU) ----
    echo "===== $(date '+%F %T') FW_D_s$SEED features: start ====="
    if extract_done "$SEED"; then
        echo "===== FW_D_s$SEED features: SKIPPED, extract.log already complete ====="
    elif [ ! -f "$REPO_ROOT/$(ckpt "$SEED")" ]; then
        echo "===== FW_D_s$SEED features: FAILED, checkpoint not found: $(ckpt "$SEED") =====" >&2
        FAILED+=("FW_D_s${SEED}_features")
        continue
    else
        mkdir -p "$REPO_ROOT/$FEAT_DIR"
        if [ -e "$EXTRACT_LOG" ]; then
            aside="$EXTRACT_LOG.incomplete_$(date +%Y%m%d_%H%M%S)"
            mv "$EXTRACT_LOG" "$aside"
            echo "===== FW_D_s$SEED features: unfinished extract.log moved aside to $aside ====="
        fi
        $(extract_cmd "$SEED") 2>&1 | tee "$EXTRACT_LOG"
        if extract_done "$SEED"; then
            echo "===== $(date '+%F %T') FW_D_s$SEED features: OK ====="
        else
            echo "===== $(date '+%F %T') FW_D_s$SEED features: FAILED -- skipping this seed's D rows =====" >&2
            FAILED+=("FW_D_s${SEED}_features")
            continue
        fi
    fi

    # ---- 2. D1-D6 + ablations (CPU) ----
    for row in "${D_ROWS[@]}"; do
        IFS='|' read -r name script extra <<< "$row"
        EXP_NAME=FW_${name}_s${SEED}
        RUN_LOG="$REPO_ROOT/results/$EXP_NAME/run.log"
        echo "===== $(date '+%F %T') $EXP_NAME: start ====="
        if d_done "$EXP_NAME"; then
            echo "===== $EXP_NAME: SKIPPED, run.log already has the FINAL test line ====="
            continue
        fi
        if needs_d4 "$name" && ! d_done "FW_D4_s$SEED"; then
            echo "===== $EXP_NAME: FAILED, FW_D4_s$SEED did not finish (no warm start) =====" >&2
            FAILED+=("$EXP_NAME")
            continue
        fi
        if [ -e "$RUN_LOG" ]; then
            aside="$RUN_LOG.incomplete_$(date +%Y%m%d_%H%M%S)"
            mv "$RUN_LOG" "$aside"
            echo "===== $EXP_NAME: unfinished run.log moved aside to $aside ====="
        fi
        $(d_cmd "$SEED" "$name" "$script" "$extra")
        rc=$?
        if [ $rc -eq 0 ] && d_done "$EXP_NAME"; then
            echo "===== $(date '+%F %T') $EXP_NAME: OK ====="
        else
            echo "===== $(date '+%F %T') $EXP_NAME: FAILED (exit code $rc) -- continuing =====" >&2
            FAILED+=("$EXP_NAME")
        fi
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "===== D done, FAILED: ${FAILED[*]} =====" >&2
    exit 1
fi
echo "===== D done, all steps OK or skipped ====="
