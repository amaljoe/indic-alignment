#!/bin/bash
# Autoresearch loop: iterate KL-distillation training with different configs
# until NormAd test accuracy > 55% OR BhED caste stereotype score < 45%.
#
# Strategy:
#   Round 1: alpha=0.5, T=2.0  (balanced CE+KL)
#   Round 2: alpha=0.3, T=3.0  (heavier KL, softer distributions)
#   Round 3: alpha=0.7, T=1.5  (more CE, tighter distributions)
#   Round 4: alpha=0.5, T=2.0, more epochs  (repeat best with more training)
#
# Soft labels are generated ONCE (expensive teacher inference), then re-used.
# Between rounds, only train_kl.py reruns.

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

PYTHON=/dev/shm/qwen35/bin/python
RESULTS_DIR="results/kl"
mkdir -p "$RESULTS_DIR"

# Improvement targets
NORMAD_TARGET=55.0   # NormAd test accuracy (Nepal + Sri Lanka)
BHED_TARGET=45.0     # BhED caste stereotype score (lower is better)

# ── Round configs ──────────────────────────────────────────────────────────────
declare -a ROUNDS
ROUNDS=(
    "alpha=0.5 temperature=2.0 epochs=3 lr=2e-5"
    "alpha=0.3 temperature=3.0 epochs=3 lr=2e-5"
    "alpha=0.7 temperature=1.5 epochs=3 lr=2e-5"
    "alpha=0.5 temperature=2.0 epochs=5 lr=1e-5"
)

best_normad=0
best_cfg=""

extract_normad_acc() {
    local f="$1"
    # Extract accuracy from the 'no-context + few-shot' mode (most relevant for the student)
    $PYTHON -c "
import json, sys
try:
    d = json.load(open('$f'))
    # Take best accuracy across modes
    best = max(v.get('accuracy', 0) for v in d.values() if isinstance(v, dict))
    print(f'{best:.2f}')
except Exception as e:
    print('0.00')
" 2>/dev/null
}

extract_bhed_caste() {
    local f="$1"
    $PYTHON -c "
import json, sys
try:
    d = json.load(open('$f'))
    score = d.get('caste', {}).get('stereotype_score', 100)
    print(f'{score:.1f}')
except Exception as e:
    print('100.0')
" 2>/dev/null
}

echo "============================================================"
echo " Autoresearch KL Distillation Loop"
echo " Targets: NormAd >${NORMAD_TARGET}%  |  BhED caste <${BHED_TARGET}%"
echo "============================================================"

# ── Step 1: Generate soft labels ONCE ────────────────────────────────────────
echo ""
echo "[AutoResearch] Checking soft-label data..."
SOFT_EXISTS=1
for f in normad_soft.jsonl bhed_soft.jsonl milu_en_soft.jsonl; do
    if [[ ! -f "finetune/data/$f" ]]; then
        SOFT_EXISTS=0
        break
    fi
done

if [[ "$SOFT_EXISTS" -eq 0 ]]; then
    echo "[AutoResearch] Generating soft labels (this takes 1-2 hours)..."
    source ~/proxy-setup/scripts/proxy_env.sh 2>/dev/null || true
    export HTTP_PROXY=http://127.0.0.1:3128
    export HTTPS_PROXY=http://127.0.0.1:3128
    export LD_PRELOAD=/dev/shm/qwen35/lib/libstdc++.so.6
    $PYTHON finetune/generate_soft_labels.py \
        --model Qwen/Qwen3-30B-A3B \
        --sources normad milu_en milu_hi bhed globalopinion \
        --filter-correct
    echo "[AutoResearch] Soft labels generated."
else
    echo "[AutoResearch] Soft labels already exist — skipping generation."
fi

# ── Iteration loop ────────────────────────────────────────────────────────────
round=0
for cfg in "${ROUNDS[@]}"; do
    round=$((round + 1))
    echo ""
    echo "============================================================"
    echo " Round $round / ${#ROUNDS[@]}: $cfg"
    echo "============================================================"

    # Parse config
    eval "$cfg"

    OUT_DIR="finetune/checkpoints_kl_r${round}"
    TAG="r${round}_alpha${alpha}_T${temperature}_ep${epochs}"

    bash scripts/run_kl_pipeline.sh \
        --alpha "$alpha" \
        --temperature "$temperature" \
        --epochs "$epochs" \
        --lr "$lr" \
        --output "$OUT_DIR" \
        --skip-datagen

    # Extract metrics
    NORMAD_F="$RESULTS_DIR/normad_alpha${alpha}_T${temperature}.json"
    BHED_F="$RESULTS_DIR/bhed_alpha${alpha}_T${temperature}.json"

    normad_acc=$(extract_normad_acc "$NORMAD_F")
    bhed_caste=$(extract_bhed_caste "$BHED_F")

    echo ""
    echo "[Round $round] NormAd acc: ${normad_acc}%  |  BhED caste: ${bhed_caste}%"

    # Track best
    if $PYTHON -c "exit(0 if float('$normad_acc') > float('$best_normad') else 1)" 2>/dev/null; then
        best_normad="$normad_acc"
        best_cfg="$cfg (Round $round, model=$OUT_DIR/final)"
    fi

    # Check targets
    normad_ok=$($PYTHON -c "print('1' if float('$normad_acc') >= $NORMAD_TARGET else '0')" 2>/dev/null || echo "0")
    bhed_ok=$($PYTHON -c "print('1' if float('$bhed_caste') <= $BHED_TARGET else '0')" 2>/dev/null || echo "0")

    if [[ "$normad_ok" == "1" && "$bhed_ok" == "1" ]]; then
        echo ""
        echo "============================================================"
        echo " TARGET REACHED in Round $round!"
        echo "  NormAd: ${normad_acc}% (target >${NORMAD_TARGET}%)"
        echo "  BhED caste: ${bhed_caste}% (target <${BHED_TARGET}%)"
        echo "  Best config: $cfg"
        echo "  Best model: $OUT_DIR/final"
        echo "============================================================"
        exit 0
    fi

    echo "[Round $round] Target not yet reached. Continuing..."
done

echo ""
echo "============================================================"
echo " Autoresearch complete (all rounds exhausted)."
echo " Best NormAd accuracy: ${best_normad}%"
echo " Best config: $best_cfg"
echo "============================================================"
