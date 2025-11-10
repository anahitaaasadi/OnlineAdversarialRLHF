#!/bin/bash

# =============================================================================
# IHL UNLEARNING TRAINING SCRIPT
# =============================================================================
# This script runs adversarial RLHF training with IHL unlearning.
# The parameters are chosen to ensure good corruption detection and unlearning.
# =============================================================================

# Basic configuration
TRAIN_DATA="data/train.jsonl"
SAVE_DIR="runs/phi2_dpo_ihl"
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
MAX_LEN=512
SEED=42
LR=5e-5

# DPO parameters
ETA=0.1  # KL penalty (beta in DPO paper)

# Adversary simulation - START WITH MODERATE FLIP RATE
FLIP_RATE=0.25  # 25% of samples will be flipped
FLIP_STRATEGY="uncertainty"  # Target uncertain examples (most realistic)

# IHL unlearning parameters - BALANCED STARTING POINT
IHL_WEIGHT=0.5  # Moderate unlearning strength
IHL_TYPE="hinge"  # Default hinge loss
IHL_MARGIN=0.3  # Moderately aggressive margin
IHL_ADAPTIVE=""  # Add "--ihl-adaptive" to enable adaptive weighting

# Corruption detection - MODERATE THRESHOLD
FILTER_TAU=1.0  # Start higher to detect more samples as corrupt
# Lower values (0.3-0.5) are stricter, higher (0.8-1.5) are more lenient

# DPO weight for corrupt samples
ADV_DPO_WEIGHT=0.3  # Down-weight corrupt samples in DPO

# Agreement filter (optional)
USE_AGREEMENT=""  # Add "--use-agreement-filter" to enable

echo "================================================================================"
echo "Starting IHL Unlearning Training"
echo "================================================================================"
echo "Configuration:"
echo "  Data: $TRAIN_DATA"
echo "  Epochs: $EPOCHS | Batch size: $BATCH_SIZE | Grad accum: $GRAD_ACCUM"
echo "  Learning rate: $LR | Max length: $MAX_LEN"
echo ""
echo "Adversary:"
echo "  Flip rate: $FLIP_RATE ($((FLIP_RATE * 100))% of samples)"
echo "  Strategy: $FLIP_STRATEGY"
echo ""
echo "IHL Unlearning:"
echo "  Weight: $IHL_WEIGHT | Type: $IHL_TYPE | Margin: $IHL_MARGIN"
echo "  Adaptive: $([ -z "$IHL_ADAPTIVE" ] && echo "NO" || echo "YES")"
echo ""
echo "Corruption Detection:"
echo "  Filter tau: $FILTER_TAU"
echo "  Agreement filter: $([ -z "$USE_AGREEMENT" ] && echo "NO" || echo "YES")"
echo "  Corrupt DPO weight: $ADV_DPO_WEIGHT"
echo "================================================================================"
echo ""

python train_phi2_dpo.py \
    --train "$TRAIN_DATA" \
    --epochs $EPOCHS \
    --bsz $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --max-len $MAX_LEN \
    --seed $SEED \
    --lr $LR \
    --eta $ETA \
    --flip-rate $FLIP_RATE \
    --flip-strategy $FLIP_STRATEGY \
    --ihl-weight $IHL_WEIGHT \
    --ihl-type $IHL_TYPE \
    --ihl-margin $IHL_MARGIN \
    $IHL_ADAPTIVE \
    --filter-tau $FILTER_TAU \
    $USE_AGREEMENT \
    --adv-dpo-weight $ADV_DPO_WEIGHT \
    --save-dir "$SAVE_DIR" \
    2>&1 | tee "${SAVE_DIR}/training.log"

echo ""
echo "================================================================================"
echo "Training completed! Results saved to: $SAVE_DIR"
echo "================================================================================"
