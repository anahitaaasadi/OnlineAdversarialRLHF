#!/bin/bash

# =============================================================================
# QUICK TEST RUN - Verify corruption detection is working
# =============================================================================
# This uses a small subset to quickly verify that:
# 1. Corruption detection identifies flipped samples
# 2. IHL loss is being applied
# 3. Metrics are being logged correctly
# =============================================================================

echo "================================================================================"
echo "QUICK TEST: Corruption Detection Verification"
echo "================================================================================"

python train_phi2_dpo.py \
    --train data/train.jsonl \
    --epochs 1 \
    --bsz 4 \
    --grad_accum 2 \
    --max-len 512 \
    --seed 42 \
    --lr 5e-5 \
    --eta 0.1 \
    --flip-rate 0.3 \
    --flip-strategy uncertainty \
    --ihl-weight 0.5 \
    --ihl-type hinge \
    --ihl-margin 0.3 \
    --filter-tau 1.0 \
    --adv-dpo-weight 0.3 \
    --save-dir runs/test_ihl \
    2>&1 | tee runs/test_ihl.log

echo ""
echo "================================================================================"
echo "Check the log above for:"
echo "  1. 'flip_frac' should be ~0.3 (30% as specified)"
echo "  2. 'corrupt_frac' should be >0 (detection working)"
echo "  3. 'loss_ihl' should be >0 when corrupt_frac >0 (unlearning active)"
echo "  4. 'policy_accuracy' should stay relatively high (not breaking clean data)"
echo "================================================================================"
