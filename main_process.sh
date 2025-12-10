#!/bin/bash

# Activate the rlhf2 virtual environment
source rlhf2/bin/activate

for i in $(seq 1 2); do
    echo "===== Iteration $i: Generating preference samples ====="
    python data.py config_corruption_mitigation.yaml
    
    echo "===== Iteration $i: Running DPO training ====="
    accelerate launch --config_file ds_zero3.yaml DPO.py config_corruption_mitigation.yaml
    
    # Check if corrupted samples were detected
    FORGET_FILE="data/forget_samples/corrupted_samples_iter_${i}.json"
    if [ -f "$FORGET_FILE" ]; then
        echo "===== Iteration $i: Running IHL forget on corrupted samples ====="
        
        # Convert corrupted samples to IHL forget format
        python prepare_forget_data.py $i
        
        # Get the latest checkpoint
        CHECKPOINT_DIR="checkpoints/iter_DPO_mitigation"
        LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "final_checkpoint_*" -type d | sort -V | tail -n 1)
        
        if [ -z "$LATEST_CHECKPOINT" ]; then
            echo "Warning: No checkpoint found, using base model"
            LATEST_CHECKPOINT=$(grep "ref_model:" config_corruption_mitigation.yaml | cut -d'"' -f2)
        fi
        
        echo "Using model: $LATEST_CHECKPOINT"
        
        # Set up save directory for forgotten model
        FORGET_SAVE_DIR="llm_weights/forget_corrupted_iter_${i}"
        
        # Run IHL forget with the corrupted samples
        export MODEL_PATH="$LATEST_CHECKPOINT"
        export SPLIT="corrupted_iter_${i}"
        export DATA_PATH="data/forget_samples/forget_data_iter_${i}.json"
        export SAVE_DIR="$FORGET_SAVE_DIR"
        bash IHL_forget.sh
        
        # Update checkpoint to use the forgotten model for next iteration
        if [ -d "$FORGET_SAVE_DIR" ]; then
            echo "Updating checkpoint reference to forgotten model: $FORGET_SAVE_DIR"
            # The next DPO iteration will automatically pick up the latest checkpoint
        fi
        
        echo "===== Iteration $i: IHL forget completed ====="
    else
        echo "===== Iteration $i: No corrupted samples detected, skipping IHL forget ====="
    fi
    
    echo "===== Iteration $i completed ====="
    echo ""
done