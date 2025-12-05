#!/bin/bash

# Activate virtual environment
source /home/aasadi5/Codes/OnlineAdversarialRLHF/rlhf2/bin/activate

# Set GPUs to use (default: 0,1)
GPUS=${GPUS:-"0,1"}
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

# Default training parameters
MODEL_FAMILY=${MODEL_FAMILY:-"phi"}
SPLIT=${SPLIT:-"full"}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

# Run training
cd /home/aasadi5/Codes/OnlineAdversarialRLHF && \
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS \
    IHL_finetune.py \
    model_family=$MODEL_FAMILY \
    split=$SPLIT \
    batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=$GRAD_ACC_STEPS \
    lr=$LEARNING_RATE
