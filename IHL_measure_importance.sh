#!/bin/bash

source /home/aasadi5/Codes/OnlineAdversarialRLHF/rlhf2/bin/activate

GPUS=${GPUS:-"0,1"}
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

MODEL_FAMILY=${MODEL_FAMILY:-"phi"}
MODEL_PATH=${MODEL_PATH:-"./llm_weights/ft_epoch5_lr0.0001_phi_forget10_wd0.01/checkpoint-62"}
SPLIT=${SPLIT:-"forget01"}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}

cd /home/aasadi5/Codes/OnlineAdversarialRLHF && \
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS \
    IHL_measure_importance.py \
    model_family=$MODEL_FAMILY \
    model_path=$MODEL_PATH \
    split=$SPLIT \
    batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=$GRAD_ACC_STEPS
