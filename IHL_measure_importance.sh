#!/bin/bash

GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS \
    IHL_measure_importance.py config_corruption_IHL_mitgiation.yaml
