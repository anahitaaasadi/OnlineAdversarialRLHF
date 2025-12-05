source /home/aasadi5/Codes/OnlineAdversarialRLHF/rlhf2/bin/activate

GPUS=${GPUS:-"0,1"}
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

MODEL_FAMILY=${MODEL_FAMILY:-"phi"}
MODEL_PATH=${MODEL_PATH:-"./llm_weights/ft_epoch5_lr0.0001_phi_forget10_wd0.01/checkpoint-62"}
SPLIT=${SPLIT:-"forget01"}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
FORGET_LOSS=${FORGET_LOSS:-"grad_ascent"}
NUM_EPOCHS=${NUM_EPOCHS:-5}

cd /home/aasadi5/Codes/OnlineAdversarialRLHF && \
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NUM_GPUS \
    IHL_forget.py \
    model_family=$MODEL_FAMILY \
    model_path=$MODEL_PATH \
    split=$SPLIT \
    batch_size=$BATCH_SIZE \
    gradient_accumulation_steps=$GRAD_ACC_STEPS \
    lr=$LEARNING_RATE \
    forget_loss=$FORGET_LOSS \
    num_epochs=$NUM_EPOCHS
