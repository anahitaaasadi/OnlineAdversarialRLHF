cd /home/aasadi5/Codes/OnlineAdversarialRLHF && \
source /home/aasadi5/Codes/OnlineAdversarialRLHF/rlhf2/bin/activate && \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 IHL_finetune.py \
    model_family=phi \
    split=full \
    batch_size=4 \
    gradient_accumulation_steps=4 \
    lr=1e-5
