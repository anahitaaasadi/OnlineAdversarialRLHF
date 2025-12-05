find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0,1

source rlhf2/bin/activate
python IHL_evaluate.py \
    --config-name eval_everything \
    model_path=./llm_weights/ft_epoch5_lr0.0001_phi_forget10_wd0.01/checkpoint-62 \
    base_model_path=microsoft/phi-1_5 \
    split=forget01_perturbed
