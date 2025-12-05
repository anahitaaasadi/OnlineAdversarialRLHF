source rlhf2/bin/activate

CKPT_RESULT="./llm_weights/ft_epoch5_lr0.0001_phi_forget10_wd0.01/checkpoint-62/eval_results/ds_size300/eval_log_aggregated.json"
RETAIN_RESULT="./llm_weights/ft_epoch5_lr1e-05_phi_full_wd0.01/eval_results/ds_size300/eval_log_aggregated.json"

SAVE_FILE="aggr_result.csv"

METHOD_NAME="gradient_ascent_unlearning"
SUBMITTED_BY="aasadi5"

if [ ! -f "$CKPT_RESULT" ]; then
    echo "ERROR: Checkpoint result file not found: $CKPT_RESULT"
    echo "Please run evaluation on the unlearned model first."
    exit 1
fi

if [ ! -f "$RETAIN_RESULT" ]; then
    echo "ERROR: Retain result file not found: $RETAIN_RESULT"
    echo "Please run evaluation on the baseline model first."
    echo ""
    echo "To generate the baseline evaluation, run:"
    echo "python IHL_evaluate.py --config-name eval_everything \\"
    echo "    model_path=./llm_weights/ft_epoch5_lr1e-05_phi_full_wd0.01 \\"
    echo "    base_model_path=microsoft/phi-1_5 \\"
    echo "    split=forget01_perturbed"
    exit 1
fi

python IHL_aggregate_eval_stat.py \
    retain_result="$RETAIN_RESULT" \
    ckpt_result="$CKPT_RESULT" \
    method_name="$METHOD_NAME" \
    submitted_by="$SUBMITTED_BY" \
    save_file="$SAVE_FILE"

echo ""
echo "Aggregation complete! Results saved to: $SAVE_FILE"
