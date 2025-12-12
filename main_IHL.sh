
for i in $(seq 2 2); do
    # Get samples, rank them into preference pairs, get uncertainty scores.
    python data.py config_corruption_IHL_mitgiation.yaml
    # Do DPO training.
    accelerate launch --config_file ds_zero3.yaml DPO.py config_corruption_IHL_mitgiation.yaml
    # Do IHL training on forget set.
    # Order of calling should be  - IHL_measure_importance > IHL_forget.py 
    # Don't have enough space on disk so skipping FILA.
    if [[ "$i" -lt 2 ]]; then
        ./IHL_forget.sh
    fi
done