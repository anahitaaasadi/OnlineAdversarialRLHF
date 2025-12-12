for i in $(seq 1 2); do
    # Get samples, rank them into preference pairs, get uncertainty scores.
    python data.py config_corruption_mitgation.yaml
    
    # Do DPO training.
    accelerate launch --config_file ds_zero3.yaml DPO.py config_corruption_mitgation.yaml
done