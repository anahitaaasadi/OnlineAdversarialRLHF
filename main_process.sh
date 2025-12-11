for i in $(seq 1 2); do
    python data.py config_corruption_mitigation.yaml
    accelerate launch --config_file ds_zero3.yaml DPO.py config_corruption_mitigation.yaml
done