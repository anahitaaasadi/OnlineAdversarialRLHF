
# Phi-2 RLHF (DPO + optional IHL) — Minimal Trainer

Train **microsoft/phi-2** with **DPO** on pairwise preferences, with optional:
- **Uncertainty-targeted adversarial flips** (simulated)
- **Filtering** by small policy margin
- **IHL (logit hinge)** unlearning on flagged-corrupt pairs
- **PEFT QLoRA** so it runs on a single GPU (<=12–16GB)

## Data format (JSONL)
Each line:
```json
{"prompt": "...", "pos": "preferred response", "neg": "rejected response"}
```
Put your file at `data/train.jsonl`.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Example run (no adversary, pure DPO):
python train_phi2_dpo.py --train data/train.jsonl --epochs 1 --lr 1e-4 --eta 0.1 --ihl-weight 0.0

# With adversary and filtering + small IHL:
python train_phi2_dpo.py --train data/train.jsonl --epochs 1 --lr 1e-4 --eta 0.1 \
  --flip-rate 0.3 --filter-tau 0.5 --ihl-weight 0.1 --save-dir runs/phi2_dpo
```

## Outputs
- Console metrics each step: DPO loss, IHL, margin stats.
- Saved adapter at `runs/.../adapter`. Load with PEFT on top of Phi-2.
