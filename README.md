
# Online RLHF with Adversarial Feedback (Toy Simulator)

This is a **minimal, runnable** PyTorch project that implements:

- Online preference data stream (Bradley–Terry ground-truth reward)
- **Uncertainty-targeting adversary** that flips labels at small margins
- **Filter** that flags likely-corrupted pairs using predicted margin/uncertainty
- **Hybrid update**:
  - **DPO loss** on predicted-clean pairs
  - **IHL (Inverted Hinge Loss)** unlearning on predicted-corrupt winners
- **Retroactive unlearning** pass over the history of flagged-corrupt samples

It’s a pedagogical scaffold to plug in your real model, prompts, and responses.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --steps 1000 --batch-size 256 --flip-rate 0.2 --ihl-weight 0.5
```

Key flags:
- `--flip-rate`: adversary's expected fraction of pairs it flips each round.
- `--ihl-weight`: weight for IHL unlearning loss.
- `--retro-K`: run retroactive unlearning every K steps (0 disables).

Outputs:
- Console logs with online metrics (clean AUROC proxy, policy reward, corruption rate).
- A `.jsonl` log under `runs/`.
# OnlineAdversarialRLHF
