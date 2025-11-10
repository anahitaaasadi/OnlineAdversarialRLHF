# Corruption Detection Troubleshooting Guide

## Why Corruption Detection Might Not Work

### Problem: `corrupt_frac = 0.0` (No samples detected as corrupt)

This happens when **all margins are above the threshold** (`filter_tau`). 

### Understanding the Margin

The margin is calculated as:
```
margin = |log P(chosen|x) - log P(rejected|x)|
```

**At the start of training:**
- Model hasn't learned preferences yet
- Margins are very small (close to 0)
- **Solution:** Use HIGHER `filter_tau` (1.0 - 2.0) initially

**After some training:**
- Model learns to distinguish chosen from rejected
- Margins increase
- **Solution:** Use LOWER `filter_tau` (0.3 - 0.5)

## Quick Fix: Adjust `filter_tau`

### If corrupt_frac = 0.0:
```bash
# INCREASE filter_tau to detect more samples
--filter-tau 1.5   # Very lenient (detects ~40-60% as corrupt)
--filter-tau 1.0   # Moderate (detects ~20-40% as corrupt)
--filter-tau 0.5   # Strict (detects ~5-20% as corrupt)
```

### Goal:
```
corrupt_frac ≈ flip_frac
```
If you set `--flip-rate 0.3`, you want `corrupt_frac` around 0.25-0.35

## Recommended Starting Configurations

### 1. FIRST RUN (Model hasn't seen data)
```bash
python train_phi2_dpo.py \
    --train data/train.jsonl \
    --epochs 3 \
    --bsz 2 \
    --grad_accum 8 \
    --flip-rate 0.25 \
    --filter-tau 1.5 \        # HIGH - margins will be small initially
    --ihl-weight 0.5 \
    --ihl-margin 0.3 \
    --adv-dpo-weight 0.3 \
    --save-dir runs/phi2_ihl_v1
```

### 2. AFTER INITIAL TRAINING (Model learned some preferences)
```bash
python train_phi2_dpo.py \
    --train data/train.jsonl \
    --epochs 3 \
    --bsz 2 \
    --grad_accum 8 \
    --flip-rate 0.25 \
    --filter-tau 0.5 \        # LOWER - margins are larger now
    --ihl-weight 0.7 \        # Can be more aggressive
    --ihl-margin 0.2 \
    --ihl-adaptive \          # Use adaptive weighting
    --adv-dpo-weight 0.2 \
    --save-dir runs/phi2_ihl_v2
```

### 3. AGGRESSIVE UNLEARNING (High noise scenario)
```bash
python train_phi2_dpo.py \
    --train data/train.jsonl \
    --epochs 3 \
    --bsz 2 \
    --grad_accum 8 \
    --flip-rate 0.4 \         # More corruption
    --filter-tau 1.0 \
    --ihl-weight 1.0 \        # Strong unlearning
    --ihl-type reverse \      # Most aggressive
    --ihl-reverse-temp 1.5 \
    --ihl-adaptive \
    --use-agreement-filter \  # Extra filtering
    --adv-dpo-weight 0.2 \
    --save-dir runs/phi2_ihl_aggressive
```

## Monitoring During Training

Watch these metrics in the JSON logs:

### 1. Corruption Detection Working?
```json
{
  "flip_frac": 0.25,        // Ground truth: 25% flipped
  "corrupt_frac": 0.28,     // ✓ GOOD: Detected ~28% as corrupt
  "flip_count": 1,          // Number flipped in this batch
}
```

### 2. IHL Active?
```json
{
  "loss_dpo": 0.52,         // DPO loss
  "loss_ihl": 0.15,         // ✓ GOOD: IHL is non-zero
  "ihl_corrupt_margin_mean": 0.42  // Margin of detected corrupt samples
}
```

### 3. Not Breaking Clean Data?
```json
{
  "policy_accuracy": 0.68,  // ✓ GOOD: >0.5 means learning correct preferences
  "margin_mean": 0.35,      // ✓ GOOD: Positive means chosen > rejected
  "kl_total": 0.08          // ✓ GOOD: Low KL = not drifting too far
}
```

## If Still Not Detecting Corruption

### Check 1: Print First Batch Margins
Add this debug code after line 633 (in training loop):
```python
if step == 1:
    print(f"\n=== DEBUG: First Batch ===")
    print(f"Margins: {margin.cpu().numpy()}")
    print(f"Min margin: {margin.min().item():.4f}")
    print(f"Max margin: {margin.max().item():.4f}")
    print(f"Mean margin: {margin.mean().item():.4f}")
    print(f"Filter tau: {args.filter_tau}")
    print(f"Samples below tau: {(margin < args.filter_tau).sum().item()}")
    print(f"========================\n")
```

### Check 2: Verify Flipping is Happening
```python
if step == 1:
    print(f"Flip mask: {flip_mask.cpu().numpy()}")
    print(f"Number flipped: {flip_mask.sum().item()}")
```

### Check 3: Temporarily Force Detection
For testing, you can force higher detection by increasing tau:
```bash
--filter-tau 10.0  # Very high - will detect almost everything as corrupt
```
This verifies that the IHL machinery is working.

## Recommended Workflow

### Step 1: Quick Test (5-10 minutes)
```bash
chmod +x quick_test.sh
./quick_test.sh
```

Check the output:
- ✓ `flip_frac` > 0
- ✓ `corrupt_frac` > 0
- ✓ `loss_ihl` > 0
- ✓ `policy_accuracy` > 0.5

### Step 2: Tune `filter_tau`
If `corrupt_frac = 0`:
- Increase `--filter-tau` by 0.5 increments
- Rerun quick test
- Repeat until `corrupt_frac ≈ flip_frac`

### Step 3: Full Training
```bash
chmod +x run_ihl_training.sh
# Edit filter_tau in the script based on Step 2
./run_ihl_training.sh
```

### Step 4: Monitor & Adjust
Watch the logs:
- If `loss_ihl` stays at 0: increase `filter_tau`
- If `policy_accuracy` drops: reduce `ihl_weight`
- If `kl_total` too high: reduce `lr` or `ihl_weight`

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `corrupt_frac = 0` always | Increase `--filter-tau` (try 1.0, 1.5, 2.0) |
| `corrupt_frac = 1.0` always | Decrease `--filter-tau` (try 0.5, 0.3, 0.1) |
| `loss_ihl = 0` always | Check `corrupt_frac > 0` first, then increase `--ihl-weight` |
| `policy_accuracy` dropping | Reduce `--ihl-weight` or increase `--filter-tau` (fewer false positives) |
| Training too slow | Reduce `--bsz`, increase `--grad_accum`, use smaller `--max-len` |
| Out of memory | Enable `--ihl-adaptive` might help, or reduce batch size |

## Expected Timeline

With the provided data and parameters:

| Epoch | Expected Behavior |
|-------|-------------------|
| 1 | Margins ~0.2-0.5, corrupt detection starts working |
| 2 | Margins increase to ~0.5-1.0, better separation |
| 3 | Model stable, robust to flips, policy_accuracy >0.6 |

## Need More Help?

1. Share the output of `quick_test.sh`
2. Share first 10 lines of training log showing JSON metrics
3. Check if GPU memory is sufficient (4GB minimum for Phi-2 with 4-bit)
