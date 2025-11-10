# Quick Start Guide

## 🚀 How to Run IHL Unlearning Training

### Problem: Corruption Not Being Detected?

The key issue is usually the **`filter_tau`** parameter. At the start of training, margins between chosen/rejected are small, so you need a HIGHER threshold to detect corruption.

### ✅ Recommended Approach

#### Option 1: Quick Test (Recommended First)
```bash
./quick_test.sh
```
This runs a short test to verify everything works. Look for:
- ✓ `flip_frac` = 0.3 (30% of samples flipped)
- ✓ `corrupt_frac` > 0 (detection working!)
- ✓ `loss_ihl` > 0 (unlearning active!)

**If `corrupt_frac = 0`:** Edit `quick_test.sh` and increase `--filter-tau` from 1.0 to 1.5 or 2.0

#### Option 2: Full Training
```bash
./run_ihl_training.sh
```
This runs the full 3-epoch training with optimal parameters.

#### Option 3: Manual Command
```bash
python train_phi2_dpo.py \
    --train data/train.jsonl \
    --epochs 3 \
    --bsz 2 \
    --grad_accum 8 \
    --flip-rate 0.25 \
    --filter-tau 1.5 \
    --ihl-weight 0.5 \
    --ihl-margin 0.3 \
    --save-dir runs/phi2_ihl
```

## 🔧 Key Parameters Explained

### Must Tune: `--filter-tau`
Controls corruption detection sensitivity:
- **1.5-2.0**: Lenient (detects 40-60% as corrupt) - USE THIS FIRST
- **1.0**: Moderate (detects 20-40% as corrupt)
- **0.3-0.5**: Strict (detects 5-20% as corrupt)

**Rule of thumb:** Start HIGH (1.5), then decrease as model learns

### Adversary Settings
- `--flip-rate 0.25`: 25% of samples will be flipped
- `--flip-strategy uncertainty`: Target uncertain examples (most realistic)

### IHL Unlearning
- `--ihl-weight 0.5`: Moderate unlearning strength
- `--ihl-type hinge`: Default loss (use `reverse` for aggressive)
- `--ihl-margin 0.3`: Target margin for unlearning

## 📊 What to Watch During Training

The script logs JSON metrics every 10 steps. Key metrics:

```json
{
  "flip_frac": 0.25,        // ✓ Should match --flip-rate
  "corrupt_frac": 0.28,     // ✓ Should be >0 and ≈ flip_frac
  "loss_ihl": 0.15,         // ✓ Should be >0 when corrupt_frac >0
  "policy_accuracy": 0.65,  // ✓ Should stay >0.5
  "margin_mean": 0.35       // ✓ Should increase over time
}
```

## 🐛 Troubleshooting

### Issue: `corrupt_frac = 0.0`
**Solution:** Increase `--filter-tau`
```bash
# Try these values in order:
--filter-tau 1.5
--filter-tau 2.0
--filter-tau 2.5
```

### Issue: `loss_ihl = 0.0`
**Cause:** No corrupt samples detected
**Solution:** Fix corrupt_frac first (see above)

### Issue: `policy_accuracy` dropping below 0.5
**Cause:** Too aggressive unlearning
**Solution:** Reduce `--ihl-weight` to 0.3 or 0.2

### Issue: Out of memory
**Solution:**
```bash
--bsz 1              # Reduce batch size
--grad_accum 16      # Increase gradient accumulation
--max-len 256        # Reduce sequence length
```

## 📁 Output Files

After training, you'll find:
```
runs/phi2_ihl/
├── adapter_epoch_1/        # LoRA weights after epoch 1
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_state.pt   # Optimizer state for resuming
├── adapter_epoch_2/
├── adapter_epoch_3/
└── training.log            # Full training log with all metrics
```

## 🎯 Expected Results

With good parameters:
- **Epoch 1:** Margins ~0.2-0.5, corruption detection starts
- **Epoch 2:** Margins ~0.5-1.0, better robustness
- **Epoch 3:** Stable model, `policy_accuracy` >0.6

## 💡 Pro Tips

1. **Always run quick_test.sh first** to verify corruption detection
2. **Start with HIGH filter_tau** (1.5-2.0), then decrease
3. **Monitor `corrupt_frac ≈ flip_frac`** for good detection
4. **Watch `policy_accuracy` >0.5** to ensure not breaking model
5. **Use adaptive weighting** (`--ihl-adaptive`) for better results

## 📚 More Information

- Full documentation: `IHL_UNLEARNING_GUIDE.md`
- Troubleshooting: `CORRUPTION_DETECTION_FIX.md`
- Theory: See docstring in `train_phi2_dpo.py`

## 🤝 Need Help?

If corruption detection still doesn't work:
1. Run `quick_test.sh` and share the output
2. Check the first few JSON log lines
3. Try `--filter-tau 2.0` or even `3.0`
