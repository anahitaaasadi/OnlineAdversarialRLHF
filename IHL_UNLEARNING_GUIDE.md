# IHL Unlearning for Adversarial RLHF

## Overview

This implementation provides a robust training framework for language models that can handle adversarial label flipping in preference feedback using **Inverse Hinge Loss (IHL)** for unlearning.

## What is IHL Unlearning?

**Inverse Hinge Loss (IHL)** is designed to "unlearn" the effects of adversarial label flips in preference learning. When an adversary flips labels (making the rejected response appear as the winner), standard training would reinforce the wrong preference. IHL counters this by:

1. **Detecting** likely corrupted samples (via margin-based or agreement-based filtering)
2. **Applying IHL** to the corrupt subset to reduce/reverse the adversarial signal
3. **Continuing DPO** on all samples (with down-weighting for corrupt ones)

### How It Works

**Standard Training (Vulnerable):**
- Adversary flips: `chosen ↔ rejected` 
- Model learns: `P(rejected|x) > P(chosen|x)` ❌
- Result: Model learns wrong preferences

**IHL Unlearning (Robust):**
- Adversary flips: `chosen ↔ rejected`
- Detection identifies flip as "corrupt"
- IHL penalizes: `margin(P_obs_win, P_obs_lose)` 
- Result: Model resists adversarial signal ✓

## IHL Loss Types

### 1. Hinge Loss (Default)
```
L_IHL = [margin + log P(y_obs_win) - log P(y_obs_lose)]_+
```
- **When to use:** General purpose, balanced unlearning
- **Parameters:** `--ihl-margin` (0.1-0.5, lower = stronger)
- **Behavior:** Reduces margin between observed winner/loser

### 2. Squared Loss
```
L_IHL = (log P(y_obs_win) - log P(y_obs_lose))²
```
- **When to use:** Want smoother gradients
- **Parameters:** None required
- **Behavior:** Encourages equal probabilities for flipped pairs

### 3. Reverse Ranking
```
L_IHL = log(1 + exp(-T × (log P(y_obs_lose) - log P(y_obs_win))))
```
- **When to use:** Aggressive unlearning needed
- **Parameters:** `--ihl-reverse-temp` (0.5-2.0)
- **Behavior:** Actively tries to reverse the corrupted preference

## Usage Examples

### Basic IHL Training
```bash
python train_phi2_dpo.py \
    --train data/train.jsonl \
    --epochs 3 \
    --bsz 4 \
    --flip-rate 0.2 \
    --ihl-weight 0.5 \
    --ihl-type hinge \
    --ihl-margin 0.3
```

### Aggressive Unlearning
```bash
python train_phi2_dpo.py \
    --train data/train.jsonl \
    --flip-rate 0.3 \
    --flip-strategy uncertainty \
    --ihl-weight 1.0 \
    --ihl-type reverse \
    --ihl-reverse-temp 1.5 \
    --ihl-adaptive \
    --filter-tau 0.3
```

### Conservative (High Precision Detection)
```bash
python train_phi2_dpo.py \
    --train data/train.jsonl \
    --flip-rate 0.15 \
    --ihl-weight 0.3 \
    --ihl-type squared \
    --filter-tau 0.2 \
    --use-agreement-filter
```

## Key Parameters

### IHL Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ihl-weight` | 0.0 | Weight for IHL loss (0=disabled, higher=stronger) |
| `--ihl-type` | hinge | Loss type: `hinge`, `squared`, `reverse` |
| `--ihl-margin` | 0.5 | Target margin for hinge loss (lower=more aggressive) |
| `--ihl-adaptive` | False | Enable adaptive weighting by margin |
| `--ihl-reverse-temp` | 1.0 | Temperature for reverse ranking |

### Adversary Simulation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--flip-rate` | 0.0 | Fraction of labels to flip (0.0-1.0) |
| `--flip-strategy` | uncertainty | Strategy: `uncertainty`, `random`, `confidence` |

### Corruption Detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--filter-tau` | 0.5 | Margin threshold (lower=stricter filtering) |
| `--use-agreement-filter` | False | Also filter by policy-reference disagreement |
| `--adv-dpo-weight` | 0.5 | Weight for corrupt samples in DPO (0-1) |

## Adversary Strategies

### 1. Uncertainty-Targeted (Default)
Flips labels where model is most uncertain (smallest margins).
- **Most realistic:** Adversaries target ambiguous examples
- **Most damaging:** Uncertain examples have highest learning impact
```bash
--flip-strategy uncertainty
```

### 2. Confidence-Targeted
Flips labels where model is most confident (largest margins).
- **Corrupts anchors:** Damages reliable training signal
- **Harder to detect:** High-confidence flips contradict strong beliefs
```bash
--flip-strategy confidence
```

### 3. Random
Randomly flips labels without targeting.
- **Baseline comparison:** Measures targeted vs. non-targeted attacks
```bash
--flip-strategy random
```

## Monitoring Unlearning Progress

Key metrics to track:

### Unlearning Effectiveness
- `loss_ihl`: Should decrease → unlearning working
- `ihl_corrupt_margin_mean`: Should increase → model becoming more robust
- `policy_accuracy`: Should stay high → not harming clean data

### Corruption Detection
- `corrupt_frac`: Fraction detected as corrupt
- `flip_frac`: Actual flip rate (ground truth in simulation)
- Goal: `corrupt_frac ≈ flip_frac` (accurate detection)

### Model Drift
- `kl_total`: KL divergence from reference
- Monitor to ensure not drifting too far from base model
- High KL + high IHL weight → reduce one or both

## Tuning Guidelines

### Start Conservative
```bash
--ihl-weight 0.3 \
--ihl-type hinge \
--ihl-margin 0.5 \
--filter-tau 0.5
```

### If Unlearning Insufficient
1. Increase `--ihl-weight` (0.3 → 0.5 → 1.0)
2. Decrease `--ihl-margin` (0.5 → 0.3 → 0.1)
3. Switch to `--ihl-type reverse`
4. Enable `--ihl-adaptive`

### If Hurting Clean Performance
1. Decrease `--ihl-weight` (1.0 → 0.5 → 0.3)
2. Increase `--filter-tau` (more selective detection)
3. Add `--use-agreement-filter` (stricter filtering)
4. Increase `--adv-dpo-weight` (less down-weighting)

### If High False Positive Detection
1. Decrease `--filter-tau` (stricter margin threshold)
2. Enable `--use-agreement-filter`
3. Switch to `--flip-strategy confidence` (easier to detect)

## Advanced: Adaptive Weighting

When `--ihl-adaptive` is enabled, IHL loss is weighted per-sample based on corruption confidence:

```
weight(margin) = 0.1 + 0.9 × sigmoid(-5 × (margin/tau - 1))
```

- Low margin → high weight (likely corrupt)
- High margin → low weight (likely clean)
- Smooth transition around `filter-tau`

**Benefits:**
- Focuses unlearning on most uncertain examples
- Reduces harm to borderline-but-clean samples
- Generally improves robustness-performance tradeoff

**When to use:**
- High flip rates (> 20%)
- Noisy margin estimates
- Want to minimize clean data harm

## Validation Strategy

1. **Track metrics per epoch:**
   - Save all JSON logs
   - Plot `loss_ihl`, `policy_accuracy`, `corrupt_frac` over time

2. **Evaluate on clean held-out set:**
   - Test without any flips
   - Ensure performance doesn't degrade

3. **Evaluate on adversarial held-out set:**
   - Test with known flip rate
   - Measure robustness improvement

4. **Compare strategies:**
   - Baseline (no IHL): `--ihl-weight 0.0`
   - Your IHL config
   - Measure accuracy difference under attack

## Theoretical Justification

**Why does IHL work?**

1. **Corrupted signal:** When adversary flips, observed data suggests `P(neg|x) > P(pos|x)`
2. **Standard loss:** Pushes model to increase `P(neg|x)`, decrease `P(pos|x)` ❌
3. **IHL:** Penalizes high `P(observed_win|x)` on corrupt subset
4. **Effect:** Reduces adversarial gradient, preserving true preference knowledge ✓

**Key insight:** Don't need to know the *true* label, just need to identify corruption and suppress its influence.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{adversarial_rlhf_ihl,
  title={Adversarial RLHF with IHL Unlearning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/OnlineAdversarialRLHF}
}
```

## Troubleshooting

**Q: Loss not decreasing?**
- Check `--ihl-weight` not too high (try 0.3-0.5)
- Verify `--filter-tau` detecting some corrupt samples
- Ensure `--flip-rate > 0` for simulation

**Q: Performance degrading on clean data?**
- Reduce `--ihl-weight`
- Increase `--filter-tau` (fewer false positives)
- Use `--ihl-type squared` (gentler)

**Q: Not robust to attacks?**
- Increase `--ihl-weight`
- Decrease `--ihl-margin`
- Try `--ihl-adaptive` and `--ihl-type reverse`

**Q: High KL divergence?**
- Increase `--eta` (stronger KL penalty)
- Reduce `--ihl-weight`
- Reduce `--lr`
