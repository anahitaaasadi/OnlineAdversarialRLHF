
import os, json, math, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ap = argparse.ArgumentParser()
ap.add_argument("--train", type=str, required=True)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bsz", type=int, default=4)
ap.add_argument("--grad_accum", type=int, default=4)
ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--eta", type=float, default=0.1)

# IHL (Inverse Hinge Loss) unlearning parameters
ap.add_argument("--ihl-weight", type=float, default=0.0,
                help="Weight for IHL unlearning loss (0.0 = disabled, higher = stronger unlearning)")
ap.add_argument("--ihl-margin", type=float, default=0.5,
                help="Target margin for IHL hinge loss (smaller = more aggressive unlearning)")
ap.add_argument("--ihl-type", type=str, default="hinge", choices=["hinge", "squared", "reverse"],
                help="Type of IHL loss: hinge (default), squared (smoother), or reverse (aggressive)")
ap.add_argument("--ihl-adaptive", action="store_true",
                help="Use adaptive IHL weighting based on margin (prioritize uncertain examples)")
ap.add_argument("--ihl-reverse-temp", type=float, default=1.0,
                help="Temperature for reverse ranking IHL (only used with --ihl-type=reverse)")

# Adversary simulation parameters
ap.add_argument("--flip-rate", type=float, default=0.0,
                help="Fraction of labels to flip adversarially (0.0-1.0)")
ap.add_argument("--flip-strategy", type=str, default="uncertainty", 
                choices=["uncertainty", "random", "confidence"],
                help="Adversary strategy: uncertainty (target ambiguous), random, or confidence (target certain)")

# Filtering and robustness parameters
ap.add_argument("--filter-tau", type=float, default=0.5,
                help="Margin threshold for filtering corrupt samples (lower = stricter)")
ap.add_argument("--use-agreement-filter", action="store_true",
                help="Also filter based on policy-reference disagreement")
ap.add_argument("--adv-dpo-weight", type=float, default=0.5,
            help="Relative weight for adversarial/corrupt pairs in DPO (1.0 = equal).")

# General training parameters
ap.add_argument("--save-dir", type=str, default="runs/phi2_dpo")
args = ap.parse_args()

set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Print training configuration
print("=" * 80)
print("ADVERSARIAL RLHF WITH IHL UNLEARNING - Training Configuration")
print("=" * 80)
print(f"Device: {device}")
print(f"Training data: {args.train}")
print(f"Epochs: {args.epochs} | Batch size: {args.bsz} | Grad accumulation: {args.grad_accum}")
print(f"Learning rate: {args.lr} | Max length: {args.max_len}")
print(f"Seed: {args.seed}")
print("\n--- DPO Parameters ---")
print(f"Beta (eta): {args.eta}")
print(f"Adversarial weight in DPO: {args.adv_dpo_weight}")
print("\n--- Adversary Simulation ---")
print(f"Flip rate: {args.flip_rate} ({args.flip_rate*100:.1f}% of samples)")
print(f"Flip strategy: {args.flip_strategy}")
print("\n--- IHL Unlearning ---")
if args.ihl_weight > 0.0:
    print(f"IHL enabled: YES")
    print(f"  IHL weight: {args.ihl_weight}")
    print(f"  IHL type: {args.ihl_type}")
    print(f"  IHL margin: {args.ihl_margin}")
    print(f"  Adaptive weighting: {args.ihl_adaptive}")
    if args.ihl_type == "reverse":
        print(f"  Reverse temperature: {args.ihl_reverse_temp}")
else:
    print(f"IHL enabled: NO (--ihl-weight is 0)")
print("\n--- Corruption Detection ---")
print(f"Margin threshold (tau): {args.filter_tau}")
print(f"Agreement filter: {args.use_agreement_filter}")
print(f"\nSave directory: {args.save_dir}")
print("=" * 80 + "\n")

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Phi-2 in 4-bit...")
# Reference model (frozen)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

# --- load models ---
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_cfg,
    device_map="auto",
)

# Important for 4-bit fine-tuning
model = prepare_model_for_kbit_training(model)

# LoRA config (Phi-2 module names)
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["Wqkv","fc1","fc2","Wout"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap with LoRA
model = get_peft_model(model, peft_cfg)

# Optional but helpful diagnostics
model.print_trainable_parameters()

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Disable cache during training to avoid warnings
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# Reference model stays frozen, no PEFT
ref_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_cfg,
    device_map="auto",
)
ref_model.requires_grad_(False)

ds = PairDataset(args.train, tokenizer, max_len=args.max_len)
dl = DataLoader(ds, batch_size=args.bsz, shuffle=True, 
                collate_fn=lambda b: batchify(b, tokenizer, args.max_len),
                num_workers=0, pin_memory=torch.cuda.is_available())

opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

# Use automatic mixed precision for faster training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

os.makedirs(args.save_dir, exist_ok=True)
step = 0

for epoch in range(args.epochs):
    model.train()  # Ensure model is in training mode
    for batch in tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}"):
        step += 1
        
        # Move all tensors to device at once (more efficient)
        c_ids = batch["c_ids"].to(device, non_blocking=True)
        c_attn = batch["c_attn"].to(device, non_blocking=True)
        c_mask = batch["c_resp_mask"].to(device, non_blocking=True)
        r_ids = batch["r_ids"].to(device, non_blocking=True)
        r_attn = batch["r_attn"].to(device, non_blocking=True)
        r_mask = batch["r_resp_mask"].to(device, non_blocking=True)

        # Use automatic mixed precision if available
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Policy logprobs (with gradients)
            lp_c_pi = seq_logprobs(model, c_ids, c_attn, c_mask, requires_grad=True)
            lp_r_pi = seq_logprobs(model, r_ids, r_attn, r_mask, requires_grad=True)
            
            # Reference logprobs (no gradients)
            with torch.no_grad():
                lp_c_ref = seq_logprobs(ref_model, c_ids, c_attn, c_mask, requires_grad=False)
                lp_r_ref = seq_logprobs(ref_model, r_ids, r_attn, r_mask, requires_grad=False)
        
        # Simulated adversary: choose flips based on strategy
        with torch.no_grad():
            if args.flip_strategy == "uncertainty":
                flip_mask = flip_labels_uncertainty_targeting_text(lp_c_pi, lp_r_pi, args.flip_rate)
            elif args.flip_strategy == "random":
                flip_mask = flip_labels_random(lp_c_pi, lp_r_pi, args.flip_rate)
            elif args.flip_strategy == "confidence":
                flip_mask = flip_labels_confidence_targeting(lp_c_pi, lp_r_pi, args.flip_rate)
            else:
                flip_mask = flip_labels_uncertainty_targeting_text(lp_c_pi, lp_r_pi, args.flip_rate)

        # Use automatic mixed precision if available
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Observed winners/losers for BOTH policy and reference (respect flips)
            obs_win_pi   = torch.where(flip_mask, lp_r_pi, lp_c_pi)
            obs_lose_pi  = torch.where(flip_mask, lp_c_pi, lp_r_pi)
            obs_win_ref  = torch.where(flip_mask, lp_r_ref, lp_c_ref)
            obs_lose_ref = torch.where(flip_mask, lp_c_ref, lp_r_ref)

            # Margin-based partition for corruption detection
            with torch.no_grad():
                clean_mask, corrupt_mask, margin = filter_by_margin(lp_c_pi, lp_r_pi, args.filter_tau)
                
                # Optional: also use agreement-based filtering
                if args.use_agreement_filter:
                    agree_mask, disagree_mask = filter_by_agreement(lp_c_pi, lp_r_pi, lp_c_ref, lp_r_ref)
                    # Samples are "corrupt" if they have low margin OR policy/ref disagree
                    corrupt_mask = corrupt_mask | disagree_mask
                    clean_mask = ~corrupt_mask

            # Per-sample weights (down-weight adversarial/corrupt if desired)
            w = torch.ones_like(obs_win_pi)
            w = torch.where(corrupt_mask, w * args.adv_dpo_weight, w)

            # DPO on ALL samples
            loss_dpo = dpo_loss_observed(
                obs_win_pi, obs_lose_pi,
                obs_win_ref, obs_lose_ref,
                eta=args.eta,
                weight=w
            )

            # IHL (Inverse Hinge Loss) for unlearning adversarial corruption
            # Applied to detected corrupt/adversarial subset
            if args.ihl_weight > 0.0 and corrupt_mask.any():
                corrupt_win_lp = obs_win_pi[corrupt_mask]
                corrupt_lose_lp = obs_lose_pi[corrupt_mask]
                
                # Select IHL loss type
                if args.ihl_type == "hinge":
                    loss_ihl = ihl_hinge_text(corrupt_win_lp, corrupt_lose_lp, margin=args.ihl_margin)
                elif args.ihl_type == "squared":
                    loss_ihl = ihl_squared_loss(corrupt_win_lp, corrupt_lose_lp)
                elif args.ihl_type == "reverse":
                    loss_ihl = ihl_reverse_ranking(corrupt_win_lp, corrupt_lose_lp, 
                                                    temperature=args.ihl_reverse_temp)
                else:
                    loss_ihl = ihl_hinge_text(corrupt_win_lp, corrupt_lose_lp, margin=args.ihl_margin)
                
                # Apply adaptive weighting if enabled
                if args.ihl_adaptive:
                    corrupt_margins = margin[corrupt_mask]
                    adaptive_weights = adaptive_ihl_weight(corrupt_margins, 
                                                            min_weight=0.1, 
                                                            max_weight=1.0, 
                                                            threshold=args.filter_tau)
                    # Re-compute IHL with adaptive weights (element-wise multiplication)
                    if args.ihl_type == "hinge":
                        per_sample_loss = torch.relu(args.ihl_margin + corrupt_win_lp - corrupt_lose_lp)
                    elif args.ihl_type == "squared":
                        per_sample_loss = (corrupt_win_lp - corrupt_lose_lp) ** 2
                    elif args.ihl_type == "reverse":
                        logit = args.ihl_reverse_temp * (corrupt_lose_lp - corrupt_win_lp)
                        per_sample_loss = F.softplus(-logit)
                    loss_ihl = (per_sample_loss * adaptive_weights).mean()
            else:
                loss_ihl = torch.tensor(0.0, device=device, dtype=lp_c_pi.dtype)

            # Combined loss: DPO (on all) + IHL (on corrupt subset for unlearning)
            loss = loss_dpo + args.ihl_weight * loss_ihl

            # Scale loss by gradient accumulation steps
            loss = loss / args.grad_accum
        
        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.grad_accum == 0:
            # Optional: gradient clipping for stability
            if scaler is not None:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Log metrics (every 2 steps or at step 1 to ensure we see output)
        if step == 1 or step % 2 == 0:
            with torch.no_grad():
                # Compute metrics efficiently
                margin_diff = lp_c_pi - lp_r_pi
                
                # Basic metrics
                m = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": float(loss.item() * args.grad_accum),  # Unscale for logging
                    "loss_dpo": float(loss_dpo.item()),
                    "loss_ihl": float(loss_ihl.item()),
                    
                    # Margin statistics
                    "margin_mean": float(margin_diff.mean().item()),
                    "margin_abs_mean": float(margin_diff.abs().mean().item()),
                    "margin_std": float(margin_diff.std().item()),
                    
                    # Adversary & corruption detection metrics
                    "flip_frac": float(flip_mask.float().mean().item()),
                    "flip_count": int(flip_mask.sum().item()),
                    "clean_frac": float(clean_mask.float().mean().item()),
                    "corrupt_frac": float(corrupt_mask.float().mean().item()),
                    
                    # Accuracy metrics (how well does model distinguish chosen vs rejected)
                    "policy_accuracy": float((lp_c_pi > lp_r_pi).float().mean().item()),
                    "ref_accuracy": float((lp_c_ref > lp_r_ref).float().mean().item()),
                }
                
                # IHL-specific metrics when unlearning is active
                if args.ihl_weight > 0.0 and corrupt_mask.any():
                    corrupt_margins = margin[corrupt_mask]
                    m.update({
                        "ihl_corrupt_margin_mean": float(corrupt_margins.mean().item()),
                        "ihl_corrupt_margin_std": float(corrupt_margins.std().item()),
                        "ihl_unlearn_strength": float(args.ihl_weight),
                    })
                
                # Agreement-based filtering metrics
                if args.use_agreement_filter:
                    agree_mask, disagree_mask = filter_by_agreement(lp_c_pi, lp_r_pi, lp_c_ref, lp_r_ref)
                    m.update({
                        "policy_ref_agreement": float(agree_mask.float().mean().item()),
                        "policy_ref_disagreement": float(disagree_mask.float().mean().item()),
                    })
                
                # Log KL divergence from reference (measure of policy drift)
                kl_c = (lp_c_pi - lp_c_ref).mean()
                kl_r = (lp_r_pi - lp_r_ref).mean()
                m.update({
                    "kl_chosen": float(kl_c.item()),
                    "kl_rejected": float(kl_r.item()),
                    "kl_total": float((kl_c + kl_r).item() / 2),
                })
                
            print(json.dumps(m))
    
    # Clear CUDA cache at the end of each epoch to reduce memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save LoRA adapter at epoch end
    print(f"\nSaving model checkpoint for epoch {epoch+1}...")
    outdir = os.path.join(args.save_dir, f"adapter_epoch_{epoch+1}")
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"Saved adapter to {outdir}")
    
    # Optional: save optimizer state for resuming training
    torch.save({
        'epoch': epoch + 1,
        'step': step,
        'optimizer_state_dict': opt.state_dict(),
    }, os.path.join(outdir, 'training_state.pt'))