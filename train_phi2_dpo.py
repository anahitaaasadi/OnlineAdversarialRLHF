
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

"""
================================================================================
ADVERSARIAL RLHF WITH INVERSE HINGE LOSS (IHL) UNLEARNING
================================================================================

This implementation combines Direct Preference Optimization (DPO) with Inverse 
Hinge Loss (IHL) to train language models that are robust to adversarial label 
flipping in preference feedback.

KEY COMPONENTS:

1. DPO (Direct Preference Optimization):
   - Standard preference learning from chosen/rejected pairs
   - Uses KL penalty to reference model to prevent drift
   - Applied to all training samples (clean + corrupted)

2. ADVERSARY SIMULATION:
   - Simulates adversarial label flipping during training
   - Strategies: uncertainty-targeted, random, or confidence-targeted
   - Models realistic attacks on preference learning systems

3. CORRUPTION DETECTION:
   - Margin-based filtering: low margin = likely corrupted
   - Agreement-based filtering: policy-reference disagreement = suspicious
   - Separates samples into "clean" vs "corrupt" subsets

4. IHL UNLEARNING (Core Innovation):
   - Applied ONLY to detected corrupt/adversarial subset
   - GOAL: Unlearn the effects of adversarial label flips
   
   Types of IHL:
   a) Hinge Loss (default): L = [margin + log P(y_obs_win) - log P(y_obs_lose)]_+
      - Penalizes large margins in corrupted direction
      - Margin parameter controls unlearning strength
   
   b) Squared Loss: L = (log P(y_obs_win) - log P(y_obs_lose))^2
      - Smoother gradients
      - Encourages equal probabilities for flipped pairs
   
   c) Reverse Ranking: L = log(1 + exp(-T * (log P(y_obs_lose) - log P(y_obs_win))))
      - Most aggressive: actively reverses corrupted preference
      - Temperature T controls reversal strength

5. ADAPTIVE WEIGHTING:
   - Samples with smaller margins get higher IHL weight
   - Based on insight: uncertain examples more likely corrupted
   - Sigmoid-based smooth transition from high to low weight

TRAINING DYNAMICS:
- DPO learns from all samples (with down-weighting for corrupt)
- IHL simultaneously unlearns adversarial signal in corrupt subset
- Balance controlled by --ihl-weight hyperparameter
- Model learns to be robust to label noise while improving on clean data

USAGE EXAMPLE:
python train_phi2_dpo.py \\
    --train data/train.jsonl \\
    --epochs 3 \\
    --flip-rate 0.2 \\
    --flip-strategy uncertainty \\
    --ihl-weight 0.5 \\
    --ihl-type hinge \\
    --ihl-margin 0.3 \\
    --ihl-adaptive \\
    --filter-tau 0.5 \\
    --adv-dpo-weight 0.3

REFERENCES:
- DPO: Rafailov et al., "Direct Preference Optimization"
- IHL for unlearning: Applied to reverse adversarial learning signals
================================================================================
"""

# ---------- Data ----------
class PairDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 2048):
        self.items = []
        self.tok = tokenizer
        self.max_len = max_len

        dec = json.JSONDecoder()

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Fast path: whole file is a JSON array
        if content.lstrip().startswith("["):
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("Top-level JSON is not a list.")
                for i, ex in enumerate(data, 1):
                    self._validate_and_append(ex, file_line=i)
                return
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON array file '{path}': {e}") from e

        # Helpers -------------------------------------------------------------
        def advance_ws_and_seps(s: str, i: int):
            """
            Consume any combination of:
              - whitespace (including real newlines),
              - commas,
              - literal '\\n' and literal '\\r\\n' sequences **between objects**.
            Returns (new_i, advanced_any, new_lineno_delta).
            """
            N = len(s)
            advanced = False
            newlines = 0

            # consume plain whitespace first
            while i < N and s[i].isspace():
                if s[i] == "\n":
                    newlines += 1
                i += 1
                advanced = True

            # now repeatedly consume separators
            while True:
                progressed = False
                # literal '\n'
                if i + 1 < N and s[i] == "\\" and s[i+1] == "n":
                    i += 2
                    advanced = progressed = True
                # literal '\r\n'
                elif i + 3 < N and s[i:i+4] == "\\r\\n":
                    i += 4
                    advanced = progressed = True
                # stray commas
                elif i < N and s[i] == ",":
                    i += 1
                    advanced = progressed = True

                # trailing whitespace after those separators
                while i < N and s[i].isspace():
                    if s[i] == "\n":
                        newlines += 1
                    i += 1
                    advanced = True

                if not progressed:
                    break

            return i, advanced, newlines
        # --------------------------------------------------------------------

        i, N = 0, len(content)
        lineno = 1

        # main streaming loop
        while True:
            i, _, nl = advance_ws_and_seps(content, i)
            lineno += nl
            if i >= N:
                break

            # try to decode an object/value starting at i
            try:
                obj, end = dec.raw_decode(content, i)
            except json.JSONDecodeError as e:
                snippet = content[max(0, i-60):min(N, i+60)]
                raise RuntimeError(
                    f"JSON parse error in '{path}' near char {i} (approx line {lineno}): {e.msg}\n"
                    f"...{snippet}..."
                ) from e

            self._validate_and_append(obj, file_line=lineno)
            i = end  # continue scanning after this object
            if i >= N:
                break
            
    def _validate_and_append(self, ex: Dict[str, Any], file_line: int, char_pos: int = None):
        # Schema check: must have prompt/pos/neg strings
        for k in ("prompt", "pos", "neg"):
            if k not in ex or not isinstance(ex[k], str):
                loc = f"line {file_line}" + (f", char {char_pos}" if char_pos is not None else "")
                raise ValueError(f"Example at {loc} missing string field '{k}'. Got: {repr(ex.get(k))}")
        self.items.append(ex)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
    

def build_inputs(tokenizer, prompt: str, resp: str, max_len: int):
    # Left-pad so loss/labels align at the end (common for causal LM DPO)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    text = prompt + resp
    enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=max_len)
    
    # More efficient: encode prompt once and calculate response start position
    p_len = len(tokenizer(prompt, padding=False, truncation=True, max_length=max_len)["input_ids"])
    ids = enc["input_ids"][0]
    total_len = len(ids)
    
    # Response starts after prompt tokens
    resp_start = max(0, total_len - max(0, total_len - p_len))
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[resp_start:] = True
    return ids, enc["attention_mask"][0], mask

def batchify(batch, tokenizer, max_len):
    # Returns a dict of tensors for chosen and rejected
    # Pre-allocate lists with correct size for efficiency
    batch_size = len(batch)
    ids_c, attn_c, mask_c = [None] * batch_size, [None] * batch_size, [None] * batch_size
    ids_r, attn_r, mask_r = [None] * batch_size, [None] * batch_size, [None] * batch_size
    prompts, chosens, rejects = [None] * batch_size, [None] * batch_size, [None] * batch_size
    
    for i, ex in enumerate(batch):
        p, pos, neg = ex["prompt"], ex["pos"], ex["neg"]
        ic, ac, mc = build_inputs(tokenizer, p, pos, max_len)
        ir, ar, mr = build_inputs(tokenizer, p, neg, max_len)
        ids_c[i], attn_c[i], mask_c[i] = ic, ac, mc
        ids_r[i], attn_r[i], mask_r[i] = ir, ar, mr
        prompts[i], chosens[i], rejects[i] = p, pos, neg
    
    def pad_stack(seqs):
        lens = [len(s) for s in seqs]
        maxL = max(lens)
        pad_id = tokenizer.pad_token_id
        # Use stack instead of full+loop for better performance
        out = torch.full((len(seqs), maxL), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), maxL), dtype=torch.long)
        for i, s in enumerate(seqs):
            slen = len(s)
            out[i, -slen:] = s
            mask[i, -slen:] = 1
        return out, mask
    
    def pad_bool(seqs):
        maxL = max(len(s) for s in seqs)
        out = torch.zeros((len(seqs), maxL), dtype=torch.bool)
        for i, s in enumerate(seqs):
            out[i, -len(s):] = s
        return out
    
    c_ids, c_attn_mask = pad_stack(ids_c)
    r_ids, r_attn_mask = pad_stack(ids_r)
    c_resp_mask = pad_bool(mask_c)
    r_resp_mask = pad_bool(mask_r)
    
    return {
        "c_ids": c_ids, "c_attn": c_attn_mask, "c_resp_mask": c_resp_mask,
        "r_ids": r_ids, "r_attn": r_attn_mask, "r_resp_mask": r_resp_mask,
        "prompts": prompts, "chosens": chosens, "rejects": rejects
    }

# ---------- Logprob utilities ----------
def seq_logprobs(model, input_ids, attention_mask, resp_mask, requires_grad=False):
    # Compute sum log p(y|x) over response tokens only (causal shift)
    # Use context manager only if gradients not required (for reference model)
    context_mgr = torch.no_grad() if not requires_grad else torch.enable_grad()
    
    with context_mgr:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    attn = attention_mask[:, 1:]
    resp = resp_mask[:, 1:]  # align after shift
    
    # Use more efficient indexing: gather + squeeze in one op
    logp_tok = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    
    # Keep only tokens that are both attended and in response (combine operations)
    use = attn.bool() & resp.bool()
    
    # Mask and sum in one step - more efficient than masked_fill + sum
    seq_logp = (logp_tok * use.float()).sum(dim=1)
    return seq_logp

# ---------- Losses ----------
def dpo_loss_observed(lp_win_pi, lp_lose_pi, lp_win_ref, lp_lose_ref, eta: float, weight: torch.Tensor = None):
    """
    Standard DPO loss with optional per-sample weighting.
    
    Args:
        lp_win_pi: Log-probs of winning responses under policy
        lp_lose_pi: Log-probs of losing responses under policy
        lp_win_ref: Log-probs of winning responses under reference
        lp_lose_ref: Log-probs of losing responses under reference
        eta: Temperature/KL penalty coefficient (beta in DPO paper)
        weight: Optional per-sample weights for handling adversarial examples
    """
    # z_i = eta * [(lp_win_pi - lp_win_ref) - (lp_lose_pi - lp_lose_ref)]
    z = eta * ((lp_win_pi - lp_win_ref) - (lp_lose_pi - lp_lose_ref))
    loss_vec = F.binary_cross_entropy_with_logits(z, torch.ones_like(z), reduction="none")
    if weight is not None:
        denom = torch.clamp(weight.sum(), min=1e-8)
        return (loss_vec * weight).sum() / denom
    return loss_vec.mean()

def ihl_hinge_text(observed_win_lp, observed_lose_lp, margin: float = 0.5):
    """
    Inverse Hinge Loss (IHL) for unlearning adversarial preference flips.
    
    The hinge loss encourages the model to REDUCE the margin between observed winner
    and loser, effectively "unlearning" the corrupted preference signal. This is the
    opposite of standard hinge loss which maximizes the margin.
    
    Formula: L_IHL = [margin + log P(y_win|x) - log P(y_lose|x)]_+
    
    Key insight: When adversary flips labels (neg→win, pos→lose), standard training
    would increase P(neg|x) and decrease P(pos|x). IHL counters this by penalizing
    large margins in the "wrong" direction, helping the model unlearn the adversarial signal.
    
    Args:
        observed_win_lp: Log-probs of "observed" winning responses (may be adversarially flipped)
        observed_lose_lp: Log-probs of "observed" losing responses (may be adversarially flipped)
        margin: Target margin reduction (smaller = stronger unlearning)
    
    Returns:
        Mean hinge loss for unlearning
    """
    # Penalize when observed winner has higher probability than observed loser
    # The margin parameter controls how aggressively we unlearn
    return torch.relu(margin + observed_win_lp - observed_lose_lp).mean()

def ihl_squared_loss(observed_win_lp, observed_lose_lp):
    """
    Alternative IHL variant using squared loss instead of hinge.
    Provides smoother gradients for unlearning.
    
    Formula: L_IHL_sq = (log P(y_win|x) - log P(y_lose|x))^2
    """
    diff = observed_win_lp - observed_lose_lp
    return (diff ** 2).mean()

def ihl_reverse_ranking(observed_win_lp, observed_lose_lp, temperature: float = 1.0):
    """
    Reverse ranking loss: actively encourages swapping the preference order.
    More aggressive unlearning by explicitly trying to reverse the corrupted signal.
    
    Formula: L_reverse = log(1 + exp(-temperature * (log P(y_lose|x) - log P(y_win|x))))
    
    This makes the model prefer the "observed loser" over "observed winner",
    effectively reversing the adversarial flip.
    """
    # Encourage y_lose to have higher prob than y_win (reverse the corruption)
    logit = temperature * (observed_lose_lp - observed_win_lp)
    return F.softplus(-logit).mean()

def adaptive_ihl_weight(margin, min_weight=0.1, max_weight=1.0, threshold=0.5):
    """
    Adaptive weighting for IHL based on corruption confidence.
    
    Samples with smaller margins (more uncertain) get higher unlearning weight,
    as they are more likely to be adversarially corrupted.
    
    Args:
        margin: Absolute margin |log P(chosen) - log P(rejected)|
        min_weight: Minimum weight for high-margin (confident) samples
        max_weight: Maximum weight for low-margin (uncertain) samples  
        threshold: Margin threshold for full weight
    """
    # Sigmoid-based smooth transition
    normalized_margin = margin / threshold
    weight = min_weight + (max_weight - min_weight) * torch.sigmoid(-5 * (normalized_margin - 1))
    return weight

# ---------- Adversary & filter ----------
def flip_labels_uncertainty_targeting_text(lp_c_pi, lp_r_pi, flip_rate: float):
    """
    Adversarial label flipping strategy: targets uncertain examples.
    
    Simulates an adversary that flips preference labels for examples where the model
    is most uncertain (smallest margin between chosen and rejected). This is a realistic
    attack as uncertain examples are where flipping has maximum impact on learning.
    
    Args:
        lp_c_pi: Log-probs of chosen responses under current policy
        lp_r_pi: Log-probs of rejected responses under current policy
        flip_rate: Fraction of batch to flip (0.0 = no flips, 1.0 = flip all)
    
    Returns:
        Boolean mask: True = flip this example (treat rejected as winner)
    """
    # Flip labels for a fraction with smallest |margin|
    margins = (lp_c_pi - lp_r_pi).abs()
    B = margins.shape[0]
    k = int(flip_rate * B)
    flip = torch.zeros(B, dtype=torch.bool, device=margins.device)
    if k > 0:
        idx = torch.argsort(margins, descending=False)[:k]
        flip[idx] = True
    return flip  # True = flip (i.e., treat 'neg' as observed winner)

def flip_labels_random(lp_c_pi, lp_r_pi, flip_rate: float):
    """
    Random label flipping strategy for comparison.
    
    Randomly flips a fraction of labels without targeting specific examples.
    Useful as a baseline to compare against uncertainty-targeted attacks.
    """
    B = lp_c_pi.shape[0]
    k = int(flip_rate * B)
    flip = torch.zeros(B, dtype=torch.bool, device=lp_c_pi.device)
    if k > 0:
        idx = torch.randperm(B, device=lp_c_pi.device)[:k]
        flip[idx] = True
    return flip

def flip_labels_confidence_targeting(lp_c_pi, lp_r_pi, flip_rate: float):
    """
    Adversarial strategy: targets most confident examples.
    
    Flips labels where model is most confident (largest margin). This can be
    particularly damaging as it corrupts the "anchor" examples the model relies on.
    """
    margins = (lp_c_pi - lp_r_pi).abs()
    B = margins.shape[0]
    k = int(flip_rate * B)
    flip = torch.zeros(B, dtype=torch.bool, device=margins.device)
    if k > 0:
        idx = torch.argsort(margins, descending=True)[:k]  # Largest margins
        flip[idx] = True
    return flip

def filter_by_margin(lp_c_pi, lp_r_pi, tau: float):
    """
    Partition examples into clean vs. corrupt based on margin threshold.
    
    Heuristic: Examples with small margins |log P(chosen) - log P(rejected)| < tau
    are likely to be adversarially corrupted or noisy, as correctly labeled preferences
    should show clear differentiation.
    
    Args:
        lp_c_pi: Log-probs of chosen responses
        lp_r_pi: Log-probs of rejected responses  
        tau: Margin threshold (lower = stricter filtering)
    
    Returns:
        clean_mask: Boolean mask for likely clean examples
        corrupt_mask: Boolean mask for likely corrupted examples
        margin: Computed margins for all examples
    """
    margin = (lp_c_pi - lp_r_pi).abs()
    corrupt = margin < tau
    clean = ~corrupt
    return clean, corrupt, margin

def filter_by_agreement(lp_c_pi, lp_r_pi, lp_c_ref, lp_r_ref):
    """
    Alternative filtering: detect examples where policy and reference disagree.
    
    If policy prefers different response than reference model, this may indicate
    adversarial corruption has affected the policy.
    
    Returns:
        agree_mask: Policy and reference agree on preference
        disagree_mask: Policy and reference disagree
    """
    policy_prefers_chosen = lp_c_pi > lp_r_pi
    ref_prefers_chosen = lp_c_ref > lp_r_ref
    agree = policy_prefers_chosen == ref_prefers_chosen
    disagree = ~agree
    return agree, disagree

# ---------- Training ----------
def main():
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

if __name__ == "__main__":
    main()
