
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
    # z_i = eta * [(lp_win_pi - lp_win_ref) - (lp_lose_pi - lp_lose_ref)]
    z = eta * ((lp_win_pi - lp_win_ref) - (lp_lose_pi - lp_lose_ref))
    loss_vec = F.binary_cross_entropy_with_logits(z, torch.ones_like(z), reduction="none")
    if weight is not None:
        denom = torch.clamp(weight.sum(), min=1e-8)
        return (loss_vec * weight).sum() / denom
    return loss_vec.mean()

def ihl_hinge_text(observed_win_lp, observed_lose_lp, margin: float = 0.5):
    # L = [m + lp_win - lp_lose]_+ ; decreasing margin reduces lp_win vs lp_lose
    return torch.relu(margin + observed_win_lp - observed_lose_lp).mean()

# ---------- Adversary & filter ----------
def flip_labels_uncertainty_targeting_text(lp_c_pi, lp_r_pi, flip_rate: float):
    # Flip labels for a fraction with smallest |margin|
    margins = (lp_c_pi - lp_r_pi).abs()
    B = margins.shape[0]
    k = int(flip_rate * B)
    flip = torch.zeros(B, dtype=torch.bool, device=margins.device)
    if k > 0:
        idx = torch.argsort(margins, descending=False)[:k]
        flip[idx] = True
    return flip  # True = flip (i.e., treat 'neg' as observed winner)

def filter_by_margin(lp_c_pi, lp_r_pi, tau: float):
    margin = (lp_c_pi - lp_r_pi).abs()
    corrupt = margin < tau
    clean = ~corrupt
    return clean, corrupt, margin

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
    ap.add_argument("--ihl-weight", type=float, default=0.0)
    ap.add_argument("--ihl-margin", type=float, default=0.5)
    ap.add_argument("--flip-rate", type=float, default=0.0)
    ap.add_argument("--filter-tau", type=float, default=0.5)
    ap.add_argument("--adv-dpo-weight", type=float, default=0.5,
                help="Relative weight for adversarial/corrupt pairs in DPO (1.0 = equal).")
    ap.add_argument("--save-dir", type=str, default="runs/phi2_dpo")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
            
            # Simulated adversary: choose flips
            with torch.no_grad():
                flip_mask = flip_labels_uncertainty_targeting_text(lp_c_pi, lp_r_pi, args.flip_rate)

            # Use automatic mixed precision if available
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Observed winners/losers for BOTH policy and reference (respect flips)
                obs_win_pi   = torch.where(flip_mask, lp_r_pi, lp_c_pi)
                obs_lose_pi  = torch.where(flip_mask, lp_c_pi, lp_r_pi)
                obs_win_ref  = torch.where(flip_mask, lp_r_ref, lp_c_ref)
                obs_lose_ref = torch.where(flip_mask, lp_c_ref, lp_r_ref)

                # Margin-based partition
                with torch.no_grad():
                    clean_mask, corrupt_mask, margin = filter_by_margin(lp_c_pi, lp_r_pi, args.filter_tau)

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

                # Optional IHL on adversarial subset
                if args.ihl_weight > 0.0 and corrupt_mask.any():
                    loss_ihl = ihl_hinge_text(
                        obs_win_pi[corrupt_mask],
                        obs_lose_pi[corrupt_mask],
                        margin=args.ihl_margin
                    )
                else:
                    loss_ihl = torch.tensor(0.0, device=device, dtype=lp_c_pi.dtype)

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

            if step % 10 == 0:
                with torch.no_grad():
                    # Compute metrics efficiently
                    margin_diff = lp_c_pi - lp_r_pi
                    m = {
                        "step": step,
                        "loss": float(loss.item() * args.grad_accum),  # Unscale for logging
                        "loss_dpo": float(loss_dpo.item()),
                        "loss_ihl": float(loss_ihl.item()),
                        "margin_mean": float(margin_diff.mean().item()),
                        "margin_abs_mean": float(margin_diff.abs().mean().item()),
                        "flip_frac": float(flip_mask.float().mean().item()),
                        "clean_frac": float(clean_mask.float().mean().item()),
                        "corrupt_frac": float(corrupt_mask.float().mean().item())
                    }
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
