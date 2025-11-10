
import os, json, math, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# ---------- Data ----------
class PairDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 2048):
        self.items = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        return ex

def build_inputs(tokenizer, prompt: str, resp: str, max_len: int):
    # Left-pad so loss/labels align at the end (common for causal LM DPO)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    text = prompt + resp
    enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=max_len)
    # Make a mask that isolates response tokens
    p_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_len)["input_ids"][0]
    ids = enc["input_ids"][0]
    # Response starts right after prompt tokens (best effort under truncation)
    resp_start = max(0, ids.shape[0] - (tokenizer(prompt+resp, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].shape[1] - p_ids.shape[0]))
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[resp_start:] = True
    return ids, enc["attention_mask"][0], mask

def batchify(batch, tokenizer, max_len):
    # Returns a dict of tensors for chosen and rejected
    ids_c, attn_c, mask_c = [], [], []
    ids_r, attn_r, mask_r = [], [], []
    prompts, chosens, rejects = [], [], []
    for ex in batch:
        p, pos, neg = ex["prompt"], ex["pos"], ex["neg"]
        ic, ac, mc = build_inputs(tokenizer, p, pos, max_len)
        ir, ar, mr = build_inputs(tokenizer, p, neg, max_len)
        ids_c.append(ic); attn_c.append(ac); mask_c.append(mc)
        ids_r.append(ir); attn_r.append(ar); mask_r.append(mr)
        prompts.append(p); chosens.append(pos); rejects.append(neg)
    def pad_stack(seqs):
        lens = [len(s) for s in seqs]
        maxL = max(lens)
        out = torch.full((len(seqs), maxL), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), maxL), dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, -len(s):] = s
            mask[i, -len(s):] = 1
        return out, mask
    def pad_bool(seqs):
        lens = [len(s) for s in seqs]
        maxL = max(lens)
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
def seq_logprobs(model, input_ids, attention_mask, resp_mask):
    # Compute sum log p(y|x) over response tokens only (causal shift)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    attn = attention_mask[:, 1:]
    resp = resp_mask[:, 1:]  # align after shift
    logp_tok = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # Keep only tokens that are both attended and in response
    use = (attn.bool() & resp.bool())
    # Avoid -inf by masking out-of-response positions
    logp_tok = logp_tok.masked_fill(~use, 0.0)
    seq_logp = logp_tok.sum(dim=1)  # sum over response tokens
    return seq_logp

# ---------- Losses ----------
def dpo_loss_text(lp_c_pi, lp_r_pi, lp_c_ref, lp_r_ref, eta: float):
    # z = eta * [(lp_c_pi - lp_c_ref) - (lp_r_pi - lp_r_ref)]
    z = eta * ((lp_c_pi - lp_c_ref) - (lp_r_pi - lp_r_ref))
    return F.binary_cross_entropy_with_logits(z, torch.ones_like(z))

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
    ap.add_argument("--save-dir", type=str, default="runs/phi2_dpo")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Phi-2 in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["Wqkv","fc1","fc2","Wout"],  # common Phi-2 module names
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_cfg)

    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto"
    )
    ref_model.requires_grad_(False)

    ds = PairDataset(args.train, tokenizer, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=True, collate_fn=lambda b: batchify(b, tokenizer, args.max_len))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    step = 0

    for epoch in range(args.epochs):
        for batch in tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}"):
            step += 1
            c_ids = batch["c_ids"].to(device); c_attn = batch["c_attn"].to(device); c_mask = batch["c_resp_mask"].to(device)
            r_ids = batch["r_ids"].to(device); r_attn = batch["r_attn"].to(device); r_mask = batch["r_resp_mask"].to(device)

            # Policy logprobs
            lp_c_pi = seq_logprobs(model, c_ids, c_attn, c_mask)
            lp_r_pi = seq_logprobs(model, r_ids, r_attn, r_mask)
            # Reference logprobs
            lp_c_ref = seq_logprobs(ref_model, c_ids, c_attn, c_mask)
            lp_r_ref = seq_logprobs(ref_model, r_ids, r_attn, r_mask)

            # Simulated adversary: which examples to flip (if enabled)
            flip_mask = flip_labels_uncertainty_targeting_text(lp_c_pi, lp_r_pi, args.flip_rate)
            # Apply observed winner/loser
            obs_win_lp = torch.where(flip_mask, lp_r_pi, lp_c_pi)
            obs_lose_lp = torch.where(flip_mask, lp_c_pi, lp_r_pi)
            # Filtering by margin
            clean_mask, corrupt_mask, margin = filter_by_margin(lp_c_pi, lp_r_pi, args.filter_tau)

            # DPO on clean
            if clean_mask.any():
                loss_dpo = dpo_loss_text(
                    lp_c_pi[clean_mask], lp_r_pi[clean_mask],
                    lp_c_ref[clean_mask], lp_r_ref[clean_mask],
                    eta=args.eta
                )
            else:
                loss_dpo = torch.tensor(0.0, device=device)

            # IHL on corrupt (optional, gentle)
            if args.ihl_weight > 0.0 and corrupt_mask.any():
                loss_ihl = ihl_hinge_text(
                    obs_win_lp[corrupt_mask],
                    obs_lose_lp[corrupt_mask],
                    margin=args.ihl_margin
                )
            else:
                loss_ihl = torch.tensor(0.0, device=device)

            loss = loss_dpo + args.ihl_weight * loss_ihl
            loss.backward()

            if step % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()

            if step % 10 == 0:
                with torch.no_grad():
                    m = {
                        "step": step,
                        "loss": float(loss.item()),
                        "loss_dpo": float(loss_dpo.item()),
                        "loss_ihl": float(loss_ihl.item()),
                        "margin_mean": float((lp_c_pi - lp_r_pi).mean().item()),
                        "margin_abs_mean": float((lp_c_pi - lp_r_pi).abs().mean().item()),
                        "flip_frac": float(flip_mask.float().mean().item()),
                        "clean_frac": float(clean_mask.float().mean().item()),
                        "corrupt_frac": float(corrupt_mask.float().mean().item())
                    }
                print(json.dumps(m))

        # Save LoRA adapter at epoch end
        outdir = os.path.join(args.save_dir, f"adapter_epoch_{epoch+1}")
        model.save_pretrained(outdir)
        tokenizer.save_pretrained(outdir)
        print(f"Saved adapter to {outdir}")

if __name__ == "__main__":
    main()
