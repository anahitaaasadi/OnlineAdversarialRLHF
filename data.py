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