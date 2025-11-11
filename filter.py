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

def flip_labels_random(lp_c_pi, lp_r_pi, flip_rate: float):
    B = lp_c_pi.shape[0]
    k = int(flip_rate * B)
    flip = torch.zeros(B, dtype=torch.bool, device=lp_c_pi.device)
    if k > 0:
        idx = torch.randperm(B, device=lp_c_pi.device)[:k]
        flip[idx] = True
    return flip

def flip_labels_confidence_targeting(lp_c_pi, lp_r_pi, flip_rate: float):
    margins = (lp_c_pi - lp_r_pi).abs()
    B = margins.shape[0]
    k = int(flip_rate * B)
    flip = torch.zeros(B, dtype=torch.bool, device=margins.device)
    if k > 0:
        idx = torch.argsort(margins, descending=True)[:k]  # Largest margins
        flip[idx] = True
    return flip

def filter_by_margin(lp_c_pi, lp_r_pi, tau: float):
    margin = (lp_c_pi - lp_r_pi).abs()
    corrupt = margin < tau
    clean = ~corrupt
    return clean, corrupt, margin

def filter_by_agreement(lp_c_pi, lp_r_pi, lp_c_ref, lp_r_ref):
    policy_prefers_chosen = lp_c_pi > lp_r_pi
    ref_prefers_chosen = lp_c_ref > lp_r_ref
    agree = policy_prefers_chosen == ref_prefers_chosen
    disagree = ~agree
    return agree, disagree