
import torch

@torch.no_grad()
def flag_corruption_by_policy_margin(pi_logits, a_w, a_l, tau: float):
    """Flag likely corrupt pairs using *policy* margin as uncertainty proxy.
    If |logit_w - logit_l| < tau -> flag as 'corrupt' (ambiguous / high-risk).
    Returns: clean_mask, corrupt_mask
    """
    logits_w = pi_logits.gather(1, a_w.view(-1,1)).squeeze(1)
    logits_l = pi_logits.gather(1, a_l.view(-1,1)).squeeze(1)
    margin = logits_w - logits_l
    corrupt_mask = margin.abs() < tau
    clean_mask = ~corrupt_mask
    return clean_mask, corrupt_mask, margin
