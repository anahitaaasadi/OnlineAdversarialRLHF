
import torch
import torch.nn.functional as F

def dpo_loss(pi_logits, pi0_logits, a_w, a_l, eta: float = 1.0):
    log_pi = pi_logits.log_softmax(dim=-1)
    log_pi0 = pi0_logits.log_softmax(dim=-1)
    lograt_w = (log_pi - log_pi0).gather(1, a_w.view(-1,1)).squeeze(1)
    lograt_l = (log_pi - log_pi0).gather(1, a_l.view(-1,1)).squeeze(1)
    z = eta * (lograt_w - lograt_l)
    return F.binary_cross_entropy_with_logits(z, torch.ones_like(z))

def ihl_logit_hinge(pi_logits, a_w, margin: float = 1.0):
    """
    L = [m + z_w - max_{v!=w} z_v]_+
    where z are logits, m > 0.
    """
    B, A = pi_logits.shape
    z_w = pi_logits[torch.arange(B), a_w]
    # exclude a_w from competitor max
    mask = torch.ones_like(pi_logits, dtype=torch.bool)
    mask[torch.arange(B), a_w] = False
    z_comp = pi_logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    return torch.relu(margin + z_w - z_comp).mean()
