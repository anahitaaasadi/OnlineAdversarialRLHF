
import torch
import torch.nn.functional as F

def dpo_loss(pi_logits, pi0_logits, a_w, a_l, eta: float = 1.0):
    """DPO loss over pairs marked clean.
    Implements: -log sigma( eta * [log pi/pi0(a_w) - log pi/pi0(a_l)] )
    """
    log_pi = pi_logits.log_softmax(dim=-1)
    log_pi0 = pi0_logits.log_softmax(dim=-1)
    lograt_w = (log_pi - log_pi0).gather(1, a_w.view(-1,1)).squeeze(1)
    lograt_l = (log_pi - log_pi0).gather(1, a_l.view(-1,1)).squeeze(1)
    z = eta * (lograt_w - lograt_l)
    return F.binary_cross_entropy_with_logits(z, torch.ones_like(z))

def ihl_loss(pi_probs, a_w):
    """Inverted hinge loss on predicted-corrupt winners a_w.
    L = [1 + p(a_w) - max_{v!=a_w} p(v)]_+
    """
    B, A = pi_probs.shape
    p_w = pi_probs[torch.arange(B), a_w]
    # top competitor prob (mask out a_w by large negative)
    neg_inf = torch.tensor(-1e9, device=pi_probs.device)
    masked = pi_probs.clone()
    masked[torch.arange(B), a_w] = -1.0  # ensure competitor is different
    p_comp, _ = masked.max(dim=1)
    margin = 1.0 + p_w - p_comp
    return torch.relu(margin).mean()
