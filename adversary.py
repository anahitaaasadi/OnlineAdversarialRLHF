
import torch

@torch.no_grad()
def flip_labels_uncertainty_targeting(
    y, margins, flip_rate
):
    """Flip a fraction `flip_rate` of labels, prioritizing smallest |margin|.

    y: [B] in {0,1} clean preference (1 means a_w preferred)
    margins: [B] real, e.g., r*(x,a_w)-r*(x,a_l)
    """
    B = y.shape[0]
    k = int(flip_rate * B)
    if k == 0:
        return y.clone(), torch.zeros_like(y, dtype=torch.bool)
    # rank by |margin| ascending
    idx = torch.argsort(margins.abs(), descending=False)
    flip_idx = idx[:k]
    y_tilde = y.clone()
    y_tilde[flip_idx] = 1 - y_tilde[flip_idx]
    mask = torch.zeros(B, dtype=torch.bool, device=y.device)
    mask[flip_idx] = True
    return y_tilde, mask
