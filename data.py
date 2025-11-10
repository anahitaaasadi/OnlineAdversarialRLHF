
import torch

def sample_batch(obs_dim, num_actions, batch_size, device='cpu'):
    # Random prompts/features; normalize for stability.
    x = torch.randn(batch_size, obs_dim, device=device)
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    # Sample two candidate actions per example
    a_w = torch.randint(0, num_actions, (batch_size,), device=device)
    a_l = torch.randint(0, num_actions, (batch_size,), device=device)
    neq = a_w != a_l
    # Ensure different
    if not torch.all(neq):
        a_l = (a_w + 1 + torch.randint(0, num_actions-1, (batch_size,), device=device)) % num_actions
    return x, a_w, a_l
