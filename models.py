
import torch
import torch.nn as nn

class Policy(nn.Module):
    """Simple linear policy: logits = W x + b over discrete actions."""
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.net(x)  # logits [B, A]

    def probs(self, x):
        return torch.softmax(self.forward(x), dim=-1)

class RewardOracle(nn.Module):
    """Ground-truth reward r*(x, a) = <w_a, x> (Bradley–Terry-compatible)."""
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_actions, obs_dim) * 0.5)

    @torch.no_grad()
    def score(self, x):
        # x: [B, D], return [B, A] rewards
        return x @ self.W.t()
