
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 0
    steps: int = 2000
    batch_size: int = 256
    obs_dim: int = 32          # feature dim of prompts x
    num_actions: int = 8       # candidate responses A
    eta: float = 1.0           # DPO temperature (log-ratio scale)
    lr: float = 3e-3
    flip_rate: float = 0.2     # adversary budget (expected fraction flipped)
    ihl_weight: float = 0.5    # weight for IHL
    filter_tau: float = 0.3    # threshold on |margin| for flagging corruption
    retro_K: int = 200         # retroactive unlearning period (0 = disable)
    retro_passes: int = 1
