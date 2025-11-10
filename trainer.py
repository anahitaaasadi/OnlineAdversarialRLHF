
from dataclasses import asdict
import torch, json, os
from tqdm import trange

from configs import Config
from models import Policy, RewardOracle
from data import sample_batch
from adversary import flip_labels_uncertainty_targeting
from filtering import flag_corruption_by_policy_margin
from losses import dpo_loss, ihl_loss

def bt_preference(reward_w, reward_l):
    # Bradley–Terry preference probability for y=1 (a_w preferred)
    return torch.sigmoid(reward_w - reward_l)

def run(cfg: Config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.seed)

    policy = Policy(cfg.obs_dim, cfg.num_actions).to(device)
    pi0 = Policy(cfg.obs_dim, cfg.num_actions).to(device)
    # Freeze reference policy pi0
    for p in pi0.parameters(): p.requires_grad = False
    # Initialize pi0 = current policy snapshot
    pi0.load_state_dict(policy.state_dict())

    oracle = RewardOracle(cfg.obs_dim, cfg.num_actions).to(device)

    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    history_corrupt = []  # buffer of (x, a_w) flagged corrupt (for retroactive unlearning)

    os.makedirs("runs", exist_ok=True)
    log_path = os.path.join("runs", "online_run.jsonl")
    with open(log_path, "w") as f:
        f.write("")

    for t in trange(cfg.steps, desc="online-rlhf"):
        x, a_w, a_l = sample_batch(cfg.obs_dim, cfg.num_actions, cfg.batch_size, device=device)
        with torch.no_grad():
            rewards = oracle.score(x)                  # [B, A]
            r_w = rewards[torch.arange(cfg.batch_size), a_w]
            r_l = rewards[torch.arange(cfg.batch_size), a_l]
            p_clean = bt_preference(r_w, r_l)         # clean pref prob
            y = torch.bernoulli(p_clean).long()       # clean labels {0,1}
            margins = r_w - r_l

        # Adversary flips labels, prioritizing small |margin|
        y_tilde, flip_mask = flip_labels_uncertainty_targeting(y, margins, cfg.flip_rate)

        # Compute policy logits and filter
        pi_logits = policy(x)             # [B, A]
        pi0_logits = pi0(x)
        clean_mask, corrupt_mask, pol_margins = flag_corruption_by_policy_margin(pi_logits, a_w, a_l, cfg.filter_tau)

        # Split pairs for losses
        clean_idx = clean_mask.nonzero(as_tuple=False).squeeze(1)
        corrupt_idx = corrupt_mask.nonzero(as_tuple=False).squeeze(1)

        # Winners under observed (possibly flipped) labels:
        # if y_tilde=1 -> a_w preferred; else swap roles for DPO on clean.
        a_w_obs = torch.where(y_tilde==1, a_w, a_l)
        a_l_obs = torch.where(y_tilde==1, a_l, a_w)

        opt.zero_grad()

        loss = torch.tensor(0.0, device=device)

        # DPO on clean
        if clean_idx.numel() > 0:
            dpo = dpo_loss(
                pi_logits[clean_idx], pi0_logits[clean_idx],
                a_w_obs[clean_idx], a_l_obs[clean_idx], eta=cfg.eta
            )
            loss = loss + dpo

        # IHL on corrupt: demote the observed "winner" (likely corrupted) a_w_obs
        if corrupt_idx.numel() > 0 and cfg.ihl_weight > 0:
            pi_probs = torch.softmax(pi_logits[corrupt_idx], dim=-1)
            ihl = ihl_loss(pi_probs, a_w_obs[corrupt_idx])
            loss = loss + cfg.ihl_weight * ihl

            # Save for retroactive unlearning: (x, winner)
            history_corrupt.append((x[corrupt_idx].detach().cpu(), a_w_obs[corrupt_idx].detach().cpu()))

        loss.backward()
        opt.step()

        # Retroactive unlearning pass
        if cfg.retro_K > 0 and (t+1) % cfg.retro_K == 0 and len(history_corrupt) > 0:
            for _ in range(cfg.retro_passes):
                Xr = torch.cat([p[0] for p in history_corrupt], dim=0).to(device)
                Ar = torch.cat([p[1] for p in history_corrupt], dim=0).to(device)
                pr = torch.softmax(policy(Xr), dim=-1)
                ihl_r = ihl_loss(pr, Ar)
                opt.zero_grad()
                (cfg.ihl_weight * ihl_r).backward()
                opt.step()

        # Basic logging
        with torch.no_grad():
            pi_probs = torch.softmax(policy(x), dim=-1)
            # Expected reward under policy (proxy objective)
            exp_reward = (pi_probs * rewards).sum(dim=1).mean().item()
            metrics = {
                "step": t+1,
                "loss": float(loss.item()),
                "exp_reward": exp_reward,
                "flip_rate": float(cfg.flip_rate),
                "clean_frac": float(clean_mask.float().mean().item()),
                "corrupt_frac": float(corrupt_mask.float().mean().item()),
                "actual_flipped_frac": float(flip_mask.float().mean().item()),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

    return {"log_path": log_path}
