import torch, json, argparse
from configs import Config
from models import Policy, RewardOracle
from data import sample_batch
from adversary import flip_labels_uncertainty_targeting
from losses import dpo_loss, ihl_logit_hinge  # <-- use logit hinge

def bt_preference(rw, rl):
    return torch.sigmoid(rw - rl)

@torch.no_grad()
def metrics(policy, pi0, oracle, obs_dim, num_actions, device='cpu', batch=2048, eta=1.0):
    x, aw, al = sample_batch(obs_dim, num_actions, batch, device=device)
    rew = oracle.score(x)
    pv = torch.softmax(policy(x), dim=-1)

    # Expected reward
    exp_reward = (pv * rew).sum(dim=1).mean().item()

    # Clean pairwise win-rate
    rw = rew[torch.arange(batch), aw]
    rl = rew[torch.arange(batch), al]
    true_win = (rw > rl)
    prob_w = pv[torch.arange(batch), aw]
    prob_l = pv[torch.arange(batch), al]
    pred_win = (prob_w > prob_l)
    winrate = (pred_win == true_win).float().mean().item()

    # Regret estimate via reverse-KL target
    p0 = torch.softmax(pi0(x), dim=-1)
    kl = (pv * (torch.log(pv+1e-8) - torch.log(p0+1e-8))).sum(dim=1).mean().item()
    J_pi = ((pv * rew).sum(dim=1) - eta * (pv * (torch.log(pv+1e-8) - torch.log(p0+1e-8))).sum(dim=1)).mean().item()
    logits_star = torch.log(p0 + 1e-8) + rew / eta
    pi_star = torch.softmax(logits_star, dim=-1)
    J_star = ((pi_star * rew).sum(dim=1) -
              eta * (pi_star * (torch.log(pi_star+1e-8) - torch.log(p0+1e-8))).sum(dim=1)).mean().item()

    regret = J_star - J_pi
    return {"exp_reward": exp_reward, "winrate": winrate, "kl": kl, "regret": regret}

def train_round(policy, pi0, oracle, cfg, device, batch_size, mode, eta, flip_rate=0.0, ihl_weight=0.3):
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    for _ in range(800):
        x, aw, al = sample_batch(cfg.obs_dim, cfg.num_actions, batch_size, device=device)
        with torch.no_grad():
            rew = oracle.score(x)
            rw = rew[torch.arange(batch_size), aw]
            rl = rew[torch.arange(batch_size), al]
            y = torch.bernoulli(bt_preference(rw, rl)).long()
            margins = rw - rl

        # Apply adversary flips if enabled
        if flip_rate > 0:
            y_tilde, flip_mask = flip_labels_uncertainty_targeting(y, margins, flip_rate)
        else:
            y_tilde, flip_mask = y, torch.zeros_like(y, dtype=torch.bool)

        # Observed winners after corruption
        a_w_obs = torch.where(y_tilde==1, aw, al)
        a_l_obs = torch.where(y_tilde==1, al, aw)

        logits = policy(x)
        logits0 = pi0(x)

        # MODE A: Clean DPO only
        if mode == "CLEAN_DPO":
            loss = dpo_loss(logits, logits0, aw, al, eta=eta)

        # MODE C: DPO on corrupted labels (baseline failure mode)
        elif mode == "ADV_DPO_NOFILTER":
            loss = dpo_loss(logits, logits0, a_w_obs, a_l_obs, eta=eta)

        # MODE B: Adversary + PERFECT filtering using true flip mask + IHL (logit hinge)
        elif mode == "ADV_DPO_PERFECT_IHL":
            clean_idx = (~flip_mask).nonzero(as_tuple=False).squeeze(1)
            corr_idx = (flip_mask).nonzero(as_tuple=False).squeeze(1)
            losses = []
            if clean_idx.numel() > 0:
                losses.append(dpo_loss(logits[clean_idx], logits0[clean_idx], aw[clean_idx], al[clean_idx], eta=eta))
            if corr_idx.numel() > 0:
                losses.append(ihl_weight * ihl_logit_hinge(logits[corr_idx], a_w_obs[corr_idx], margin=1.0))
            loss = sum(losses) if len(losses) > 0 else 0.0

        # MODE D: IHL only (logit hinge) on corrupt indices
        elif mode == "IHL_ONLY_CORR":
            corr_idx = (flip_mask).nonzero(as_tuple=False).squeeze(1)
            if corr_idx.numel() == 0:
                continue
            loss = ihl_weight * ihl_logit_hinge(logits[corr_idx], a_w_obs[corr_idx], margin=1.0)

        else:
            raise ValueError(mode)

        if not isinstance(loss, float):
            opt.zero_grad()
            loss.backward()
            opt.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--flip_rate", type=float, default=0.5)
    parser.add_argument("--ihl_weight", type=float, default=0.3)  # <-- add IHL weight
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config()
    torch.manual_seed(cfg.seed)

    policy = Policy(cfg.obs_dim, cfg.num_actions).to(device)
    pi0 = Policy(cfg.obs_dim, cfg.num_actions).to(device)
    pi0.load_state_dict(policy.state_dict())
    for p in pi0.parameters(): p.requires_grad = False
    oracle = RewardOracle(cfg.obs_dim, cfg.num_actions).to(device)

    # Baseline
    base = metrics(policy, pi0, oracle, cfg.obs_dim, cfg.num_actions, device, eta=args.eta)

    # A: clean-only DPO
    pA = Policy(cfg.obs_dim, cfg.num_actions).to(device); pA.load_state_dict(policy.state_dict())
    train_round(pA, pi0, oracle, cfg, device, 512, "CLEAN_DPO", args.eta, ihl_weight=args.ihl_weight)
    A = metrics(pA, pi0, oracle, cfg.obs_dim, cfg.num_actions, device, eta=args.eta)

    # B: adversary + perfect filter + IHL (logit hinge)
    pB = Policy(cfg.obs_dim, cfg.num_actions).to(device); pB.load_state_dict(policy.state_dict())
    train_round(pB, pi0, oracle, cfg, device, 512, "ADV_DPO_PERFECT_IHL", args.eta, flip_rate=args.flip_rate, ihl_weight=args.ihl_weight)
    B = metrics(pB, pi0, oracle, cfg.obs_dim, cfg.num_actions, device, eta=args.eta)

    # C: adversary + naive DPO
    pC = Policy(cfg.obs_dim, cfg.num_actions).to(device); pC.load_state_dict(policy.state_dict())
    train_round(pC, pi0, oracle, cfg, device, 512, "ADV_DPO_NOFILTER", args.eta, flip_rate=args.flip_rate, ihl_weight=args.ihl_weight)
    C = metrics(pC, pi0, oracle, cfg.obs_dim, cfg.num_actions, device, eta=args.eta)

    # D: IHL only
    pD = Policy(cfg.obs_dim, cfg.num_actions).to(device); pD.load_state_dict(policy.state_dict())
    train_round(pD, pi0, oracle, cfg, device, 512, "IHL_ONLY_CORR", args.eta, flip_rate=args.flip_rate, ihl_weight=args.ihl_weight)
    D = metrics(pD, pi0, oracle, cfg.obs_dim, cfg.num_actions, device, eta=args.eta)

    print(json.dumps({
        "baseline": base,
        "A_clean_dpo": A,
        "B_adv_perfect_filter_ihl": B,
        "C_adv_dpo_nofilter": C,
        "D_ihl_only_corrupt": D
    }, indent=2))

if __name__ == "__main__":
    main()
