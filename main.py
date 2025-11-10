
import argparse
from configs import Config
from trainer import run

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--obs-dim', type=int, default=32)
    p.add_argument('--num-actions', type=int, default=8)
    p.add_argument('--eta', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--flip-rate', type=float, default=0.2)
    p.add_argument('--ihl-weight', type=float, default=0)
    p.add_argument('--filter-tau', type=float, default=0.3)
    p.add_argument('--retro-K', type=int, default=200)
    p.add_argument('--retro-passes', type=int, default=1)
    return p.parse_args()

def main():
    args = get_args()
    cfg = Config(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        obs_dim=args.obs_dim,
        num_actions=args.num_actions,
        eta=args.eta,
        lr=args.lr,
        flip_rate=args.flip_rate,
        ihl_weight=args.ihl_weight,
        filter_tau=args.filter_tau,
        retro_K=args.retro_K,
        retro_passes=args.retro_passes,
    )
    result = run(cfg)
    print("Run complete. Logs at:", result['log_path'])

if __name__ == "__main__":
    main()
