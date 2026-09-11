"""
SAC fine-tuning starting from BC policy warm-start.

Usage:
    python src/rl_finetune.py \
        --xml         /path/to/allegro_cube.xml \
        --bc_policy   checkpoints/bc_policy.pt \
        --save_dir    checkpoints/ \
        --timesteps   500000
"""

import argparse
import os
import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from env import AllegroGraspEnv, OBS_DIM, ACT_DIM
from bc_train import BCPolicy


def load_bc_weights_into_sac(model: SAC, bc_path: str):
    """Copy BC MLP weights into SAC actor network."""
    bc = BCPolicy()
    bc.load_state_dict(torch.load(bc_path, map_location='cpu'))

    sac_actor = model.policy.actor
    with torch.no_grad():
        # SB3 actor latent_pi: Linear layers 0,2
        sac_actor.latent_pi[0].weight.copy_(bc.net[0].weight)
        sac_actor.latent_pi[0].bias.copy_(bc.net[0].bias)
        sac_actor.latent_pi[2].weight.copy_(bc.net[2].weight)
        sac_actor.latent_pi[2].bias.copy_(bc.net[2].bias)
        # mu layer
        sac_actor.mu.weight.copy_(bc.net[4].weight)
        sac_actor.mu.bias.copy_(bc.net[4].bias)

    print(f"BC weights loaded from {bc_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml',       required=True)
    parser.add_argument('--bc_policy', default=None)
    parser.add_argument('--save_dir',  default='checkpoints')
    parser.add_argument('--timesteps', type=int, default=500_000)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    vec_env = DummyVecEnv([lambda: AllegroGraspEnv(args.xml)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = SAC(
        'MlpPolicy', vec_env,
        learning_rate=1e-4,
        buffer_size=500_000,
        batch_size=512,
        ent_coef='auto',
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=args.save_dir,
    )

    if args.bc_policy and os.path.exists(args.bc_policy):
        load_bc_weights_into_sac(model, args.bc_policy)

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=args.save_dir,
        name_prefix='sac_dex',
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(os.path.join(args.save_dir, 'sac_dex_final'))
    vec_env.save(os.path.join(args.save_dir, 'vecnorm_dex.pkl'))
    print("Done.")


if __name__ == '__main__':
    main()