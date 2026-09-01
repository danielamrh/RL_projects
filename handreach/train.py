"""
HandReach-v3 — SAC + HER Training

Shadow Hand reach task: move all 5 fingertips to target positions.
Uses Hindsight Experience Replay (HER) for sparse reward learning.

Usage:
    python train.py
"""

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)

# ── Configuration ─────────────────────────────────────────────
ENV_ID            = "HandReach-v3"
TOTAL_TIMESTEPS   = 1_000_000
EVAL_FREQ         = 10_000
N_EVAL_EPISODES   = 20
MODEL_SAVE_PATH   = "models/best_model"
CHECKPOINT_PATH   = "models/checkpoints"
LOG_DIR           = "logs/"

# HER configuration
N_SAMPLED_GOAL    = 4        # how many relabeled goals per real transition
GOAL_STRATEGY     = "future" # best strategy from the HER paper

def make_env():
    env = gym.make(ENV_ID)
    return env

def train():
    env      = make_env()
    eval_env = make_env()

    model = SAC(
        policy        = "MultiInputPolicy",   # required for dict observations (obs + goal)
        env           = env,
        replay_buffer_class  = HerReplayBuffer,
        replay_buffer_kwargs = {
            "n_sampled_goal":   N_SAMPLED_GOAL,
            "goal_selection_strategy": GOAL_STRATEGY,
        },
        verbose        = 1,
        tensorboard_log= LOG_DIR,
        learning_rate  = 1e-3,
        batch_size     = 256,
        buffer_size    = 1_000_000,
        learning_starts= 1_000,
        gamma          = 0.95,
        tau            = 0.05,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = MODEL_SAVE_PATH,
        log_path             = LOG_DIR,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        verbose              = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq  = 50_000,
        save_path  = CHECKPOINT_PATH,
        name_prefix= "sac_handreach",
    )

    print(f"Training {ENV_ID} for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [eval_callback, checkpoint_callback],
    )

    model.save("models/sac_handreach_final")
    print("Training complete.")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()