"""
PPO Training Script for the 4-DOF Robot Arm Reach Task.

Trains a PPO agent using Stable-Baselines3 on the custom
RobotArmEnv gymnasium environment. Saves the best model
and periodic checkpoints during training.

Usage:
    python train.py

Monitor training:
    tensorboard --logdir logs/
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from robot_arm_env import RobotArmEnv


# ── Configuration ─────────────────────────────────────────────
NUM_ENVS        = 8
TOTAL_TIMESTEPS = 1_000_000

LOG_DIR         = "logs/"
CHECKPOINT_DIR  = "checkpoints/"
BEST_MODEL_PATH = "models/best_model"
FINAL_MODEL_PATH= "models/ppo_robot_arm_final"

PPO_HYPERPARAMS = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    n_steps       = 1024,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    policy_kwargs = dict(net_arch=[128, 128]),
)

EVAL_FREQ      = 10_000
N_EVAL_EPISODES= 20
CHECKPOINT_FREQ= 50_000


# ── Setup ─────────────────────────────────────────────────────
def setup_directories() -> None:
    """Create output directories if they don't exist."""
    for path in [LOG_DIR, CHECKPOINT_DIR, "models"]:
        os.makedirs(path, exist_ok=True)


def create_environments() -> tuple:
    """
    Create training and evaluation environments.

    Returns:
        train_env: vectorized environment for training
        eval_env:  single monitored environment for evaluation
    """
    train_env = make_vec_env(RobotArmEnv, n_envs=NUM_ENVS)
    eval_env  = Monitor(RobotArmEnv())
    return train_env, eval_env


def create_callbacks(eval_env: Monitor) -> list:
    """
    Create training callbacks for evaluation and checkpointing.

    Args:
        eval_env: environment used for periodic evaluation

    Returns:
        list of callbacks to pass to model.learn()
    """
    eval_callback = EvalCallback(
        eval_env             = eval_env,
        best_model_save_path = BEST_MODEL_PATH,
        log_path             = LOG_DIR,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        verbose              = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "ppo_robot_arm",
        verbose     = 1,
    )

    return [eval_callback, checkpoint_callback]


# ── Training ──────────────────────────────────────────────────
def train(model: PPO, callbacks: list) -> PPO:
    """
    Run the PPO training loop.

    Args:
        model:     PPO model to train
        callbacks: list of SB3 callbacks

    Returns:
        trained PPO model
    """
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"  Environments: {NUM_ENVS}")
    print(f"  TensorBoard:  tensorboard --logdir {LOG_DIR}")
    print()

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = callbacks,
        progress_bar    = True,
    )

    model.save(FINAL_MODEL_PATH)
    print(f"\nTraining complete. Model saved to '{FINAL_MODEL_PATH}.zip'")

    return model


# ── Quick post-training evaluation ────────────────────────────
def evaluate(model: PPO, eval_env: Monitor, n_episodes: int = 20) -> None:
    """
    Run a quick evaluation after training and print results.

    Args:
        model:      trained PPO model
        eval_env:   monitored environment
        n_episodes: number of evaluation episodes
    """
    print(f"\nEvaluating over {n_episodes} episodes...")

    rewards   = []
    successes = 0

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if info.get("success"):
            successes += 1

    print(f"  Mean reward:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Success rate: {successes}/{n_episodes}")


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    setup_directories()

    train_env, eval_env = create_environments()
    callbacks           = create_callbacks(eval_env)

    model = PPO(
        env             = train_env,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
        **PPO_HYPERPARAMS,
    )

    model = train(model, callbacks)
    evaluate(model, eval_env)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()