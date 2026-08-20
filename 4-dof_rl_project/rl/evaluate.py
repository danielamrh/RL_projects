"""
Evaluation Script for the 4-DOF Robot Arm PPO Agent.

Loads the best saved model and evaluates it over multiple
episodes, printing per-episode results and a final summary.

Usage:
    python evaluate.py
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from robot_arm_env import RobotArmEnv


# ── Configuration ─────────────────────────────────────────────
MODEL_PATH   = "models/best_model/best_model"
NUM_EPISODES = 20


# ── Evaluation ────────────────────────────────────────────────
def evaluate(model: PPO, env: Monitor, n_episodes: int) -> tuple[list, int]:
    """
    Run the agent for n_episodes and collect results.

    Args:
        model:      trained PPO model
        env:        monitored RobotArmEnv
        n_episodes: number of episodes to evaluate

    Returns:
        rewards:   list of total rewards per episode
        successes: number of successful episodes
    """
    rewards   = []
    successes = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        success = info.get("success", False)
        if success:
            successes += 1

        print(
            f"Episode {episode + 1:2d}/{n_episodes} | "
            f"Reward: {total_reward:7.2f} | "
            f"Success: {success}"
        )

    return rewards, successes


def print_summary(rewards: list, successes: int, n_episodes: int) -> None:
    """
    Print a summary of the evaluation results.

    Args:
        rewards:    list of total rewards per episode
        successes:  number of successful episodes
        n_episodes: total number of episodes
    """
    print("\n=== Results ===")
    print(f"  Mean reward:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Success rate: {successes}/{n_episodes}")
    print(f"  Best reward:  {np.max(rewards):.2f}")
    print(f"  Worst reward: {np.min(rewards):.2f}")


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    env   = Monitor(RobotArmEnv(render_mode="human"))

    print(f"Evaluating over {NUM_EPISODES} episodes...\n")
    rewards, successes = evaluate(model, env, NUM_EPISODES)
    print_summary(rewards, successes, NUM_EPISODES)

    env.close()


if __name__ == "__main__":
    main()