"""
HandReach-v3 — Model Evaluation

Loads the trained SAC+HER model and evaluates it over
multiple episodes, reporting success rate and mean reward.

Usage:
    python evaluate.py
    python evaluate.py --episodes 50
"""

import argparse
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

# ── Configuration ─────────────────────────────────────────────
ENV_ID       = "HandReach-v3"
MODEL_PATH   = "models/best_model/best_model"
NUM_EPISODES = 20

def evaluate(model, env, n_episodes: int) -> dict:
    rewards   = []
    successes = 0

    for ep in range(n_episodes):
        obs, _       = env.reset()
        total_reward = 0.0
        done         = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        success = info.get("is_success", False)
        if success:
            successes += 1
        rewards.append(total_reward)

        print(
            f"Episode {ep + 1:3d}/{n_episodes} | "
            f"Reward: {total_reward:7.2f} | "
            f"{'✓' if success else '✗'}"
        )

    return {
        "success_rate": successes / n_episodes,
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "successes":    successes,
        "n_episodes":   n_episodes,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--model",    type=str, default=MODEL_PATH)
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = SAC.load(args.model)
    env   = gym.make(ENV_ID)

    print(f"Evaluating over {args.episodes} episodes...\n")
    results = evaluate(model, env, args.episodes)

    print(f"\n=== Results ===")
    print(f"  Success rate: {results['successes']}/{results['n_episodes']} "
          f"({results['success_rate']:.0%})")
    print(f"  Mean reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")

    env.close()


if __name__ == "__main__":
    main()