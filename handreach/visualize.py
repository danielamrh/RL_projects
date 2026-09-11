"""
HandReach-v3 — Live Visualization

Loads the trained SAC+HER model and renders the Shadow Hand
in the MuJoCo viewer in real-time.

Usage:
    python visualize.py
    python visualize.py --episodes 5
"""

import argparse
import time
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

# ── Configuration ─────────────────────────────────────────────
ENV_ID      = "FetchReach-v4"
MODEL_PATH  = "models/best_model/best_model"
STEP_DELAY  = 0.02   # seconds between steps (50 fps)

def run(n_episodes: int, model_path: str) -> None:
    print(f"Loading model from: {model_path}")
    env   = gym.make(ENV_ID, render_mode="human")
    model = SAC.load(model_path, env=env)

    print(f"Running {n_episodes} episodes — close window to stop.")

    for episode in range(1, n_episodes + 1):
        obs, _       = env.reset()
        total_reward = 0.0
        done         = False
        step         = 0

        print(f"\nEpisode {episode}/{n_episodes}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

            env.render()
            time.sleep(STEP_DELAY)

        success = info.get("is_success", False)
        result  = "SUCCESS ✓" if success else "FAILED ✗"
        print(f"  {result} | Steps: {step} | Reward: {total_reward:.2f}")

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model",    type=str, default=MODEL_PATH)
    args = parser.parse_args()
    run(args.episodes, args.model)


if __name__ == "__main__":
    main()