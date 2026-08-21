"""
Hard-Target Evaluation for the 4-DOF Robot Arm PPO Agent.

Tests the agent on three challenging target zones:
  - FAR:      arm fully extended, low shoulder angle
  - LOW:      targets close to ground level (z < 5 cm)
  - SIDEWAYS: base at extreme angles (< 30° or > 150°)

Usage:
    python evaluate_hard.py
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from robot_arm_env import RobotArmEnv, forward_kinematics


# ── Configuration ─────────────────────────────────────────────
MODEL_PATH   = "models/best_model/best_model"
NUM_EPISODES = 20   # per zone


# ── Target generators ─────────────────────────────────────────

def target_far() -> np.ndarray:
    """
    Far targets: arm nearly fully extended.
    Shoulder close to horizontal (70-110°), elbow close to straight (70-110°).
    Requires maximum reach.
    """
    while True:
        angles = np.array([
            np.random.uniform(20, 160),    # base: full range
            np.random.uniform(70, 110),    # shoulder: near horizontal
            np.random.uniform(70, 110),    # elbow: near straight
            90.0,
        ])
        target = forward_kinematics(angles)
        reach  = np.linalg.norm(target[:2])   # horizontal distance
        if target[2] > 1.0 and reach > 16.0:  # far = horizontal reach > 16 cm
            return target


def target_low() -> np.ndarray:
    """
    Low targets: close to ground level (z between 1 and 6 cm).
    Requires shoulder to point downward.
    """
    while True:
        angles = np.array([
            np.random.uniform(20, 160),    # base: full range
            np.random.uniform(35, 65),     # shoulder: pointing down
            np.random.uniform(45, 135),    # elbow: free
            90.0,
        ])
        target = forward_kinematics(angles)
        if 1.0 < target[2] < 6.0:         # z between 1 and 6 cm
            return target


def target_sideways() -> np.ndarray:
    """
    Sideways targets: base at extreme angles (< 35° or > 145°).
    Tests the agent's ability to rotate the base far from home.
    """
    while True:
        # Pick either far-left or far-right base angle
        if np.random.random() < 0.5:
            base = np.random.uniform(20, 35)    # far right
        else:
            base = np.random.uniform(145, 160)  # far left

        angles = np.array([
            base,
            np.random.uniform(45, 135),
            np.random.uniform(45, 135),
            90.0,
        ])
        target = forward_kinematics(angles)
        if target[2] > 1.0:
            return target


# ── Evaluation ────────────────────────────────────────────────

class FixedTargetEnv(RobotArmEnv):
    """RobotArmEnv with a custom target generator injected at reset."""

    def __init__(self, target_fn):
        super().__init__()
        self._target_fn = target_fn

    def _random_target(self) -> np.ndarray:
        return self._target_fn()


def evaluate_zone(model: PPO, target_fn, zone_name: str, n_episodes: int) -> dict:
    """
    Evaluate the agent on a specific target zone.

    Args:
        model:       trained PPO model
        target_fn:   callable that returns a target np.ndarray
        zone_name:   display name of the zone
        n_episodes:  number of episodes to run

    Returns:
        dict with keys: zone, successes, mean_reward, mean_distance
    """
    env      = Monitor(FixedTargetEnv(target_fn))
    rewards  = []
    finals   = []
    successes = 0

    print(f"\n── Zone: {zone_name} ({'─' * (40 - len(zone_name))})")

    for ep in range(n_episodes):
        obs, _       = env.reset()
        total_reward = 0.0
        done         = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        success = info.get("success", False)
        dist    = info.get("distance_cm", float("nan"))

        if success:
            successes += 1
        rewards.append(total_reward)
        finals.append(dist)

        print(
            f"  Ep {ep + 1:2d}/{n_episodes} | "
            f"Reward: {total_reward:7.2f} | "
            f"Dist: {dist:5.2f} cm | "
            f"{'✓' if success else '✗'}"
        )

    env.close()

    result = {
        "zone":          zone_name,
        "successes":     successes,
        "n_episodes":    n_episodes,
        "mean_reward":   float(np.mean(rewards)),
        "mean_distance": float(np.mean(finals)),
    }

    print(
        f"  → {successes}/{n_episodes} success | "
        f"Mean reward: {result['mean_reward']:.2f} | "
        f"Mean final dist: {result['mean_distance']:.2f} cm"
    )

    return result


# ── Summary ────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    """Print a side-by-side comparison of all zone results."""
    print("\n" + "=" * 55)
    print(f"{'Zone':<12} {'Success':>10} {'Mean Reward':>13} {'Mean Dist':>11}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['zone']:<12} "
            f"{r['successes']:>4}/{r['n_episodes']:<5} "
            f"{r['mean_reward']:>12.2f} "
            f"{r['mean_distance']:>10.2f} cm"
        )
    print("=" * 55)

    total_success = sum(r["successes"] for r in results)
    total_eps     = sum(r["n_episodes"] for r in results)
    print(f"\nOverall: {total_success}/{total_eps} success across all zones")


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    zones = [
        (target_far,      "FAR"),
        (target_low,      "LOW"),
        (target_sideways, "SIDEWAYS"),
    ]

    results = []
    for target_fn, name in zones:
        result = evaluate_zone(model, target_fn, name, NUM_EPISODES)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()