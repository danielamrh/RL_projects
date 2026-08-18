"""
Visualizer for the 4-DOF Robot Arm PPO Agent.

Loads the trained model, runs one episode, and saves
the arm movement as a 3D animation (MP4).

Usage:
    python visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from robot_arm_env import RobotArmEnv, L1, L2, L3


# ── Configuration ─────────────────────────────────────────────
MODEL_PATH  = "models/best_model/best_model"
OUTPUT_PATH = "robot_arm_simulation.gif"
FPS         = 15
INTERVAL_MS = 100   # milliseconds between frames in the animation

# ── Geometry ──────────────────────────────────────────────────
def get_joint_positions(angles_deg: np.ndarray) -> tuple:
    """
    Compute the 3D coordinates of each joint given joint angles.

    Returns the positions of:
        p0 - base (origin)
        p1 - shoulder
        p2 - elbow
        p3 - gripper tip

    Args:
        angles_deg: [base, shoulder, elbow, gripper] in degrees

    Returns:
        tuple of four np.ndarray, each of shape (3,)
    """
    base_rad     = np.radians(angles_deg[0])
    shoulder_rad = np.radians(angles_deg[1])
    elbow_rad    = np.radians(angles_deg[2])

    # Convert servo angles to standard math angles (0 = horizontal)
    s = shoulder_rad - np.pi / 2
    e = elbow_rad    - np.pi / 2

    p0 = np.array([0.0, 0.0, 0.0])                          # base
    p1 = np.array([0.0, 0.0, L1])                           # shoulder

    r2 = L2 * np.cos(s)
    p2 = np.array([
        r2 * np.cos(base_rad),
        r2 * np.sin(base_rad),
        L1 + L2 * np.sin(s),
    ])

    r3 = L2 * np.cos(s) + L3 * np.cos(s + e)
    p3 = np.array([
        r3 * np.cos(base_rad),
        r3 * np.sin(base_rad),
        L1 + L2 * np.sin(s) + L3 * np.sin(s + e),
    ])

    return p0, p1, p2, p3

# ── Episode recording ─────────────────────────────────────────
def record_episode(model: PPO, env: Monitor) -> list[dict]:
    """
    Run one episode with the trained model and record each step.

    Args:
        model: trained PPO model
        env:   wrapped RobotArmEnv

    Returns:
        List of dicts with keys: angles, target, tip
    """
    obs, _ = env.reset()
    done   = False
    frames = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frames.append({
            "angles": env.env._angles.copy(),
            "target": env.env._target.copy(),
            "tip":    info["tip_position"].copy(),
        })

    return frames

# ── Animation ─────────────────────────────────────────────────
def build_animation(frames: list[dict]) -> animation.FuncAnimation:
    """
    Build a 3D matplotlib animation from the recorded episode.

    Args:
        frames: list of recorded steps from record_episode()

    Returns:
        FuncAnimation object ready to save or display
    """
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="3d")
    tip_trail_x, tip_trail_y, tip_trail_z = [], [], []

    def setup_axes():
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_zlim(0,   35)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_zlabel("Z (cm)")

    def update(frame_idx: int):
        ax.cla()
        setup_axes()

        angles = frames[frame_idx]["angles"]
        target = frames[frame_idx]["target"]
        p0, p1, p2, p3 = get_joint_positions(angles)

        # Draw arm segments
        xs = [p0[0], p1[0], p2[0], p3[0]]
        ys = [p0[1], p1[1], p2[1], p3[1]]
        zs = [p0[2], p1[2], p2[2], p3[2]]
        ax.plot(xs, ys, zs, "o-", color="royalblue", linewidth=3, markersize=8, label="Arm")

        # Draw target
        ax.plot(*target, "r*", markersize=15, label="Target")

        # Draw gripper trail
        tip_trail_x.append(p3[0])
        tip_trail_y.append(p3[1])
        tip_trail_z.append(p3[2])
        ax.plot(tip_trail_x, tip_trail_y, tip_trail_z,
                "g--", linewidth=1, alpha=0.5, label="Tip trail")

        # Title with current distance to target
        distance = np.linalg.norm(p3 - target)
        ax.set_title(
            f"Step {frame_idx + 1}/{len(frames)} | "
            f"Distance to target: {distance:.2f} cm",
            fontsize=11,
        )
        ax.legend(loc="upper left")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=INTERVAL_MS,
        repeat=False,
    )

    return ani

# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    env   = Monitor(RobotArmEnv())

    print("Recording episode...")
    frames = record_episode(model, env)
    print(f"  Recorded {len(frames)} steps")
    print(f"  Final distance: {np.linalg.norm(frames[-1]['tip'] - frames[-1]['target']):.2f} cm")

    print("Building animation...")
    ani = build_animation(frames)

    print(f"Saving to {OUTPUT_PATH} ...")
    ani.save(OUTPUT_PATH, writer="pillow", fps=FPS)
    print("Done!")

    env.close()
    plt.show()


if __name__ == "__main__":
    main()