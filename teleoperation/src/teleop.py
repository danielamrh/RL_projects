"""
Teleoperation Main Loop

Pipeline: Webcam → MediaPipe → Retarget → MuJoCo Shadow Hand

Usage:
    python teleop.py
    python teleop.py --env HandManipulate-v1
"""

import argparse
import threading
import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers robotics envs

from detect import run_detection
from retarget import retarget

# ── Configuration ─────────────────────────────────────────────
DEFAULT_ENV = "HandReach-v3"   # Shadow Hand environment

# Global shared state (written by webcam thread, read by MuJoCo thread)
_latest_joints: np.ndarray | None = None
_lock = threading.Lock()

_smoothed_joints: np.ndarray | None = None
ALPHA = 0.2  # smoothing factor: 0=sehr träge, 1=keine Glättung

_debug_counter = 0

def _on_landmarks(landmarks: np.ndarray) -> None:
    """Callback: convert landmarks → joints, store globally."""
    global _latest_joints, _debug_counter, _smoothed_joints
    joints = retarget(landmarks)

    # Exponential moving average
    if _smoothed_joints is None:
        _smoothed_joints = joints.copy()
    else:
        _smoothed_joints = ALPHA * joints + (1 - ALPHA) * _smoothed_joints

    with _lock:
        _latest_joints = _smoothed_joints.copy()

    # Print curl values every 30 frames so you can see open vs closed
    _debug_counter += 1
    if _debug_counter % 30 == 0:
        from retarget import _finger_flex, _thumb_flex
        idx = _finger_flex(landmarks,  5,  6,  7,  8)
        mid = _finger_flex(landmarks,  9, 10, 11, 12)
        thm = _thumb_flex(landmarks)
        print(f"[curl] idx={idx:.2f} mid={mid:.2f} thm={thm:.2f}  "
              f"joints[:4]={joints[:4].round(2)}")


def run_teleop(env_id: str) -> None:
    """
    Opens MuJoCo environment and drives it from webcam landmarks.

    The webcam runs in a background thread at ~30 fps.
    MuJoCo steps at ~30 fps in the main thread.
    """
    print(f"Starting teleoperation: {env_id}")
    print("Show your hand to the webcam. Press Q in the webcam window to quit.\n")

    # Start webcam detection in background thread
    cam_thread = threading.Thread(target=run_detection, args=(_on_landmarks,), daemon=True)
    cam_thread.start()

    # Open MuJoCo environment
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset()

    # Get action space info
    n_actuators = env.action_space.shape[0]
    print(f"Action space: {n_actuators} actuators")
    print(f"  low:  {env.action_space.low.round(3)}")
    print(f"  high: {env.action_space.high.round(3)}")

    try:
        while cam_thread.is_alive():
            with _lock:
                joints = _latest_joints.copy() if _latest_joints is not None else None

            if joints is None:
                # No hand detected yet — send zero action (hold position)
                action = np.zeros(n_actuators, dtype=np.float32)
            else:
                # Map retargeted joints → env action space
                action = _joints_to_action(joints, env, n_actuators)
                if _debug_counter % 30 == 0:
                    print(f"  action: {action.round(2)}")

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except TypeError:
                pass  # GLFW mjv_moveCamera bug in gymnasium

            env.render()

            if terminated or truncated:
                obs, _ = env.reset()
                continue

    finally:
        env.close()
        print("Teleoperation ended.")


def _joints_to_action(joints: np.ndarray, env: gym.Env, n_actuators: int) -> np.ndarray:
    """joints ist bereits in [-1, 1] — direkt clippen und zurückgeben."""
    return np.clip(joints[:n_actuators], -1.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Shadow Hand Teleoperation")
    parser.add_argument("--env", type=str, default=DEFAULT_ENV,
                        help=f"Gymnasium env ID (default: {DEFAULT_ENV})")
    args = parser.parse_args()
    run_teleop(args.env)


if __name__ == "__main__":
    main()