"""
PyBullet Visualizer for the 4-DOF Robot Arm PPO Agent.

Loads the trained model, runs one episode, and renders
the arm movement in a real-time 3D PyBullet window.

Usage:
    python visualize_pybullet.py
"""

import os

import time
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from robot_arm_env import RobotArmEnv, JOINT_MIN, JOINT_MAX


# ── Configuration ─────────────────────────────────────────────
MODEL_PATH = "models/best_model/best_model"
URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "urdf", "robot_arm.urdf")
STEP_DELAY = 0.05   # seconds between frames (controls playback speed)

# Joint indices in the URDF (order matches joint definitions)
JOINT_BASE     = 0
JOINT_SHOULDER = 1
JOINT_ELBOW    = 2


# ── PyBullet Setup ────────────────────────────────────────────
def setup_pybullet() -> int:
    """
    Initialize PyBullet, load the ground plane and robot URDF.

    Returns:
        robot_id: PyBullet body ID of the loaded robot arm
    """
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0], useFixedBase=True)

    # Camera position for good view
    p.resetDebugVisualizerCamera(
        cameraDistance    = 0.5,
        cameraYaw         = 45,
        cameraPitch       = -30,
        cameraTargetPosition = [0, 0, 0.15],
    )

    return robot_id

# ── Joint Control ─────────────────────────────────────────────
def set_joint_angles(robot_id: int, angles_deg: np.ndarray) -> None:
    """
    Apply environment joint angles to the PyBullet robot.

    Converts from environment convention (degrees, 0-180)
    to URDF convention (radians, relative to joint origin).

    Args:
        robot_id:   PyBullet body ID
        angles_deg: [base, shoulder, elbow, gripper] in degrees
    """
    # Base: 0-180° → 0 to π
    base_rad = np.radians(angles_deg[0])

    # Shoulder/Elbow: 90° = horizontal = 0 in URDF frame
    shoulder_rad = np.radians(angles_deg[1] - 90.0)
    elbow_rad    = np.radians(angles_deg[2] - 90.0)

    p.resetJointState(robot_id, JOINT_BASE,     base_rad)
    p.resetJointState(robot_id, JOINT_SHOULDER, shoulder_rad)
    p.resetJointState(robot_id, JOINT_ELBOW,    elbow_rad)

# ── Target Visualization ──────────────────────────────────────
def draw_target(target_cm: np.ndarray) -> None:
    """
    Draw the target position as a red sphere in the PyBullet scene.

    Args:
        target_cm: [x, y, z] target position in cm
    """
    target_m = target_cm / 100.0   # cm → meters

    visual_id = p.createVisualShape(
        shapeType    = p.GEOM_SPHERE,
        radius       = 0.02,
        rgbaColor    = [1, 0, 0, 0.8],   # red, slightly transparent
    )

    p.createMultiBody(
        baseMass          = 0,
        baseVisualShapeIndex = visual_id,
        basePosition      = target_m.tolist(),
    )

# ── Episode Playback ──────────────────────────────────────────
def run_episode(model: PPO, env: Monitor, robot_id: int) -> None:
    """
    Run one episode and render each step in PyBullet.

    Args:
        model:    trained PPO model
        env:      wrapped RobotArmEnv
        robot_id: PyBullet body ID
    """
    obs, _ = env.reset()
    done   = False

    draw_target(env.env._target)

    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        set_joint_angles(robot_id, env.env._angles)
        p.stepSimulation()

        distance = info["distance_cm"]
        print(f"Step {step+1:3d} | Distance: {distance:.2f} cm | "
              f"Collision: {info['collision']} | Success: {info['success']}")

        time.sleep(STEP_DELAY)
        step += 1

    print(f"\nEpisode finished after {step} steps.")


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    env   = Monitor(RobotArmEnv())

    robot_id = setup_pybullet()

    print("Running episode... (close PyBullet window to exit)")
    run_episode(model, env, robot_id)

    input("Press Enter to close...")
    p.disconnect()
    env.close()


if __name__ == "__main__":
    main()