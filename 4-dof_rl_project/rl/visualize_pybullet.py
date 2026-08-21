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

from robot_arm_env import RobotArmEnv, JOINT_MIN, JOINT_MAX, forward_kinematics


# ── Configuration ─────────────────────────────────────────────
MODEL_PATH = "models/best_model/best_model"
URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "urdf", "robot_arm.urdf")
STEP_DELAY = 0.05   # seconds between frames (controls playback speed)

# Joint indices in the URDF (order matches joint definitions)
JOINT_BASE     = 0
JOINT_SHOULDER = 1
JOINT_ELBOW    = 2


def get_pybullet_tip(robot_id):
    link_state = p.getLinkState(robot_id, JOINT_ELBOW)
    pos        = np.array(link_state[0])
    orn        = link_state[1]
    tip_offset = np.array(p.rotateVector(orn, [0.11, 0.0, 0.0]))
    return (pos + tip_offset) * 100.0

def verify_kinematics(robot_id, angles_deg):
    tip_fk  = forward_kinematics(angles_deg)
    tip_pbs = get_pybullet_tip(robot_id)
    err     = np.linalg.norm(tip_fk - tip_pbs)
    print(f"FK tip:       {tip_fk}")
    print(f"PyBullet tip: {tip_pbs}")
    print(f"Error:        {err:.3f} cm")

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
    shoulder_rad = -np.radians(angles_deg[1] - 90.0)
    elbow_rad    = -np.radians(angles_deg[2] - 90.0)

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
    Run episodes continuously in real-time PyBullet rendering.
    Automatically resets and starts a new episode on completion.
    """
    episode = 0

    while True:
        obs, _ = env.reset()
        done   = False
        episode += 1

        # Redraw target for new episode
        p.removeAllUserDebugItems()
        draw_target(env.env._target)

        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            set_joint_angles(robot_id, env.env._angles)

            if step == 0:
                verify_kinematics(robot_id, env.env._angles)

            p.stepSimulation()

            print(f"Ep {episode} | Step {step+1:3d} | "
                  f"Dist: {info['distance_cm']:.2f} cm | "
                  f"Success: {info['success']}", end="\r")

            time.sleep(STEP_DELAY)
            step += 1

        result = "SUCCESS" if info["success"] else "FAILED"
        print(f"\nEpisode {episode} {result} after {step} steps.")
        time.sleep(1.0)


def main() -> None:
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    env   = Monitor(RobotArmEnv())

    robot_id = setup_pybullet()

    print("Real-time rendering... (Ctrl+C or close window to exit)")
    try:
        run_episode(model, env, robot_id)
    except (KeyboardInterrupt, p.error):
        print("\nStopped.")
    finally:
        env.close()
        p.disconnect()


if __name__ == "__main__":
    main()