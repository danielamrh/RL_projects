"""
Custom Gymnasium Environment for the Spacnana SNAM1500 4-DOF Robot Arm
Hardware: ESP32-S3 + 4x SG90 servos

Joints:
  - Joint 0 (base):     rotates left/right around Z axis
  - Joint 1 (shoulder): lifts/lowers the upper arm
  - Joint 2 (elbow):    bends the forearm
  - Joint 3 (gripper):  opens/closes (ignored during reach task)

Task: Move the gripper to a random target position (reach task).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Robot geometry (measured in cm from assembled arm photos) ──
L1 = 8.0    # height of shoulder joint above base
L2 = 10.0   # upper arm length (shoulder → elbow)
L3 = 11.0   # forearm + gripper length (elbow → tip)

# ── Joint limits in degrees ────────────────────────────────────
JOINT_MIN = np.array([0.0,  30.0,  30.0,  60.0])   # [base, shoulder, elbow, gripper]
JOINT_MAX = np.array([180.0, 150.0, 150.0, 120.0])
HOME      = np.array([90.0,  90.0,  90.0,  90.0])


GRAVITY = 9.81  # m/s², standard gravity for simulating accelerometer readings
ACCEL_NOISE_STD = 0.05  # m/s², standard deviation of accelerometer noise
GYRO_NOISE_STD  = 0.01  # rad/s, standard deviation of gyroscope noise
DT = 0.02  # seconds, time step for simulation

def forward_kinematics(angles_deg: np.ndarray) -> np.ndarray:
    """
    Compute the 3D position (x, y, z) of the gripper tip in cm.

    Coordinate system:
      - Origin at base center on the ground
      - Z points upward
      - X/Y form the horizontal plane

    Args:
        angles_deg: [base, shoulder, elbow, gripper] in degrees

    Returns:
        np.ndarray: [x, y, z] position of gripper tip in cm
    """
    base_rad     = np.radians(angles_deg[0])
    shoulder_rad = np.radians(angles_deg[1])
    elbow_rad    = np.radians(angles_deg[2])

    # Work in the vertical plane defined by the base rotation.
    # Shoulder angle: 90° = horizontal, 180° = pointing up, 0° = pointing down.
    # Convert to standard math angle (0 = horizontal forward).
    s = shoulder_rad - np.pi / 2  # shoulder relative to horizontal
    e = elbow_rad    - np.pi / 2  # elbow relative to upper arm

    # Horizontal reach and vertical position in the sagittal plane
    r = L2 * np.cos(s) + L3 * np.cos(s + e)
    z = L1 + L2 * np.sin(s) + L3 * np.sin(s + e)

    # Rotate by base angle into 3D
    x = r * np.cos(base_rad)
    y = r * np.sin(base_rad)

    return np.array([x, y, z], dtype=np.float32)

def simulate_imu(angles_deg: np.ndarray, prev_angles_deg: np.ndarray) -> np.ndarray:
    """
    Simulate IMU readings (accelerometer + gyroscope) from joint angles.

    Computes angular velocity from angle changes and adds realistic
    sensor noise to mimic a real MPU-6050.

    Args:
        angles_deg:      current joint angles in degrees
        prev_angles_deg: joint angles from previous timestep

    Returns:
        np.ndarray: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    """
    # Compute gripper tip positions
    tip_current  = forward_kinematics(angles_deg)
    tip_previous = forward_kinematics(prev_angles_deg)

    # Approximate linear acceleration (cm/s² → m/s²) + gravity on Z
    accel = (tip_current - tip_previous) / (DT * 100.0)
    accel[2] += GRAVITY

    # Approximate angular velocity from angle changes (deg/s → rad/s)
    delta_angles = np.radians(angles_deg[:3] - prev_angles_deg[:3])
    gyro = delta_angles / DT

    # Add sensor noise
    accel += np.random.normal(0, ACCEL_NOISE_STD, size=3)
    gyro  += np.random.normal(0, GYRO_NOISE_STD,  size=3)

    return np.concatenate([accel, gyro]).astype(np.float32)


class RobotArmEnv(gym.Env):
    """
    Reach task: move the gripper as close as possible to a random target.

    Observation (9 values):
        [base_norm, shoulder_norm, elbow_norm,   # current joint angles normalized to [-1, 1]
         tip_x, tip_y, tip_z,                    # current gripper position (cm)
         target_x, target_y, target_z]           # target position (cm)

    Action (3 values, continuous [-1, 1]):
        Desired joint angles for [base, shoulder, elbow], normalized.
        Gripper is excluded from the reach task.

    Reward:
        - Main reward: -distance to target (always negative, 0 = perfect)
        - Success bonus: +100 when distance < SUCCESS_THRESHOLD
        - Time penalty: -0.1 per step (encourages speed)
    """

    metadata = {"render_modes": ["human"]}

    SUCCESS_THRESHOLD = 2.0   # cm — gripper within 2cm of target = success
    MAX_STEPS         = 200

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ── Action space: 3 joints (base, shoulder, elbow), normalized ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ── Observation space ─────────────────────────────────────────
        # [3 normalized angles] + [3 tip position] + [3 target position] + [6 IMU readings]
        obs_low  = np.array([-1, -1, -1, -30, -30,  0, -30, -30,  0, -50, -50, -50, -10, -10, -10], dtype=np.float32)
        obs_high = np.array([ 1,  1,  1,  30,  30, 30,  30,  30, 30,  50,  50,  50,  10,  10,  10], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Internal state
        self._target  = np.zeros(3)     # target position [cm]
        self._steps   = 0

        # Previous angles for IMU simulation 
        self._prev_angles = HOME.copy()

    # ── Helpers ───────────────────────────────────────────────────────

    def _normalize_angle(self, angles_deg: np.ndarray) -> np.ndarray:
        """Normalize joint angles from [JOINT_MIN, JOINT_MAX] to [-1, 1]."""
        return 2.0 * (angles_deg - JOINT_MIN[:3]) / (JOINT_MAX[:3] - JOINT_MIN[:3]) - 1.0

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert action from [-1, 1] to joint angles in degrees."""
        return JOINT_MIN[:3] + (action + 1.0) / 2.0 * (JOINT_MAX[:3] - JOINT_MIN[:3])

    def _get_obs(self) -> np.ndarray:
        tip         = forward_kinematics(self._angles)
        angles_norm = self._normalize_angle(self._angles[:3])
        imu         = simulate_imu(self._angles, self._prev_angles)
        return np.concatenate([angles_norm, tip, self._target, imu], dtype=np.float32)

    def _random_target(self) -> np.ndarray:
        """Sample a reachable target position."""
        while True:
            # Random angles in a safe range
            rand_angles = np.array([
                np.random.uniform(20, 160),   # base
                np.random.uniform(45, 135),   # shoulder
                np.random.uniform(45, 135),   # elbow
                90.0,                          # gripper (fixed)
            ])
            target = forward_kinematics(rand_angles)
            if target[2] > 1.0:  # target must be above ground
                return target

    # ── Core Gymnasium methods ────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start near home position with small random perturbation
        self._angles = HOME.copy()
        self._angles[:3] += self.np_random.uniform(-5, 5, size=3)
        self._angles = np.clip(self._angles, JOINT_MIN, JOINT_MAX)

        self._target = self._random_target()
        self._steps  = 0

        self._prev_angles = self._angles.copy()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._steps += 1

        # Save previous angles before moving
        self._prev_angles = self._angles.copy()

        # Apply action: set joint angles directly
        new_angles_deg = self._denormalize_action(np.clip(action, -1.0, 1.0))
        self._angles[:3] = np.clip(new_angles_deg, JOINT_MIN[:3], JOINT_MAX[:3])

        # Compute current gripper position
        tip = forward_kinematics(self._angles)

        # Distance to target
        distance = np.linalg.norm(tip - self._target)

        # Reward
        reward = -distance * 0.1          # distance penalty (scaled)
        reward -= 0.1                     # time penalty

        # Success
        success = distance < self.SUCCESS_THRESHOLD
        if success:
            reward += 100.0

        # Done
        terminated = success
        truncated  = self._steps >= self.MAX_STEPS

        info = {
            "distance_cm": distance,
            "success": success,
            "tip_position": tip,
            "target_position": self._target,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            tip = forward_kinematics(self._angles)
            dist = np.linalg.norm(tip - self._target)
            print(
                f"Step {self._steps:3d} | "
                f"Angles: {self._angles[:3].round(1)} | "
                f"Tip: {tip.round(1)} cm | "
                f"Target: {self._target.round(1)} cm | "
                f"Dist: {dist:.2f} cm"
            )

    def close(self):
        pass

    def angles_to_servo(self) -> np.ndarray:
        """
        Returns the current joint angles ready to send to the ESP32.
        Output: [base, shoulder, elbow, gripper] in degrees (0-180).
        """
        return self._angles.copy()


# ── Quick sanity check ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Forward Kinematics Test ===")
    test_angles = [
        ([90, 90, 90, 90],  "Home position"),
        ([90, 45, 90, 90],  "Shoulder raised"),
        ([0,  90, 90, 90],  "Base rotated left"),
        ([90, 90, 45, 90],  "Elbow bent"),
    ]
    for angles, label in test_angles:
        pos = forward_kinematics(np.array(angles, dtype=float))
        print(f"  {label:25s} → x={pos[0]:6.2f} y={pos[1]:6.2f} z={pos[2]:6.2f} cm")

    print("\n=== Environment Test ===")
    env = RobotArmEnv(render_mode="human")
    obs, _ = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space:      {env.action_space}")
    print(f"Target position:   {env._target.round(2)} cm")
    print()

    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            break

    print(f"\nTotal reward after {step+1} random steps: {total_reward:.2f}")
    env.close()