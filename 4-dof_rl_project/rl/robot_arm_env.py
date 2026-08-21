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

# ── Maximum joint angle change per step ─────────────────────────
MAX_DELTA = 6

# ── IMU simulation parameters ─────────────────────────────────
GRAVITY = 9.81  # m/s², standard gravity for simulating accelerometer readings
ACCEL_NOISE_STD = 0.05  # m/s², standard deviation of accelerometer noise
GYRO_NOISE_STD  = 0.01  # rad/s, standard deviation of gyroscope noise
DT = 0.02  # seconds, time step for simulation

# ── Camera simulation parameters ───────────────────────────────
CAMERA_POS_NOISE_STD = 0.5  # cm, standard deviation of camera position noise
CAMERA_POSE_NOISE_STD = 2.0  # degrees, standard deviation of camera orientation noise


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

def simulate_camera_target(target: np.ndarray) -> np.ndarray:
    """
    Simulate a camera-based target detection reading.

    In simulation: returns the true target position plus realistic
    noise to mimic a calibrated webcam detecting a colored object.
    In real deployment: replace with actual OpenCV detection output.

    Args:
        target: true target position [x, y, z] in cm

    Returns:
        np.ndarray: noisy target position [x, y, z] in cm
    """
    noise = np.random.normal(0, CAMERA_POS_NOISE_STD, size=3).astype(np.float32)
    return (target + noise).astype(np.float32)

def simulate_camera_pose(angles_deg: np.ndarray) -> np.ndarray:
    """
    Simulate a camera-based arm pose estimation reading.

    In simulation: returns the true joint angles plus realistic
    noise to mimic a computer vision pose estimator (e.g. MediaPipe).
    In real deployment: replace with actual CV pipeline output.

    Args:
        angles_deg: true joint angles [base, shoulder, elbow] in degrees

    Returns:
        np.ndarray: noisy joint angles [base, shoulder, elbow] normalized to [-1, 1]
    """
    noise = np.random.normal(0, CAMERA_POSE_NOISE_STD, size=3).astype(np.float32)
    noisy_angles = angles_deg[:3] + noise
    noisy_angles = np.clip(noisy_angles, JOINT_MIN[:3], JOINT_MAX[:3])

    # Normalize to [-1, 1] — same as _normalize_angle()
    return (2.0 * (noisy_angles - JOINT_MIN[:3]) / (JOINT_MAX[:3] - JOINT_MIN[:3]) - 1.0).astype(np.float32)


class RobotArmEnv(gym.Env):
    """
    Reach task: move the gripper as close as possible to a random target.

    Observation (21 values):
        [base_norm, shoulder_norm, elbow_norm,      # joint angles normalized to [-1, 1]
         tip_x, tip_y, tip_z,                       # gripper position (cm)
         target_x, target_y, target_z,              # true target position (cm)
         accel_x, accel_y, accel_z,                 # simulated accelerometer (m/s²)
         gyro_x, gyro_y, gyro_z,                    # simulated gyroscope (rad/s)
         cam_target_x, cam_target_y, cam_target_z,  # camera-detected target (cm)
         cam_base_norm, cam_shoulder_norm, cam_elbow_norm]  # camera pose estimate [-1, 1]

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
    MAX_COLLISIONS    = 10    # max allowed collisions before episode ends
    MAX_STEPS         = 500

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ── Action space: 3 joints (base, shoulder, elbow), normalized ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ── Observation space ─────────────────────────────────────────
        # [3 normalized angles] + [3 tip position] + [3 target position] + [6 IMU readings] + [3 camera readings] 
        obs_low  = np.array([-1,-1,-1, -30,-30,0, -30,-30,0, -50,-50,-50, -10,-10,-10, -30,-30,0, -1,-1,-1], dtype=np.float32)
        obs_high = np.array([ 1, 1, 1,  30,30,30,  30,30,30,  50,50,50,   10,10,10,    30,30,30,   1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Internal state
        self._angles  = HOME.copy()  # current joint angles [deg]
        self._target  = np.zeros(3)  # target position [cm]
        self._steps   = 0
        self._collisions_count = 0

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
        cam_target  = simulate_camera_target(self._target)
        cam_pose    = simulate_camera_pose(self._angles)
        return np.concatenate([angles_norm, tip, self._target, imu, cam_target, cam_pose], dtype=np.float32)

    def _get_joint_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 3D coordinates of all joints.

        Returns:
            p0: base origin [0, 0, 0]
            p1: shoulder joint
            p2: elbow joint
            p3: gripper tip
        """
        base_rad     = np.radians(self._angles[0])
        shoulder_rad = np.radians(self._angles[1])
        elbow_rad    = np.radians(self._angles[2])

        s = shoulder_rad - np.pi / 2
        e = elbow_rad    - np.pi / 2

        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.0, L1])

        r2 = L2 * np.cos(s)
        p2 = np.array([r2 * np.cos(base_rad), r2 * np.sin(base_rad), L1 + L2 * np.sin(s)])

        r3 = L2 * np.cos(s) + L3 * np.cos(s + e)
        p3 = np.array([r3 * np.cos(base_rad), r3 * np.sin(base_rad), L1 + L2 * np.sin(s) + L3 * np.sin(s + e)])

        return p0, p1, p2, p3

    def _check_ground_collision(self, p0: np.ndarray, p1: np.ndarray,
                             p2: np.ndarray, p3: np.ndarray) -> bool:
        """
        Check if any joint has gone below the ground plane (z < 0).

        Args:
            p0, p1, p2, p3: 3D positions of base, shoulder, elbow, gripper

        Returns:
            True if a ground collision is detected
        """
        return any(p[2] < 0.0 for p in [p0, p1, p2, p3])
    
    @staticmethod
    def _segment_to_segment_distance(p1: np.ndarray, p2: np.ndarray,
                                    p3: np.ndarray, p4: np.ndarray) -> float:
        """
        Compute the minimum distance between two line segments in 3D.

        Segment A: from p1 to p2
        Segment B: from p3 to p4

        Returns:
            Minimum distance between the two segments
        """
        d1 = p2 - p1   # direction of segment A
        d2 = p4 - p3   # direction of segment B
        r  = p1 - p3

        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)

        if a < 1e-10 and e < 1e-10:   # both segments are points
            return np.linalg.norm(r)

        if a < 1e-10:                  # segment A is a point
            s, t = 0.0, np.clip(f / e, 0.0, 1.0)
        else:
            c = np.dot(d1, r)
            if e < 1e-10:              # segment B is a point
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            else:
                b    = np.dot(d1, d2)
                denom = a * e - b * b
                if denom != 0.0:
                    s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
                else:
                    s = 0.0
                t = (b * s + f) / e
                if t < 0.0:
                    t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
                elif t > 1.0:
                    t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)

        closest_a = p1 + s * d1
        closest_b = p3 + t * d2
        return np.linalg.norm(closest_a - closest_b)

    def _check_self_collision(self, p0: np.ndarray, p1: np.ndarray,
                            p2: np.ndarray, p3: np.ndarray) -> bool:
        """
        Check if non-adjacent arm segments are too close to each other.

        Checks segment S0 (base column) against S2 (forearm),
        since adjacent segments (S0-S1, S1-S2) always share a joint.

        Args:
            p0, p1, p2, p3: 3D positions of base, shoulder, elbow, gripper

        Returns:
            True if a self-collision is detected
        """
        arm_radius = 2.0   # cm — physical arm thickness / 2

        dist_s0_s2 = self._segment_to_segment_distance(p0, p1, p2, p3)
        return dist_s0_s2 < arm_radius

    def _check_base_collision(self, p2: np.ndarray, p3: np.ndarray) -> bool:
        """
        Check if the forearm segment intersects the base cylinder.

        The base is modeled as a vertical cylinder at the origin
        with radius BASE_RADIUS and height L1.

        Args:
            p2: elbow joint position
            p3: gripper tip position

        Returns:
            True if the forearm intersects the base cylinder
        """
        BASE_RADIUS = 3.0   # cm — physical base width / 2

        # Sample points along the forearm segment and test each against
        # the base cylinder (only check within cylinder height)
        for t in np.linspace(0.0, 1.0, 20):
            point = p2 + t * (p3 - p2)
            if point[2] <= L1:                                    # within cylinder height
                if np.sqrt(point[0]**2 + point[1]**2) < BASE_RADIUS:  # within cylinder radius
                    return True
        return False

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
        self._collisions_count = 0

        self._prev_angles = self._angles.copy()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._steps += 1
        self._prev_angles = self._angles.copy()

        # Apply action to update joint angles
        # Clip the action to [-1, 1] and scale to MAX_DELTA degrees per step
        delta = np.clip(action, -1.0, 1.0) * MAX_DELTA
        self._angles[:3] = np.clip(self._angles[:3] + delta, JOINT_MIN[:3], JOINT_MAX[:3])

        p0, p1, p2, p3 = self._get_joint_positions()
        tip = p3

        ground_collision = self._check_ground_collision(p0, p1, p2, p3)
        self_collision   = self._check_self_collision(p0, p1, p2, p3)
        base_collision   = self._check_base_collision(p2, p3)
        collision        = ground_collision or self_collision or base_collision

        distance = np.linalg.norm(tip - self._target)
        reward   = -distance * 0.1 - 0.1

        if collision:
            reward -= 10.0
            self._collisions_count += 1

        success = distance < self.SUCCESS_THRESHOLD and not collision
        if success:
            reward += 100.0

        terminated = success or self._collisions_count >= self.MAX_COLLISIONS
        truncated  = self._steps >= self.MAX_STEPS

        info = {
            "distance_cm":      distance,
            "success":          success,
            "collision":        collision,
            "ground_collision": ground_collision,
            "self_collision":   self_collision,
            "base_collision":   base_collision,
            "tip_position":     tip,
            "target_position":  self._target,
            "collisions":       self._collisions_count,
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