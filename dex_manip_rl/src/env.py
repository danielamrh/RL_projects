"""
MuJoCo environment: Allegro Hand + Cube grasping.

Obs  : hand joint positions (16) + fingertip positions (4×3) + cube pose (7) = 35
Act  : target joint angles (16), scaled delta
Reward: grasp_reward + lift_reward + alive_bonus
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# Allegro hand has 16 joints (4 fingers × 4 DOF)
N_JOINTS   = 16
N_FINGERS  = 4
OBS_DIM    = N_JOINTS + N_FINGERS * 3 + 7   # 35
ACT_DIM    = N_JOINTS

# Joint limits (Allegro Hand, radians)
JOINT_MIN = np.array([
    -0.47, -0.196, -0.174, -0.227,   # index
    -0.47, -0.196, -0.174, -0.227,   # middle
    -0.47, -0.196, -0.174, -0.227,   # ring
     0.263, -0.105, -0.189, -0.162,  # thumb
], dtype=np.float32)

JOINT_MAX = np.array([
     0.47, 1.61, 1.709, 1.618,
     0.47, 1.61, 1.709, 1.618,
     0.47, 1.61, 1.709, 1.618,
     1.396, 1.163, 1.644, 1.719,
], dtype=np.float32)

FINGERTIP_BODIES = ['index_tip', 'middle_tip', 'ring_tip', 'thumb_tip']
CUBE_LIFT_HEIGHT  = 0.05   # metres above table — counts as "lifted"


class AllegroGraspEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, xml_path: str, render_mode=None):
        super().__init__()
        self.xml_path    = xml_path
        self.render_mode = render_mode
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)

        self._find_ids()

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

        self._init_cube_pos = None
        self.renderer = None
        if render_mode == 'rgb_array':
            self.renderer = mujoco.Renderer(self.m, height=480, width=640)

    # ── internals ────────────────────────────────────────────────────────────

    def _find_ids(self):
        self._joint_ids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f'joint_{i}')
            for i in range(N_JOINTS)
        ]
        self._fingertip_ids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FINGERTIP_BODIES
        ]
        self._cube_body_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, 'cube'
        )
        self._cube_joint_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, 'cube_free'
        )

    def _obs(self):
        joint_pos   = self.d.qpos[self._joint_ids].astype(np.float32)
        fingertips  = np.array([
            self.d.xpos[fid] for fid in self._fingertip_ids
        ], np.float32).flatten()
        cube_pos    = self.d.xpos[self._cube_body_id].astype(np.float32)
        cube_quat   = self.d.xquat[self._cube_body_id].astype(np.float32)
        return np.concatenate([joint_pos, fingertips, cube_pos, cube_quat])

    def _cube_height(self):
        return float(self.d.xpos[self._cube_body_id][2])

    # ── gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)

        # Randomise cube XY position slightly
        cube_qadr = self.m.jnt_qposadr[self._cube_joint_id]
        self.d.qpos[cube_qadr]     += self.np_random.uniform(-0.02, 0.02)
        self.d.qpos[cube_qadr + 1] += self.np_random.uniform(-0.02, 0.02)
        mujoco.mj_forward(self.m, self.d)

        self._init_cube_pos = self.d.xpos[self._cube_body_id].copy()
        return self._obs(), {}

    def step(self, action):
        # Map [-1,1] → joint range
        target = 0.5 * (action + 1.0) * (JOINT_MAX - JOINT_MIN) + JOINT_MIN
        self.d.ctrl[:N_JOINTS] = target
        mujoco.mj_step(self.m, self.d)

        obs    = self._obs()
        reward = self._reward()
        done   = self._cube_height() > self._init_cube_pos[2] + CUBE_LIFT_HEIGHT

        return obs, reward, done, False, {}

    def _reward(self):
        cube_pos = self.d.xpos[self._cube_body_id]
        tips     = np.array([self.d.xpos[fid] for fid in self._fingertip_ids])

        # Distance from each fingertip to cube centre
        grasp_r = -float(np.mean(np.linalg.norm(tips - cube_pos, axis=1)))

        # Lift reward
        lift_r  = float(cube_pos[2] - self._init_cube_pos[2]) * 10.0

        return grasp_r + lift_r + 0.01   # alive bonus

    def render(self):
        if self.renderer is None:
            return None
        self.renderer.update_scene(self.d)
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()