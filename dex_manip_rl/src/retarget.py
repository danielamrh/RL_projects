"""
Retarget MediaPipe hand landmarks (21 × 3) → Allegro joint angles (16).

MediaPipe landmark indices (relevant ones):
  0  = wrist
  5,6,7,8   = index  MCP, PIP, DIP, tip
  9,10,11,12 = middle MCP, PIP, DIP, tip
 13,14,15,16 = ring   MCP, PIP, DIP, tip
  1,2,3,4   = thumb  CMC, MCP, IP,  tip

Allegro joint order (16):
  0-3:  index  (spread, MCP, PIP, DIP)
  4-7:  middle (spread, MCP, PIP, DIP)
  8-11: ring   (spread, MCP, PIP, DIP)
 12-15: thumb  (rot, MCP, IP, spread)

Strategy: estimate bend angle from landmark chain dot products,
then scale to Allegro joint limits.
"""

import numpy as np
from env import JOINT_MIN, JOINT_MAX

# MediaPipe landmark groups: (MCP, PIP, DIP, TIP)
MP_CHAINS = {
    'index':  [5,  6,  7,  8],
    'middle': [9,  10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'thumb':  [1,  2,  3,  4],
}
WRIST_IDX = 0


def _angle_between(a, b, c):
    """Angle at vertex b (points a-b-c)."""
    ba = a - b; ca = c - b
    ba /= (np.linalg.norm(ba) + 1e-8)
    ca /= (np.linalg.norm(ca) + 1e-8)
    return np.arccos(np.clip(np.dot(ba, ca), -1.0, 1.0))


def landmarks_to_allegro(lm_flat: np.ndarray) -> np.ndarray:
    """
    lm_flat : (63,) — 21 landmarks × (x,y,z)
    returns  : (16,) Allegro joint angles in radians
    """
    lm = lm_flat.reshape(21, 3)
    angles = np.zeros(16, np.float32)

    # Each finger: 4 joints, first is lateral spread (set to 0), rest are bends
    for fi, (fname, chain) in enumerate(MP_CHAINS.items()):
        mcp, pip, dip, tip = [lm[i] for i in chain]
        wrist = lm[WRIST_IDX]

        bend_mcp = _angle_between(wrist, mcp, pip)
        bend_pip = _angle_between(mcp,   pip, dip)
        bend_dip = _angle_between(pip,   dip, tip)

        base = fi * 4
        if fname == 'thumb':
            # thumb joint 0: rotation (use wrist→CMC angle as proxy)
            angles[base]     = 0.0
            angles[base + 1] = np.clip(bend_mcp, 0, np.pi)
            angles[base + 2] = np.clip(bend_pip, 0, np.pi)
            angles[base + 3] = 0.0
        else:
            angles[base]     = 0.0           # spread
            angles[base + 1] = np.clip(bend_mcp, 0, np.pi)
            angles[base + 2] = np.clip(bend_pip, 0, np.pi)
            angles[base + 3] = np.clip(bend_dip, 0, np.pi)

    # Clip to Allegro limits
    return np.clip(angles, JOINT_MIN, JOINT_MAX)


def retarget_demo(demo_path: str) -> np.ndarray:
    """
    demo_path : .npy file with shape (T, 63)
    returns   : (T, 16) joint angle sequence
    """
    demo = np.load(demo_path)  # (T, 63)
    return np.array([landmarks_to_allegro(f) for f in demo], np.float32)


if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', required=True, help='Input .npy landmarks')
    parser.add_argument('--out',  required=True, help='Output .npy joint angles')
    args = parser.parse_args()

    joints = retarget_demo(args.demo)
    np.save(args.out, joints)
    print(f"Retargeted {joints.shape[0]} frames → {args.out}")