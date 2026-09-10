"""
Hand Retargeting: MediaPipe 21 Landmarks → HandReach-v3 20 Actuators

Verified actuator order (from mujoco.mj_id2name):
 [0]  WRJ1  wrist flex/extend
 [1]  WRJ0  wrist side-side
 [2]  FFJ3  index MCP flex
 [3]  FFJ2  index PIP flex
 [4]  FFJ1  index DIP flex
 [5]  MFJ3  middle MCP flex
 [6]  MFJ2  middle PIP flex
 [7]  MFJ1  middle DIP flex
 [8]  RFJ3  ring MCP flex
 [9]  RFJ2  ring PIP flex
[10]  RFJ1  ring DIP flex
[11]  LFJ4  pinky metacarpal
[12]  LFJ3  pinky MCP flex
[13]  LFJ2  pinky PIP flex
[14]  LFJ1  pinky DIP flex
[15]  THJ4  thumb rotation/abduction
[16]  THJ3  thumb CMC flex
[17]  THJ2  thumb MCP flex
[18]  THJ1  thumb IP flex
[19]  THJ0  thumb DIP flex

MediaPipe landmark indices:
  0=Wrist
  1-4  Thumb  (CMC, MCP, IP, TIP)
  5-8  Index  (MCP, PIP, DIP, TIP)
  9-12 Middle (MCP, PIP, DIP, TIP)
 13-16 Ring   (MCP, PIP, DIP, TIP)
 17-20 Pinky  (MCP, PIP, DIP, TIP)
"""

import numpy as np


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Bend angle at joint b between segments a→b and b→c.
    0 = straight, increases as joint flexes.
    """
    v1 = b - a
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _finger_flex(landmarks: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> float:
    """
    Overall finger curl [0=open, 1=closed].
    Average of PIP and DIP bend angles, normalized by π/2 (90° = fully curled).
    """
    pip_bend = _joint_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
    dip_bend = _joint_angle(landmarks[pip], landmarks[dip], landmarks[tip])
    avg = (pip_bend + dip_bend) / 2.0
    return float(np.clip(avg / (np.pi / 2), 0.0, 1.0))


def _thumb_flex(landmarks: np.ndarray) -> float:
    """Daumen-Beugung über alle 3 Gelenke (CMC, MCP, IP)."""
    cmc_bend = _joint_angle(landmarks[0], landmarks[1], landmarks[2])  # Winkel am CMC
    mcp_bend = _joint_angle(landmarks[1], landmarks[2], landmarks[3])  # Winkel am MCP
    ip_bend  = _joint_angle(landmarks[2], landmarks[3], landmarks[4])  # Winkel am IP
    avg = (cmc_bend + mcp_bend + ip_bend) / 3.0
    return float(np.clip(avg / (np.pi / 2), 0.0, 1.0))


def _thumb_rotation(landmarks: np.ndarray) -> float:
    """
    THJ4: Daumen-Opposition [0=neutral, 1=voll gegenüber].
    Misst wie nah Daumenspitze an Zeigefingergrundgelenk ist.
    """
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]   # Zeigefinger MCP als Palmreferenz
    hand_size = np.linalg.norm(landmarks[9] - landmarks[0]) + 1e-6  # Handgröße
    
    dist = np.linalg.norm(thumb_tip - index_mcp) / hand_size
    # dist ~1.0 = Daumen weit weg (offen), ~0.2 = Daumen nah (opposierend)
    return float(np.clip((1.0 - dist) / 0.8, 0.0, 1.0))


def _act(curl: float) -> float:
    """Map curl [0,1] → action space [-1, +1]. -1=open, +1=closed."""
    return float(-1.0 + 2.0 * np.clip(curl, 0.0, 1.0))


def retarget(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert MediaPipe 21x3 landmarks to HandReach-v3 20 actuator values.

    Args:
        landmarks: (21, 3) float32, normalized MediaPipe output

    Returns:
        action: (20,) float32, all in [-1, +1]
    """
    idx = _finger_flex(landmarks,  5,  6,  7,  8)   # Index
    mid = _finger_flex(landmarks,  9, 10, 11, 12)   # Middle
    rng = _finger_flex(landmarks, 13, 14, 15, 16)   # Ring
    pnk = _finger_flex(landmarks, 17, 18, 19, 20)   # Pinky
    thm = _thumb_flex(landmarks)                     # Thumb
    thm_rot = _thumb_rotation(landmarks)

    return np.array([
        0.0,          # [0]  WRJ1  wrist neutral
        0.0,          # [1]  WRJ0  wrist neutral
        _act(idx),    # [2]  FFJ3  index MCP
        _act(idx),    # [3]  FFJ2  index PIP
        _act(idx),    # [4]  FFJ1  index DIP
        _act(mid),    # [5]  MFJ3  middle MCP
        _act(mid),    # [6]  MFJ2  middle PIP
        _act(mid),    # [7]  MFJ1  middle DIP
        _act(rng),    # [8]  RFJ3  ring MCP
        _act(rng),    # [9]  RFJ2  ring PIP
        _act(rng),    # [10] RFJ1  ring DIP
        -1.0,         # [11] LFJ4  pinky metacarpal (neutral)
        _act(pnk),    # [12] LFJ3  pinky MCP
        _act(pnk),    # [13] LFJ2  pinky PIP
        _act(pnk),    # [14] LFJ1  pinky DIP
        _act(thm_rot),   # [15] THJ4  thumb rotation/abduction
        _act(thm),       # [16] THJ3  thumb CMC
        _act(thm),       # [17] THJ2  thumb MCP
        _act(thm),       # [18] THJ1  thumb IP
        _act(thm),       # [19] THJ0  thumb DIP
    ], dtype=np.float32)


if __name__ == "__main__":
    lm = np.random.rand(21, 3).astype(np.float32)
    q = retarget(lm)
    print("Action:", q.round(3))
    assert q.shape == (20,)
    print("OK")