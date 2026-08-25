"""
Deploy Script — Send PPO Agent Angles to ESP32 via Serial.

Loads the trained model, runs the agent, and after each step
sends the computed joint angles to the ESP32 over USB Serial.

The ESP32 firmware must be flashed with the Serial-control version
(firmware/src/main.cpp) before running this script.

Protocol (sent from Python → ESP32):
    "<base>,<shoulder>,<elbow>,<gripper>\\n"
    Example: "90,75,110,90\\n"

Usage:
    python deploy.py --port COM3
    python deploy.py --port COM3 --flip-elbow   # if physical elbow is reversed
    python deploy.py --dry-run                  # run without hardware (test only)
"""

import argparse
import time
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from robot_arm_env import RobotArmEnv


# ── Configuration ─────────────────────────────────────────────
MODEL_PATH    = "models/best_model/best_model"
BAUD_RATE     = 115200
STEP_DELAY    = 0.4    # seconds between steps (servo needs time to move)
NUM_EPISODES  = 5      # how many episodes to run on the real robot


# ── Serial helpers ────────────────────────────────────────────

def open_serial(port: str, baud: int):
    """Open the Serial connection to the ESP32."""
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed. Run: pip install pyserial")
        sys.exit(1)

    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)   # wait for ESP32 to reboot after Serial connect
        print(f"Connected to ESP32 on {port} at {baud} baud.")
        return ser
    except Exception as e:
        print(f"ERROR: Could not open {port}: {e}")
        sys.exit(1)


def send_angles(ser, angles_deg: np.ndarray, flip_elbow: bool = False) -> None:
    """
    Send joint angles to the ESP32.

    Args:
        ser:         open Serial connection (or None in dry-run mode)
        angles_deg:  [base, shoulder, elbow, gripper] in degrees
        flip_elbow:  if True, sends 180 - elbow (compensates reversed servo)
    """
    base     = int(round(angles_deg[0]))
    shoulder = int(round(angles_deg[1]))
    elbow    = int(round(angles_deg[2]))
    gripper  = int(round(angles_deg[3]))

    if flip_elbow:
        elbow = 180 - elbow

    # Clamp to safe range
    base     = max(0,   min(180, base))
    shoulder = max(30,  min(150, shoulder))
    elbow    = max(30,  min(150, elbow))
    gripper  = max(60,  min(120, gripper))

    message = f"{base},{shoulder},{elbow},{gripper}\n"

    if ser is not None:
        ser.write(message.encode("utf-8"))
        ser.flush()
    else:
        print(f"  [DRY-RUN] → {message.strip()}")


# ── Episode loop ──────────────────────────────────────────────

def run_episodes(
    model:       PPO,
    env:         Monitor,
    ser,
    flip_elbow:  bool,
    n_episodes:  int,
) -> None:
    """
    Run the agent for n_episodes, sending angles to the ESP32 each step.

    Args:
        model:       trained PPO model
        env:         monitored RobotArmEnv
        ser:         open Serial connection (or None for dry-run)
        flip_elbow:  compensate reversed elbow servo
        n_episodes:  number of episodes to run
    """
    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()

        # Move to home position before episode starts
        send_angles(ser, env.env._angles, flip_elbow)
        time.sleep(1.0)

        print(f"\n── Episode {episode}/{n_episodes} ───────────────────────")
        print(f"   Target: {env.env._target.round(1)} cm")

        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            angles = env.env._angles
            send_angles(ser, angles, flip_elbow)

            print(
                f"   Step {step + 1:3d} | "
                f"Angles: [{angles[0]:.0f}°, {angles[1]:.0f}°, "
                f"{angles[2]:.0f}°, {angles[3]:.0f}°] | "
                f"Dist: {info['distance_cm']:.2f} cm"
            )

            time.sleep(STEP_DELAY)
            step += 1

        result = "SUCCESS ✓" if info["success"] else "FAILED ✗"
        print(f"   → Episode {episode} {result} after {step} steps.")
        time.sleep(1.5)


# ── Main ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy PPO agent to ESP32 via Serial.")
    parser.add_argument(
        "--port", type=str, default=None,
        help="Serial port (e.g. COM3 on Windows, /dev/ttyUSB0 on Linux)"
    )
    parser.add_argument(
        "--flip-elbow", action="store_true",
        help="Reverse elbow angle (use if physical servo direction is opposite)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without hardware — prints angles instead of sending via Serial"
    )
    parser.add_argument(
        "--episodes", type=int, default=NUM_EPISODES,
        help=f"Number of episodes to run (default: {NUM_EPISODES})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dry_run and args.port is None:
        print("ERROR: Provide --port (e.g. --port COM3) or use --dry-run.")
        sys.exit(1)

    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    env   = Monitor(RobotArmEnv())

    ser = None
    if not args.dry_run:
        ser = open_serial(args.port, BAUD_RATE)
    else:
        print("DRY-RUN mode — no Serial connection.")

    if args.flip_elbow:
        print("Elbow flip enabled (sending 180 - elbow to compensate reversed servo).")

    try:
        run_episodes(model, env, ser, args.flip_elbow, args.episodes)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # Move to home before disconnecting
        home = np.array([90.0, 90.0, 90.0, 90.0])
        print("Moving to home position...")
        send_angles(ser, home, flip_elbow=False)
        time.sleep(1.0)

        if ser is not None:
            ser.close()
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()