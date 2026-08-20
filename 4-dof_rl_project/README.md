# 4-DOF Robot Arm — Reinforcement Learning Project

A wooden 4-DOF robot arm (Spacnana SNAM1500) controlled via an ESP32-S3 and trained with Reinforcement Learning (PPO) to perform a reach task.

## Project Structure

```
├── firmware/       ESP32-S3 firmware (PlatformIO / C++)
├── rl/             Simulation environment and RL training (Python)
└── README.md
```

## Hardware

| Component | Details |
|-----------|---------|
| Robot Arm | Spacnana SNAM1500 (4-DOF, wooden) |
| Controller | ESP32-S3-DevKitC-1 |
| Servos | 4x SG90 (GPIO 4, 5, 6, 7) |
| IMU (planned) | MPU-6050 (SDA→GPIO8, SCL→GPIO9) |

## Joints

| Joint | Function | Min | Max |
|-------|----------|-----|-----|
| 0 — Base | Rotates left/right | 0° | 180° |
| 1 — Shoulder | Lifts/lowers upper arm | 30° | 150° |
| 2 — Elbow | Bends forearm | 30° | 150° |
| 3 — Gripper | Opens/closes | 60° | 120° |

## Approach

1. Train the agent in simulation (Gymnasium + Stable-Baselines3)
2. Export learned joint angles
3. Deploy to ESP32 via Serial (Phase 3)

## Quick Start

See [`rl/README.md`](rl/README.md) for training instructions and [`firmware/README.md`](firmware/README.md) for flashing the ESP32.