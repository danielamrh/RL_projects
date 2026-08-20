# RL — 4-DOF Robot Arm Reach Task

Custom Gymnasium environment and PPO training pipeline for the Spacnana SNAM1500 robot arm.

## Requirements

```
pip install gymnasium stable-baselines3 numpy matplotlib pillow
```

## Files

| File | Description |
|------|-------------|
| `robot_arm_env.py` | Custom Gymnasium environment |
| `train.py` | PPO training script |
| `evaluate.py` | Load and evaluate a saved model |
| `visualize.py` | Record and save a 3D animation |

## Environment

**Observation space (15 values):**
- 3 normalized joint angles (base, shoulder, elbow) in [-1, 1]
- 3 gripper tip position in cm
- 3 target position in cm
- 3 simulated accelerometer readings (m/s²)
- 3 simulated gyroscope readings (rad/s)

**Action space (3 values, continuous [-1, 1]):**
- Desired angles for base, shoulder, elbow (normalized)

**Reward:**
- `-distance × 0.1` per step (distance penalty)
- `-0.1` per step (time penalty)
- `-10.0` on collision (ground / self / base)
- `+100.0` on success (distance < 2 cm)

**Robot geometry:**
- L1 = 8 cm (base height)
- L2 = 10 cm (upper arm)
- L3 = 11 cm (forearm + gripper)

## Training

```bash
python train.py
```

Monitor with TensorBoard:

```bash
tensorboard --logdir logs/
```

Saved models:
- `models/best_model/best_model.zip` — best eval checkpoint
- `models/ppo_robot_arm_final.zip` — final model after training

## Evaluation

```bash
python evaluate.py
```

Prints per-episode reward and success rate over 20 episodes.

## Visualization

```bash
python visualize.py
```

Saves a 3D animation of one episode to `robot_arm_simulation.gif`.

## Collision Detection

The environment checks for three collision types:

| Type | Description |
|------|-------------|
| Ground | Any joint goes below z = 0 |
| Self | Forearm (S2) gets within 2 cm of base column (S0) |
| Base | Forearm intersects the base cylinder (r < 3 cm, z ≤ L1) |

Collisions apply a -10 penalty but do not terminate the episode.