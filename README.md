# HandReach-v3 — SAC + HER

Reinforcement Learning agent for the Shadow Hand reach task using
Soft Actor-Critic (SAC) with Hindsight Experience Replay (HER).

**Task:** Move all 5 fingertips of a Shadow Hand (24 DoF) simultaneously
to their target positions.

## Background

HandReach uses sparse rewards — the agent receives `-1` every step
and `0` on success. Without HER, the agent almost never finds success
by chance and learns nothing. HER solves this by relabeling failed
episodes: if the agent reached position X instead of the goal, the
episode is stored again with X as the goal — turning failures into
useful learning signal.

## Environment

| Property | Value |
|----------|-------|
| Environment | `HandReach-v3` (gymnasium-robotics) |
| Observation | 63 values (joint angles, velocities) |
| Goal | 15 values (5 fingertip target positions × 3) |
| Action space | 20 continuous joint commands |
| Reward | Sparse: -1 per step, 0 on success |
| Max steps | 50 per episode |

## Algorithm

| Component | Choice | Reason |
|-----------|--------|--------|
| RL Algorithm | SAC | Off-policy, sample efficient, built-in entropy exploration |
| Replay Buffer | HER | Handles sparse rewards via goal relabeling |
| Policy | MultiInputPolicy | Handles dict observations (state + goal) |
| Goal strategy | `future` | Best strategy from the HER paper |
| Relabeled goals | 4 per transition | Optimal ratio from the HER paper |

## Setup

```bash
conda create -n mujoco_env python=3.11
conda activate mujoco_env
pip install mujoco gymnasium[robotics] gymnasium-robotics stable-baselines3 tensorboard
```

## Training

```bash
python train.py
```

Monitor with TensorBoard:

```bash
tensorboard --logdir logs/
```

Key metric to watch: `eval/success_rate` — starts at 0, should rise after ~200k timesteps.

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `total_timesteps` | 1,000,000 | Total training steps |
| `buffer_size` | 1,000,000 | Replay buffer capacity |
| `batch_size` | 256 | Minibatch size |
| `learning_rate` | 1e-3 | Adam optimizer LR |
| `gamma` | 0.95 | Discount factor |
| `n_sampled_goal` | 4 | HER relabeled goals per transition |

## Reference

- [Hindsight Experience Replay — Andrychowicz et al. 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf)
- [Multi-Goal RL Environments — arXiv 1802.09464](https://arxiv.org/abs/1802.09464)