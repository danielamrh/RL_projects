import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from robot_arm_env import RobotArmEnv

model = PPO.load("4-dof_rl_project/models/best_model/best_model.zip")
eval_env = Monitor(RobotArmEnv(render_mode="human"))

NUM_EPISODES = 20
successes = 0
rewards = []

for episode in range(NUM_EPISODES):
    obs, _ = eval_env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)
    if info.get("success"):
        successes += 1
    print(f"Episode {episode+1:2d} | Reward: {total_reward:7.2f} | Success: {info.get('success')}")

print("\n=== Results ===")
print(f"Mean reward:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"Success rate: {successes}/{NUM_EPISODES}")
print(f"Best reward:  {np.max(rewards):.2f}")
print(f"Worst reward: {np.min(rewards):.2f}")

eval_env.close()