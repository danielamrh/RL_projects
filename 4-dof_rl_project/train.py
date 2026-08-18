import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from robot_arm_env import RobotArmEnv

LOG_DIR        = "logs/"
CHECKPOINT_DIR = "checkpoints/"

os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("models",       exist_ok=True)

NUM_ENEVS = 8

train_env = make_vec_env(RobotArmEnv, n_envs=NUM_ENEVS)
eval_env = Monitor(RobotArmEnv())

model = PPO(
    policy          ="MlpPolicy",
    env             =train_env,
    learning_rate   =3e-4,
    n_steps         =1024,
    batch_size      =256,
    n_epochs        =10,
    gamma           =0.99,
    gae_lambda      =0.95,
    clip_range      =0.2,
    ent_coef        =0.0,
    verbose         =1,
    tensorboard_log  =LOG_DIR,
    policy_kwargs   =dict(net_arch=[128, 128]),
)

eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="models/best_model",
    log_path=LOG_DIR,   
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=CHECKPOINT_DIR,
    name_prefix="ppo_robot_arm",
    verbose=1,
)

TOTAL_TIMESTEPS = 1_000_000

print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True,
)

model.save("models/ppo_robot_arm_final")
print("Training complete. Model saved to 'models/ppo_robot_arm_final.zip'.")

print("Evaluating the trained model...")

model = PPO.load("models/ppo_robot_arm_final")
successes = 0
rewards = []

for episode in range(20):
    abs, _ = eval_env.reset()
    total_reward = 0
    
    done = False

    while not done:
        action, _ = model.predict(abs, deterministic=True)
        abs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)
    if info[0].get("success", False):
        successes += 1

print(f"Evaluation complete. Success rate: {successes}/20, Average reward: {np.mean(rewards):.2f}")

