import gymnasium as gym
import gymnasium_robotics
import mujoco

env = gym.make("HandReach-v3")
model = env.unwrapped.model
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"[{i:2d}] {name}")
env.close()