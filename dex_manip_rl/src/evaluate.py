import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import AllegroGraspEnv

XML_PATH  = r"C:\Users\User\OneDrive\Dokumente\RL_projects\dex_manip_rl\mujoco_menagerie\wonik_allegro\allegro_cube.xml"
SAVE_DIR  = "checkpoints"
N_EPISODES = 20

# Laden
dummy = DummyVecEnv([lambda: AllegroGraspEnv(XML_PATH)])
vec_env = VecNormalize.load(f"{SAVE_DIR}/vecnorm_dex.pkl", dummy)
vec_env.training = False
vec_env.norm_reward = False

model = SAC.load(f"{SAVE_DIR}/sac_dex_final", env=vec_env)
print(f"Model loaded.")

# Test
successes = 0
rewards_all = []

for ep in range(N_EPISODES):
    env_e = AllegroGraspEnv(XML_PATH)
    obs, _ = env_e.reset()
    ep_reward = 0.0
    done = False

    while not done:
        obs_n = vec_env.normalize_obs(obs[np.newaxis])
        action, _ = model.predict(obs_n, deterministic=True)
        obs, r, done, _, _ = env_e.step(action[0])
        ep_reward += r

    cube_height = env_e._cube_height()
    lifted = cube_height > env_e._init_cube_pos[2] + 0.05
    if lifted:
        successes += 1

    rewards_all.append(ep_reward)
    print(f"  Ep {ep+1:02d}: reward={ep_reward:.2f}  cube_z={cube_height:.3f}  {'✓ LIFTED' if lifted else '✗'}")
    env_e.close()

print(f"\nSuccess rate: {successes}/{N_EPISODES} ({successes/N_EPISODES*100:.0f}%)")
print(f"Mean reward : {np.mean(rewards_all):.2f} ± {np.std(rewards_all):.2f}")