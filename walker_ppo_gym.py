# ppo - no noise
# /opt/anaconda3/bin/python
import sys
sys.path.insert(0, '/opt/anaconda3/envs/latest')
import gymnasium as gym
import numpy as np
import wandb
import logging
from wandb.integration.sb3 import WandbCallback
# import pdb

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# Setting the seed for reproducibility
seed = 42
np.random.seed(seed)

def make_env():
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    env.reset(seed=seed) 
    env.action_space.seed(seed)
    return env

run = wandb.init(
    project="bipedal_walker_ppo_gym",
    config = {"policy": "PPO_MlpPolicy", "learning_rate": 0.000827, "gamma":0.96979, "gae_lambda":0.92680, "ent_coef":.0000199},
    monitor_gym=True,
    sync_tensorboard=True,
    save_code=True

)

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger = lambda x: x % 2000 == 0, video_length=200)


model = PPO("MlpPolicy", env, learning_rate= 0.000827, gamma = 0.96979, gae_lambda= 0.92680, ent_coef= 0.0000199, verbose=1, tensorboard_log=f"runs/ppo_gym")
model.learn(total_timesteps=1000000,
            callback=WandbCallback(gradient_save_freq=200, model_save_path=f"models/{run.id}", verbose=2)
)


model.save("walk_ppo_gym")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = PPO.load("walk_ppo_gym")

obs = vec_env.reset()

reward_total = 0
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
    reward_total += rewards
    wandb.log({"returns" : reward_total, "reward": rewards})
    #print(rewards)
    if done:
        reward_total = 0
        obs = vec_env.reset()

print(reward_total)

# env.close()
