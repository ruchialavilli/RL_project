# ppo - no noise - skip 
# /opt/anaconda3/bin/python
import sys
sys.path.insert(0, '/opt/anaconda3/envs/latest')
import gymnasium as gym
import wandb
# import pdb

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    env = make_vec_env("BipedalWalker-v3", n_envs=4)
    # env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env

run = wandb.init(
    project="bipedal_walker_ppo",

    config = {"episodes": 32, "learning_rate": 0.0003},

    monitor_gym=True,

    sync_tensorboard=True

)


vec_env = DummyVecEnv([make_env]) 

model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=f"runs/ppo")
model.learn(total_timesteps=25000, log_interval=10)
wandb.finish()

model.save("walk_ppo")


del model # remove to demonstrate saving and loading

model = PPO.load("walk_ppo")

obs = vec_env.reset()

reward_total = 0
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    reward_total += rewards
    # vec_env.render("human")
    if dones.all():
        reward_total = 0
        obs = vec_env.reset()

#print(reward_total)

# env.close()
