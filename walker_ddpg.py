# ddpg
# /opt/anaconda3/bin/python
import sys
sys.path.insert(0, '/opt/anaconda3/envs/latest')
import gymnasium as gym
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback


from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def make_env():
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env

run = wandb.init(
    project="bipedal_walker_ddpg",
    config = {"learning_rate": 0.0003}, 
    monitor_gym=True,
    sync_tensorboard=True,
    save_code=True

)

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger = lambda x: x % 2000 == 0, video_length=200)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=f"runs/ddpg")
model.learn(total_timesteps=10000, log_interval=4,
            callback=WandbCallback(gradient_save_freq=200, model_save_path=f"models/{run.id}", verbose=2)
)

model.save("walk_ddpg")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("walk_ddpg")

obs = vec_env.reset()

reward_total = 0
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
    reward_total += rewards
    #print(rewards)
    if done:
        reward_total = 0
        obs = vec_env.reset()

print(reward_total)

# env.close()
