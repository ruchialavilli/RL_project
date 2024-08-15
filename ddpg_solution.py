#ddpg
# /opt/anaconda3/envs/latest/bin/python
import sys
sys.path.insert(0, '/opt/anaconda3/envs/latest')
import gymnasium as gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make("Pendulum-v1", render_mode="rgb_array")

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save("ddpg_pendulum")
vec_env = model.get_env()

# del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_pendulum")
#vec_env = model.get_env()


obs = vec_env.reset()
reward_total = 0
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(rewards)
    #env.render("human")
