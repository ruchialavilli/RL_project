# /opt/miniconda3/envs/project/bin/python

import gymnasium as gym
import pdb
env = gym.make("BipedalWalker-v3", render_mode='human')

observation, info = env.reset()

reward_total = 0
for _ in range(2000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    reward_total += reward
    if terminated or truncated:
        print(reward_total)
        reward_total = 0
        observation, info = env.reset()

env.close()
