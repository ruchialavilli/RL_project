import gymnasium as gym
import pdb
env = gym.make("CarRacing-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    action = [0.4, 0.4, 0.4, 0.4]

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()