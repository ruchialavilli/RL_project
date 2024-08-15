import gymnasium as gym
import pdb
env = gym.make("CarRacing-v2", domain_randomize=True, render_mode="human")

observation, info = env.reset()

for _ in range(2000):
    # pdb.set_trace()
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# https://gymnasium.farama.org/content/basic_usage/
# https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
# https://gymnasium.farama.org/tutorials/training_agents/
# https://gymnasium.farama.org/tutorials/