import sys
sys.path.insert(0, '/opt/anaconda3/envs/latest')
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.noise import NormalActionNoise

import numpy as np

seed = 42
np.random.seed(seed)

import wandb
from wandb.integration.sb3 import WandbCallback
run = wandb.init(
    project="bipedal_walker_sac",
    config = {
        "policy": "SAC_MlpPolicy",
        "noise_sigma": 0.1,
    },
    monitor_gym=True,
    sync_tensorboard=True,
    save_code=True
)

def make_env():
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    env.reset(seed=seed) 
    env.action_space.seed(seed)
    return env

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger = lambda x: x % 2000 == 0, video_length=200)

# Instantiate the agent with noise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = SAC("MlpPolicy", env, action_noise=action_noise, verbose=0, tensorboard_log=f"runs/sac")

# Train the agent
model.learn(total_timesteps=int(10000), log_interval=4,
            #callback=WandbCallback(gradient_save_freq=200, model_save_path=f"models/{run.id}", verbose=0),
            callback=EvalCallback(env, best_model_save_path=f"models/best/{run.id}", log_path="./evalLogs/", eval_freq=1000)
)

# Save the agent
model.save("SAC_saved")
vec_env = model.get_env()

del model  # delete trained model to demonstrate loading

# Load the trained agent
model = SAC.load("SAC_saved")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

print(f"Mean reward: {mean_reward} +/- {std_reward}")
obs = vec_env.reset()
reward_total = 0
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    reward_total += rewards
    # print(rewards)
    if dones:
        reward_total = 0
        obs = vec_env.reset()

print(reward_total)
env.close()