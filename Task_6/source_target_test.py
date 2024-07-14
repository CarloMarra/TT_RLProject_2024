import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.custom_hopper import CustomHopper

# Load the trained model
model_path = '/home/ale/TT_RLProject_2024/Task_6/Models/time_comparison_rand.zip'
model = PPO.load(model_path)

# Create the environment and wrap it with Monitor
env = gym.make('CustomHopper-target-v0')

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=250, render=False)

print(f"Mean reward: {mean_reward} +/- {std_reward:.3f}")
