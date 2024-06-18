import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import CustomHopper

# Load the trained model
model_path = '/home/ale/TT_RLProject_2024/Task_5/FinalModels/source_final.zip'
model = PPO.load(model_path)

# Create the environment
env = gym.make('CustomHopper-source-v0')

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, render=False)

print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
