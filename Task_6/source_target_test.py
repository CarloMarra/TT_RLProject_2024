import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.custom_hopper_args import CustomHopper, register_custom_hopper

# Load the trained model
model_path = '/home/ale/TT_RLProject_2024/Task_6/SavedModels_source/PPO_0_1_0_1_0_1.zip'
model = PPO.load(model_path)

register_custom_hopper()
# Create the environment and wrap it with Monitor
env = gym.make('CustomHopper-target-v0')

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=50, render=False)

print(f"Mean reward: {mean_reward} +/- {std_reward:.3f}")
