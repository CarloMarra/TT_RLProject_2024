import gym
import time
import csv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import ProgressBarCallback, StopTrainingOnRewardThreshold, EvalCallback

from env.custom_hopper import CustomHopper


def main():
    start = time.time()
    # Create the environment
    env = make_vec_env('CustomHopper-target-v0', n_envs=1)
    # target_env = make_vec_env('CustomHopper-target-v0', n_envs=1)

    
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1800, verbose=1)
    # eval_callback = EvalCallback(target_env, callback_on_new_best=callback_on_best, verbose=1)
        
    # Load the trained model
    #model_path = '/home/ale/TT_RLProject_2024/Task_5/SavedModels_source/PPO_(0_001-4096-256-0_0001-0_99)_0hmxm2fs.zip'
    #model = PPO.load(model_path, env=env)
    model = PPO("MlpPolicy", batch_size = 256, learning_rate=0.001, gamma=0.99, ent_coef=0.0001, n_steps=4096, env=env, verbose=1)

    # Continue Training for another 250_000 timesteps
    model.learn(total_timesteps=int(1_200_000), progress_bar=True)
    
    end = time.time()
    
    elapsed_time = end - start

    print(f"Elapsed time: {elapsed_time} seconds")
    
    model.save("/home/ale/TT_RLProject_2024/Task_5/FinalModels/time_comparison_final_target")

if __name__ == '__main__':
    main()