import gym
import time
import csv


import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback

from env.custom_hopper import CustomHopper


def train():
    wandb.init()
    config = wandb.config

    
    vec_env = make_vec_env('CustomHopper-source-v0', n_envs=4)
    

    tmp_path = "/home/ale/TT_RLProject_2024/Task_4/"
    # Set up logger
    new_logger = configure(tmp_path, ["csv"])

    model = PPO("MlpPolicy", learning_rate=config.learning_rate, env=vec_env, 
                gamma=config.gamma, clip_range=config.clip_range, verbose=1)
    # Set new logger
    model.set_logger(new_logger)
    
    eval_env = make_vec_env('CustomHopper-source-v0', n_envs=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=tmp_path, log_path=tmp_path,
                                 eval_freq=500, deterministic=True, render=False)
    
    # Use a very large number of timesteps to ensure the callback determines stopping
    model.learn(total_timesteps=int(1000000), callback=eval_callback, progress_bar=True)
    model_path = tmp_path + "ppo_Hopper_v0_" + str(config.learning_rate)
    model.save(model_path)
    wandb.log({"model_path": model_path, "mean_reward": eval_callback.last_mean_reward})





def main():
    sweep_config = {
       'method': 'bayes',
       'metric': {
            'name': 'mean_reward',
            'goal': 'maximize'
       },
       'parameters': {
            'learning_rate': {'values': [0.0001, 0.0003, 0.001, 0.01]},
            'n_envs': {'values': [1, 4, 8, 16]},
            'gamma': {'values': [0.95, 0.99, 0.999]},
            'clip_range': {'values': [0.1, 0.2, 0.3]}
       }
   }
  
    sweep_id = wandb.sweep(sweep_config, project="cm_datascience")  
    wandb.agent(sweep_id, function=train, count=10)
    
if __name__ == '__main__':
    main()