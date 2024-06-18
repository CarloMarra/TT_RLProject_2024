import gym
import time
import csv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import ProgressBarCallback

from env.custom_hopper import CustomHopper


def main():
    #n_envs = 4
    vec_env = make_vec_env('CustomHopper-source-v0', n_envs=1)
    #env = make_env()

    #tmp_path = "/home/ale/TT_RLProject_2024/Task_4/"
    #set up logger
    #new_logger = configure(tmp_path, ["csv"])

    model = PPO("MlpPolicy", learning_rate=0.0003, env=vec_env, verbose=1)
    #Set new logger
    #model.set_logger(new_logger)
        
    # Use a very large number of timesteps to ensure the callback determines stopping
    model.learn(total_timesteps=int(500_000), progress_bar=True)
    model.save("/home/ale/TT_RLProject_2024/Task_4/ppo_Hopper_v0_baseline_500k")

if __name__ == '__main__':
    main()