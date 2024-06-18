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
    #vec_env = make_vec_env('CustomHopper-source-v0', n_envs=4)
    env = gym.make('CustomHopper-source-v0')
    env.reset()

    print('BODY NAMES:')
    print(env.sim.model.body_names)
    
    for i, part in enumerate(list(env.sim.model.body_names)):
        print(f'{part} mass = {env.sim.model.body_mass[i]}')
        
    env.reset()
    
    for i, part in enumerate(list(env.sim.model.body_names)):
        print(f'{part} mass = {env.sim.model.body_mass[i]}')


if __name__ == '__main__':
    main()