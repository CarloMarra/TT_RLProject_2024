import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.custom_hopper import CustomHopper

def make_env():
    return gym.make('CustomHopper-target-v0')

def main():
   
    vec_env = make_vec_env('CustomHopper-source-v0', n_envs=1)
   

    model = PPO.load("/home/ale/TT_RLProject_2024/Task_4/ppo_Hopper_v0_4env")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

if __name__ == '__main__':
    main()
