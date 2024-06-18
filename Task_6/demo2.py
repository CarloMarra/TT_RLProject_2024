import argparse
import time
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.env_checker import check_env
from env.custom_hopper_args import CustomHopper, register_custom_hopper
from stable_baselines3.common.callbacks import BaseCallback

def validate_percentage(value):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError("%s is an invalid percentage. Must be between 0 and 1." % value)
    return fvalue

class PrintMassesCallback(BaseCallback):
    def __init__(self, vecenv, verbose=0):
        super(PrintMassesCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if an episode is done
        if self.locals['dones'][0]:
            # Access the environment
            env = self.training_env.envs[0]
            # Print the masses of the torso, thigh, leg, and foot
            print(f'Masses: {env.get_parameters()}')
            print('-------------------------------')
        return True

def main():
    parser = argparse.ArgumentParser(description='Train Custom Hopper with Domain Randomization')
    parser.add_argument('--thigh_range', type=validate_percentage, default=0.2, help='Range for thigh mass variation as a percentage of the original mass')
    parser.add_argument('--leg_range', type=validate_percentage, default=0.2, help='Range for leg mass variation as a percentage of the original mass')
    parser.add_argument('--foot_range', type=validate_percentage, default=0.2, help='Range for foot mass variation as a percentage of the original mass')
    args = parser.parse_args()

    # Register environments with the specified ranges
    register_custom_hopper(args.thigh_range, args.leg_range, args.foot_range)

    # Train on source environment with randomization
    vec_env = make_vec_env('CustomHopper-source-v0', n_envs=1)
    model = PPO("MlpPolicy", learning_rate=0.0003, env=vec_env, verbose=0)
    callback = PrintMassesCallback(vecenv=vec_env, verbose=1)
    
    model.learn(total_timesteps=int(2_000_000), callback=callback, progress_bar=True)

    # Test on target environment without randomization
    # test_env = make_vec_env('CustomHopper-target-v0', n_envs=1)
    # obs = test_env.reset()
    # for _ in range(1000):  # Number of test steps
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = test_env.step(action)
    #     if dones:
    #         obs = test_env.reset()

if __name__ == '__main__':
    main()
