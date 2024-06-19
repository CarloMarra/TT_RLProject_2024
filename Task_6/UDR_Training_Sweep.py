import argparse

import gym
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


from env.custom_hopper_args import CustomHopper, register_custom_hopper

def main():
    # Define sweep configurations
    sweep_configuration = {
        'method': 'grid',
        'metric': {
            'name': 'rollout/ep_rew_mean',
            'goal': 'maximize'
        },
        'parameters': {
            'config_id': {
                'values': [0, 1, 2]
            }
        }
    }

    # Explicitly define the three configurations
    configs = [
        {'thigh_range': 0.1, 'leg_range': 0.1, 'foot_range': 0.1},
        {'thigh_range': 0.25, 'leg_range': 0.25, 'foot_range': 0.25},
        {'thigh_range': 0.5, 'leg_range': 0.5, 'foot_range': 0.5}
    ]

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='TT_RLProject')

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config

            # Select the configuration based on the sweep parameter
            selected_config = configs[config.config_id]

            # Register environments with the specified ranges
            register_custom_hopper(selected_config['thigh_range'], selected_config['leg_range'], selected_config['foot_range'])

            env_config = {
                'policy_type': 'MlpPolicy',
                'total_timesteps': 500_000,
                'env_name': 'CustomHopper-source-v0',
            }
            run = wandb.init(project='TT_RLProject', config=env_config, sync_tensorboard=True)

            # Train on source environment with randomization
            def make_env():
                env = gym.make(env_config['env_name'])
                env = Monitor(env)  # record stats such as returns
                return env

            env = DummyVecEnv([make_env])
            model = PPO(env_config['policy_type'], env=env, verbose=1, tensorboard_log=f"runs/{run.id}")
            wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2)
            model.learn(total_timesteps=env_config['total_timesteps'], callback=wandb_callback, progress_bar=True)

            model_name = (f'PPO_{selected_config["thigh_range"]}_{selected_config["leg_range"]}_{selected_config["foot_range"]}')
            model_name = model_name.replace('.', '_')
            PPO_path = f'Task_6/SavedModels_source/{model_name}'
            model.save(PPO_path)
            run.finish()

    wandb.agent(sweep_id, train, count=3)

if __name__ == '__main__':
    main()
