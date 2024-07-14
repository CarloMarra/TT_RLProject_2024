import argparse

import gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback, StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from rand_env.custom_hopper_args import CustomHopper, register_custom_hopper

def validate_percentage(value):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError("%s is an invalid percentage. Must be between 0 and 1." % value)
    return fvalue

def main():
    parser = argparse.ArgumentParser(description='Train Custom Hopper with Domain Randomization')
    parser.add_argument('--thigh_range', type=validate_percentage, default=0.1, help='Range for thigh mass variation as a percentage of the original mass')
    parser.add_argument('--leg_range', type=validate_percentage, default=0.1, help='Range for leg mass variation as a percentage of the original mass')
    parser.add_argument('--foot_range', type=validate_percentage, default=0.1, help='Range for foot mass variation as a percentage of the original mass')
    args = parser.parse_args()

    # Register environments with the specified ranges
    register_custom_hopper(args.thigh_range, args.leg_range, args.foot_range)
    
    config = {
        'policy_type':'MlpPolicy',
        'total_timesteps': 500_000,
        'env_name': 'CustomHopper-source-v0',
        'eval_env': 'CustomHopper-target-v0'
        }
    run = wandb.init(project='TT_RLProject', config=config, sync_tensorboard=True)

    # Train on source environment with randomization
    def make_env():
        env = gym.make(config['env_name'])
        env = Monitor(env)  # record stats such as returns
        return env

    # def make_eval_env():
    #     env = gym.make(config['eval_env'])
    #     env = Monitor(env)  # record stats such as returns
    #     return env

    env = DummyVecEnv([make_env])
    # eval_env = DummyVecEnv([make_eval_env])
    
    model = PPO(config['policy_type'], env=env, verbose=1)
    
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=25, verbose=1)
    # eval_callback = EvalCallback(eval_env=eval_env, eval_freq=10_000, callback_after_eval=stop_train_callback, verbose=1)
    wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2, log='all')
    
    # callback = CallbackList([eval_callback, wandb_callback])

    model.learn(total_timesteps=config['total_timesteps'], callback=wandb_callback, progress_bar=True)

    
    model_name = (f'PPO_{args.thigh_range}')
    model_name = model_name.replace('.', '_')
    
    PPO_path = f'Task_6/SavedModels_source/{model_name}'
    model.save(PPO_path)
    
    # evaluate_policy(model=model, env=env n_eval_episodes=10, render=False)
    run.finish()


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
