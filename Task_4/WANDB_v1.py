import gym
import time
import csv
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from env.custom_hopper import CustomHopper

class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=10, eval_freq=1000, verbose=1):
        super(WandbEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.n_calls = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            if self.verbose > 0:
                print(f"Mean reward: {mean_reward} +/- {std_reward} at timestep {self.num_timesteps}")
            # Log to wandb
            wandb.log({"mean_reward": mean_reward, "std_reward": std_reward, "timestep": self.num_timesteps})
        return True

def train():
    wandb.init()
    config = wandb.config

    vec_env = make_vec_env('CustomHopper-source-v0', n_envs=16)
    tmp_path = "/home/ale/TT_RLProject_2024/Task_4/"
    new_logger = configure(tmp_path, ["csv"])

    model = PPO("MlpPolicy", vec_env, learning_rate=config.learning_rate, gamma=config.gamma, clip_range=config.clip_range, verbose=1)
    model.set_logger(new_logger)

    eval_env = make_vec_env('CustomHopper-source-v0', n_envs=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=tmp_path, log_path=tmp_path, eval_freq=500, deterministic=True, render=False)
    wandb_callback = WandbEvalCallback(eval_env)

    model.learn(total_timesteps=int(100000), callback=[eval_callback, wandb_callback], progress_bar=True)
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
            'n_steps': {'values': [512, 1024, 2048, 4096]},
            'batch_size' : {'values':[64, 128]},
            'n_epochs': {'values':[5, 10, 15]},
            'ent_coef': {'values':[0, 0.01, 0.1]},
            'vf_coef':{'values':[0.35, 0.5, 0.65]},
            'gamma': {'values': [0.95, 0.99, 0.999]},
            'clip_range': {'values': [0.1, 0.2, 0.3]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="cm_datascience")
    wandb.agent(sweep_id, function=train, count=10)

if __name__ == "__main__":
    main()