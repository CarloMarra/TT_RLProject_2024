import gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback

from env.custom_hopper import CustomHopper

def train():
    config = {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "clip_range": 0.2,
        "total_timesteps": 100000,
    }
    run = wandb.init(
        project="cm_datascience",
        config=config,
        sync_tensorboard=True,
    )
    
    config = wandb.config

    vec_env = make_vec_env('CustomHopper-source-v0', n_envs=16)
    tmp_path = "/home/ale/TT_RLProject_2024/Task_4/"
    new_logger = configure(tmp_path, ["csv"])

    model = PPO("MlpPolicy", vec_env, learning_rate=config.learning_rate, gamma=config.gamma, clip_range=config.clip_range, verbose=1)
    model.set_logger(new_logger)

    eval_env = make_vec_env('CustomHopper-source-v0', n_envs=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=tmp_path, log_path=tmp_path, eval_freq=500, deterministic=True, render=False)

    model.learn(
        total_timesteps=int(config.total_timesteps),
        callback=[eval_callback, WandbCallback()],
        progress_bar=True
    )
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
