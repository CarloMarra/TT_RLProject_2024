import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from env.custom_hopper import CustomHopper

# Define the sweep configuration
sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'rollout/ep_rew_mean',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {'values': [5e-3, 1e-3, 5e-4]},
        'n_steps': {'values': [2048, 4096, 8192]},
        'batch_size': {'values': [64, 128, 256]},
        'ent_coef': {'values': [0.01, 0.001, 0.0001]},
        'gamma': {'values': [0.99, 0.999]}
        }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='TT_RLProject')

def train():
    # Initialize a new wandb run
    
    config = {
        'policy_type':'MlpPolicy',
        'total_timesteps': 250_000,
        'env_name': 'CustomHopper-source-v0'
        }
    run = wandb.init(project='TT_RLProject', config=config, sync_tensorboard=True)
    
    sweep_config = run.config

    def make_env():
        env = gym.make(config['env_name'])
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])
    
    model = PPO(
        config['policy_type'], 
        env, 
        verbose=1, 
        learning_rate=sweep_config.learning_rate,
        n_steps=sweep_config.n_steps,
        batch_size=sweep_config.batch_size,
        ent_coef=sweep_config.ent_coef,
        gamma=sweep_config.gamma,
        
        tensorboard_log=f"runs/{run.id}"
    )
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    
    model_name = (f"PPO_({sweep_config.learning_rate}-{sweep_config.n_steps}-{sweep_config.batch_size}-{sweep_config.ent_coef}-{sweep_config.gamma})_{run.id}")
    model_name = model_name.replace('.', '_')
    
    PPO_path = f'Task_5/SavedModels_source/{model_name}'
    model.save(PPO_path)
    
    # evaluate_policy(model=model, env=env n_eval_episodes=10, render=False)
    run.finish()

# Start the sweep
wandb.agent(sweep_id, train, count=10)
