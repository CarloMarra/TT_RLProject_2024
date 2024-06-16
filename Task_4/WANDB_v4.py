import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

# Define the sweep configuration
sweep_configuration = {
    'method': 'random',
    'metric': {
        'name': 'mean_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'policy_type': {
            'values': ['MlpPolicy', 'CnnPolicy']
        },
        'total_timesteps': {
            'values': [10000, 25000, 50000]
        },
        'learning_rate': {
            'max': 0.001,
            'min': 0.0001
        },
        'gamma': {
            'values': [0.8, 0.9, 0.99]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='sb3')

def train():
    # Initialize a new wandb run
    run = wandb.init()
    config = run.config

    def make_env():
        env = gym.make("CartPole-v1")
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=200,
    )
    
    model = PPO(
        config.policy_type, 
        env, 
        verbose=1, 
        learning_rate=config.learning_rate, 
        gamma=config.gamma,
        tensorboard_log=f"runs/{run.id}"
    )
    
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()

# Start the sweep
wandb.agent(sweep_id, train, count=10)
