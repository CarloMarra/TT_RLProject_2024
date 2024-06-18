### Overview of Wandb Methods

1. **Installation and Initialization**:
   - **Installation**: Install Wandb using pip.
     ```bash
     pip install wandb
     ```
   - **Login and Project Initialization**: Login to Wandb and initialize a project.
     ```python
     import wandb
     wandb.login()
     wandb.init(project="my_project")
     ```

2. **Defining and Running Sweeps**:
   - **Sweep Configuration**: Define the parameters to search over, the search strategy, and the objective metric.
     ```python
     sweep_config = {
         'method': 'bayes',  # Search strategy: 'grid', 'random', 'bayes'
         # 'grid': Exhaustive search over specified parameter values.
         # 'random': Randomly sample parameters from specified values.
         # 'bayes': Bayesian optimization to intelligently sample parameters likely to perform well.
         'metric': {
             'name': 'mean_reward',  # Metric to optimize, can be any performance metric relevant to your task.
             'goal': 'maximize'  # Goal: 'maximize' to increase the metric value, 'minimize' to decrease it.
         },
         'parameters': {
             'learning_rate': {'values': [0.0001, 0.0003, 0.001, 0.01]},  # Hyperparameter values to explore.
             'n_envs': {'values': [1, 4, 8, 16]},  # Number of parallel environments.
             'gamma': {'values': [0.95, 0.99, 0.999]},  # Discount factor for future rewards.
             'clip_range': {'values': [0.1, 0.2, 0.3]}  # Clipping range for PPO.
         }
     }
     ```
   - **Initialize the Sweep**: Create a new sweep in your project.
     ```python
     import wandb

     sweep_id = wandb.sweep(sweep_config, project="rl_project")
     ```

3. **Training Function**:
   - **Initialize Wandb Run**: Start a new Wandb run within the training function.
     ```python
     wandb.init()
     ```
   - **Get Hyperparameters**: Retrieve hyperparameters from Wandb configuration.
     ```python
     config = wandb.config
     ```
   - **Log Metrics**: Log performance metrics to Wandb.
     ```python
     wandb.log({"episode": episode, "reward": total_reward})
     ```
   - **Save Artifacts (like models)**: Save the model and log it to Wandb.
     ```python
     model.save("model_path")
     wandb.save("model_path")
     ```

4. **Running the Sweep**: Execute the sweep agent, which runs the training function with different sets of hyperparameters.
   ```python
   wandb.agent(sweep_id, function=train, count=10)  # Run the sweep with the training function
   ```
   
### Loggin In

U: 100%	 C: 95%

Yes, you do need to log in to Wandb to perform these operations. Here's a detailed guide on how to log in, retrieve information, and visualize plots.

#### Logging into Wandb

1. **Login to Wandb**:
   Before running your script, log in to Wandb from the command line:
   ```bash
   wandb login
   ```
   This will prompt you to enter an API key, which you can find in your Wandb account settings.

2. **Login Programmatically** (Optional):
   You can also log in programmatically in your script:
   ```python
   import wandb
   wandb.login(key="your_api_key_here")
   ```

#### Running the Script

After logging in, run your script. Wandb will automatically log the specified metrics, hyperparameters, and artifacts (like models) during the training process.

#### Retrieving Information and Visualizing Plots

1. **Accessing the Project Dashboard**:
   - Go to the Wandb website (https://wandb.ai/) and log in to your account.
   - Navigate to your project. You should see a dashboard with all your runs.

2. **Viewing Runs**:
   - Each run will be listed with its hyperparameters, metrics, and other logged information.
   - Click on a run to see detailed logs, charts, and any saved artifacts.

3. **Visualizing Metrics**:
   - Wandb automatically creates plots for the metrics you log.
   - For example, if you log `mean_reward`, Wandb will plot it over the course of training.
   - You can customize these plots in the dashboard by selecting different metrics, runs, or adjusting the plot settings.

4. **Downloading Data**:
   - You can download the logs and artifacts from the dashboard.
   - For example, to download the metrics as a CSV file, click on the “Download” button on the metrics panel.

5. **Creating Custom Reports**:
   - Use the “Reports” feature to create custom visualizations and summaries of your experiments.
   - Reports can be shared with your team or kept private for your own records.

#### Example: Viewing Metrics and Plots

1. **Navigate to your project** on the Wandb dashboard.
2. **Select a run** to view detailed metrics and logs.
3. **Metrics Tab**: View plots of metrics like `mean_reward` over training steps.
4. **Artifacts Tab**: Download saved models and other files.

#### Summary of Steps

1. **Log in to Wandb**:
   ```bash
   wandb login
   ```
2. **Run your training script**.
3. **Navigate to your Wandb project dashboard** to view and analyze your runs.

### Implementation in Your Script

1. **Install Wandb**:
   Ensure Wandb is installed:
   ```bash
   pip install wandb
   ```

2. **Modify the Script for Wandb Integration**:
   - **Define the Sweep Configuration**:
     ```python
     sweep_config = {
         'method': 'bayes',
         'metric': {
             'name': 'mean_reward',
             'goal': 'maximize'
         },
         'parameters': {
             'learning_rate': {'values': [0.0001, 0.0003, 0.001, 0.01]},
             'n_envs': {'values': [1, 4, 8, 16]},
             'gamma': {'values': [0.95, 0.99, 0.999]},
             'clip_range': {'values': [0.1, 0.2, 0.3]}
         }
     }
     ```

   - **Initialize the Sweep**:
     ```python
     import wandb
     sweep_id = wandb.sweep(sweep_config, project="rl_project")
     ```

   - **Define the Training Function**:
     ```python
     import gym
     import wandb
     from stable_baselines3 import PPO
     from stable_baselines3.common.env_util import make_vec_env
     from stable_baselines3.common.logger import configure
     from stable_baselines3.common.callbacks import EvalCallback

     from env.custom_hopper import CustomHopper

     def train():
         wandb.init()
         config = wandb.config
         
         vec_env = make_vec_env('CustomHopper-source-v0', n_envs=config.n_envs)
         tmp_path = "/home/ale/TT_RLProject_2024/Task_4/"
         new_logger = configure(tmp_path, ["csv"])
         
         model = PPO("MlpPolicy", vec_env, learning_rate=config.learning_rate, gamma=config.gamma, clip_range=config.clip_range, verbose=1)
         model.set_logger(new_logger)
         
         eval_env = make_vec_env('CustomHopper-source-v0', n_envs=1)
         eval_callback = EvalCallback(eval_env, best_model_save_path=tmp_path, log_path=tmp_path, eval_freq=500, deterministic=True, render=False)
         
         model.learn(total_timesteps=int(1000000), callback=eval_callback, progress_bar=True)
         model_path = tmp_path + "ppo_Hopper_v0_" + str(config.learning_rate)
         model.save(model_path)
         wandb.log({"model_path": model_path, "mean_reward": eval_callback.last_mean_reward})
     ```

   - **Run the Sweep**:
     ```python
     wandb.agent(sweep_id, function=train, count=10)
     ```

### Complete Code Integration

Here is the complete integrated code for using Wandb with your reinforcement learning script:

```python
import gym
import time
import csv
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback

from env.custom_hopper import CustomHopper

def train():
    wandb.init()
    config = wandb.config

    vec_env = make_vec_env('CustomHopper-source-v0', n_envs=config.n_envs)
    tmp_path = "/home/ale/TT_RLProject_2024/Task_4/"
    new_logger = configure(tmp_path, ["csv"])

    model = PPO("MlpPolicy", vec_env, learning_rate=config.learning_rate, gamma=config.gamma, clip_range=config.clip_range, verbose=1)
    model.set_logger(new_logger)

    eval_env = make_vec_env('CustomHopper-source-v0', n_envs=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=tmp_path, log_path=tmp_path, eval_freq=500, deterministic=True, render=False)

    model.learn(total_timesteps=int(1000000), callback=eval_callback, progress_bar=True)
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
            'n_envs': {'values': [1, 4, 8, 16]},
            'gamma': {'values': [0.95, 0.99, 0.999]},
            'clip_range': {'values': [0.1, 0.2, 0.3]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="rl_project")
    wandb.agent(sweep_id, function=train, count=10)

if __name__:
    main()
```

#### Explanation

1. **Sweep Configuration**:
   - **method**: Specifies the search strategy (e.g., 'bayes' for Bayesian optimization).
   - **metric**: Defines the performance metric to optimize ('mean_reward') and the goal ('maximize').
   - **parameters**: Lists the hyperparameters to explore and their possible values:
     - `learning_rate`: Various learning rates for PPO.
     - `n_envs`: Number of parallel environments.
     - `gamma`: Discount factor for future rewards.
     - `clip_range`: Clipping range for PPO.

2. **Training Function**:
   - **wandb.init()**: Initializes a Wandb run.
   - **config**: Retrieves hyperparameters from the Wandb configuration.
   - **model**: Sets up the PPO model with the specified hyperparameters.
   - **eval_callback**: Evaluates the model periodically and logs the mean reward.
   - **wandb.log()**: Logs the model path and mean reward to Wandb.

3. **Running the Sweep**:
   - **wandb.sweep()**: Initializes the sweep with the defined configuration.
   - **wandb.agent()**: Runs the training function multiple times with different sets of hyperparameters, as determined by the sweep configuration.

This setup allows you to

 efficiently perform hyperparameter tuning for your reinforcement learning model using Wandb, with automated logging and tracking of each experiment.
