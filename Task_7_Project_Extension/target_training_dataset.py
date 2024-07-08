import gym
import time

from stable_baselines3 import PPO

from env.custom_hopper import CustomHopper
import numpy as np

# Create the environment
env = gym.make('CustomHopper-target-v0')

# Define the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model for 250,000 steps
model.learn(total_timesteps=250_000, progress_bar=True)

# Save the trained model
model.save("Task_7_Project_Extension/PPO_data_generator")

# Load the trained model
model = PPO.load("Task_7_Project_Extension/PPO_data_generator")

# Initialize lists to store the actions, states, next_states, and done booleans
actions = []
states = []
next_states = []
dones = []

# Reset the environment to get the initial state
obs = env.reset()

# Run the environment for a specified number of steps
for _ in range(5000):  
    action, _ = model.predict(obs, deterministic=True)
    
    # Store the current state
    states.append(obs)
    
    # Store the action
    actions.append(action)
    
    # Take the action in the environment
    next_obs, reward, done, info = env.step(action)
    
    # Store the next state
    next_states.append(next_obs)
    
    # Store the done boolean
    dones.append(done)
    
    # Update the current state
    obs = next_obs
    
    if done:
        obs = env.reset()

# Convert lists to numpy arrays for easier saving
actions = np.array(actions)
states = np.array(states)
next_states = np.array(next_states)
dones = np.array(dones)

# Save the arrays to files
np.save('Task_7_Project_Extension/Dataset/test_actions.npy', actions)
np.save('Task_7_Project_Extension/Dataset/test_states.npy', states)
np.save('Task_7_Project_Extension/Dataset/test_next_states.npy', next_states)
np.save('Task_7_Project_Extension/Dataset/test_dones.npy', dones)

# Optional: Evaluation and Visualization
import time

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()