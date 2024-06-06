"""
Train an RL agent on the OpenAI Gym Hopper environment using REINFORCE and Actor-critic algorithms.
"""
import argparse
import csv
import time
import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=2500, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=500, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cuda', type=str, help='Network device [cpu, cuda]')
    
    return parser.parse_args()

args = parse_args()
# render = True

# Task 2: interleave data collection to policy updates

def main():
    """
    Main function to train the RL agent.
    """
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    # Open CSV file and write header
    with open('/home/ale/TT_RLProject_2024/Task_2/REINFORCE_training_logs.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Episode', 'Train Reward', 'Episode Duration'])

    # Record start time
    start_time = time.time()

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]
    
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    for episode in range(args.n_episodes):
        # Record episode start time
        episode_start_time = time.time()

        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward

        agent.update_policy()

        # Record episode end time and calculate duration
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time

        # Append episode result to CSV file
        with open('/home/ale/TT_RLProject_2024/Task_2/REINFORCE_training_logs.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([episode + 1, train_reward, episode_duration])

        if (episode + 1) % args.print_every == 0:
            print('Training episode:', episode + 1)
            print('Episode return:', train_reward)
            print(f'Episode duration: {episode_duration:.2f} seconds')

    # Record end time
    end_time = time.time()
    total_training_time = end_time - start_time

    print(f'Total training time: {total_training_time:.2f} seconds')

    torch.save(agent.policy.state_dict(), "/home/ale/TT_RLProject_2024/Task_2/REINFORCE.mdl")

if __name__ == '__main__':
    main()