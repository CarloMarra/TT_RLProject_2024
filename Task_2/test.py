"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent_64x2 import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="/home/ale/TT_RLProject_2024/Task_2/REINFORCE_70_64x2.mdl", type=str, help='Model path')
    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():
    """
    Main function to test the RL agent.
    """
    # env = gym.make('CustomHopper-source-v0')
    env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]
    
    policy = Policy(observation_space_dim, action_space_dim)
    
    # Load the trained model
    policy.load_state_dict(torch.load(args.model), strict=True)
    
    # Initialize the agent
    agent = Agent(policy, device=args.device)

    rewards = []

    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            # Get action from the agent
            action, _ = agent.get_action(state, evaluation=True)
            
            # Step the environment
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward

        rewards.append(test_reward)
        print(f"Episode: {episode + 1} | Return: {test_reward}")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"Mean Reward: {mean_reward}")
    print(f"Standard Deviation of Reward: {std_reward}")

if __name__ == '__main__':
    main()
