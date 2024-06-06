"""
Test an RL agent on the OpenAI Gym Hopper environment.
"""
import argparse
import torch
import gym

from env.custom_hopper import *
from Task_3.agent_A2C import Agent, Actor, Critic

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/ale/TT_RLProject_2024/Task_3/Actor_Critic.mdl', type=str, help='Model path')
    parser.add_argument('--device', default='cuda', type=str, help='Network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=500, type=int, help='Number of test episodes')

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

    # Initialize actor and critic networks
    policy = Actor(observation_space_dim, action_space_dim)
    val_func = Critic(observation_space_dim)
    
    # Load the trained model
    policy.load_state_dict(torch.load(args.model), strict=True)
    
    # Initialize the agent
    agent = Agent(policy, val_func, device=args.device)

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

        print(f"Episode: {episode + 1} | Return: {test_reward}")

if __name__ == '__main__':
    main()