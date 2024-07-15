import numpy as np
from env.custom_hopper import CustomHopper
import gym
from Task_7_Project_Extension.dropo_implementation import DomainRandomizationOptimizer

def main():
    # Initialize the simulation environment
    sim_env = gym.make('CustomHopper-v0')

    # Display initial state and action spaces
    print('State space:', sim_env.observation_space)
    print('Action space:', sim_env.action_space)
    print('Initial dynamics:', sim_env.get_parameters())

    # Load offline dataset
    observations = np.load('Task_7_Project_Extension/Dataset/test_states.npy')
    next_observations = np.load('Task_7_Project_Extension/Dataset/test_next_states.npy')
    actions = np.load('Task_7_Project_Extension/Dataset/test_actions.npy')
    terminals = np.load('Task_7_Project_Extension/Dataset/test_dones.npy')

    T = {'observations': observations, 'next_observations': next_observations, 'actions': actions, 'terminals': terminals}

    # Initialize the DROPO optimizer
    dropo = DomainRandomizationOptimizer(sim_env=sim_env, t_length=20, seed=42)

    # Set the offline dataset for the optimizer
    dropo.set_offline_dataset(T)

    # Optimize the dynamics distribution
    best_bounds, best_score, elapsed = dropo.optimize_dynamics_distribution(budget=100, epsilon=1e-3, sample_size=1)

    # Print the results
    print('\n-----------')
    print('RESULTS\n')
    print('Best parameter means and standard deviations:')
    print(dropo.print_bounds(best_bounds), '\n')
    print('Mean Squared Error (MSE):', dropo.calculate_mse(dropo.get_means(best_bounds)))
    print('Time elapsed:', round(elapsed, 2), 'seconds')

if __name__ == '__main__':
    main()