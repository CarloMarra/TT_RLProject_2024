import numpy as np
from env.custom_hopper import CustomHopper
import gym
from dropo_implem import Dropo

def main():
    sim_env = gym.make('CustomHopper-v0')

    print('State space:', sim_env.observation_space)
    print('Action space:', sim_env.action_space)
    print('Initial dynamics:', sim_env.get_parameters())

    observations = np.load('Task_7_Project_Extension/Dataset/test_states.npy')
    next_observations = np.load('Task_7_Project_Extension/Dataset/test_next_states.npy')
    actions = np.load('Task_7_Project_Extension/Dataset/test_actions.npy')
    terminals = np.load('Task_7_Project_Extension/Dataset/test_dones.npy')

    T = {'observations': observations, 'next_observations': next_observations, 'actions': actions, 'terminals': terminals}

    dropo = Dropo(sim_env=sim_env, t_length=20, seed=42)

    dropo.set_offline_dataset(T)

    (best_bounds, best_score, elapsed) = dropo.optimize_dynamics_distribution(budget=100, epsilon=1e-3, sample_size=1)

    print('\n-----------')
    print('RESULTS\n')

    print('Best means and st.devs:\n---------------')
    print(dropo.pretty_print_bounds(best_bounds), '\n')

    print('Best score (log likelihood):', best_score)
    print('MSE:', dropo.MSE(dropo.get_means(best_bounds)))
    print('Elapsed:', round(elapsed / 60, 4), 'min')

if __name__ == '__main__':
    main()
