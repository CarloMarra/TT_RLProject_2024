# Pseudo Code for Simplified DROPO Algorithm

from copy import deepcopy
import numpy as np
from sklearn.preprocessing import StandardScaler

class SimpleDropo:
    def __init__(self, sim_env, window_length, scaling=False):
        # Initialize the environment and parameters
        self.sim_env = sim_env
        self.window_length = window_length
        self.scaling = scaling
        self._raw_mjstate = deepcopy(sim_env.get_sim_state())
        if scaling:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

    def set_offline_dataset(self, T):
        # Set the offline dataset T
        self.T = T
        
        # Use all transitions in `T`
        self.transitions = list(range(len(self.T['observations'])-self.t_length))

        # ???
        # Use all available transitions directly
        # self.transitions = list(range(len(self.T['observations'])))
        
        if self.scaling:
            self.scaler.fit(T['next_observations'])

    def optimize_dynamics_distribution(self, budget=1000, sample_size=100):
        # Define search space for dynamics parameters
        search_space = self._define_search_space()
        optimizer = NevergradOptimizer(search_space, budget)
        
        # Run optimization
        best_params, best_loss = optimizer.minimize(self._objective_function, sample_size)
        return best_params, best_loss

    def _define_search_space(self):
        # Define the search space for dynamics parameters
        dim_task = len(self.sim_env.get_task())
        search_space = []
        for i in range(dim_task):
            search_space.append({
                'mean': (self.sim_env.min_task[i], self.sim_env.max_task[i]),
                'std': (1e-5, (self.sim_env.max_task[i] - self.sim_env.min_task[i]) / 4)
            })
        return search_space

    def _objective_function(self, params, sample_size):
        # Compute the likelihood of the observed transitions given the dynamics parameters
        likelihood = 0
        samples = self._sample_truncnormal(params, sample_size)
        for t in range(len(self.T['observations']) - self.window_length):
            ob = self.T['observations'][t]
            target_ob_prime = self.T['next_observations'][t + self.window_length - 1]
            mapped_samples = self._simulate_samples(ob, samples, t)
            likelihood += self._compute_likelihood(mapped_samples, target_ob_prime)
        return -likelihood

    def _sample_truncnormal(self, params, size):
        # Sample from a truncated normal distribution
        samples = []
        for param in params:
            mean, std = param['mean'], param['std']
            samples.append(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=size))
        return np.array(samples).T

    def _simulate_samples(self, ob, samples, t):
        # Simulate samples in the environment
        mapped_samples = []
        for sample in samples:
            self.sim_env.set_task(*sample)
            self.sim_env.set_sim_state(self.sim_env.get_full_mjstate(ob, self._raw_mjstate))
            for _ in range(self.window_length):
                action = self.T['actions'][t]
                s_prime, _, _, _ = self.sim_env.step(action)
            mapped_samples.append(s_prime)
        return np.array(mapped_samples)

    def _compute_likelihood(self, mapped_samples, target_ob_prime):
        # Compute the likelihood of the target observations given the mapped samples
        mean = np.mean(mapped_samples, axis=0)
        cov_matrix = np.cov(mapped_samples, rowvar=False)
        if self.scaling:
            target_ob_prime = self.scaler.transform(target_ob_prime.reshape(1, -1))[0]
            mapped_samples = self.scaler.transform(mapped_samples)
        multi_normal = multivariate_normal(mean=mean, cov=cov_matrix)
        return multi_normal.logpdf(target_ob_prime)

# Usage
# Initialize the environment and DROPO object
sim_env = HopperEnv()  # Replace with actual environment initialization
dropo = SimpleDropo(sim_env, t_length=5, scaling=True)

# Load the offline dataset
T = np.load('dataset_T.npy', allow_pickle=True).item()
dropo.set_offline_dataset(T)

# Optimize the dynamics distribution
best_params, best_loss = dropo.optimize_dynamics_distribution(budget=1000, sample_size=100)

print("Best Parameters:", best_params)
print("Best Loss:", best_loss)
