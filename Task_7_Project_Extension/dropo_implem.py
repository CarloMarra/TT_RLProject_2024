import sys
import time
from copy import deepcopy
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal, truncnorm
import nevergrad as ng

class Dropo(object):
    def __init__(self, sim_env, t_length, seed=0):
        self.sim_env = sim_env
        self.sim_env.reset()
        self._raw_mjstate = deepcopy(self.sim_env.get_sim_state())  # Save fresh full mjstate
        self.t_length = t_length
        self.current_t_length = -self.t_length
        self.scaler = StandardScaler(copy=True)
        self.T = None
        self.seed = seed

        print("Initialized Dropo object")

    def set_offline_dataset(self, T):
        self.T = T
        self.transitions = list(range(len(self.T['observations'])))

        self.scaler.fit(self.T['next_observations'])

        print("Offline dataset set with", len(self.T['observations']), "observations")

    def get_means(self, phi):
        return np.array(phi)[::2]

    def get_stdevs(self, phi):
        return np.array(phi)[1::2]

    def pretty_print_bounds(self, phi):
        assert (
            self.sim_env is not None
            and isinstance(self.sim_env.dynamics_indexes, dict)
        )
        return '\n'.join([str(self.sim_env.dynamics_indexes[i])+':\t'+str(round(phi[i*2],5))+', '+str(round(phi[i*2+1],5)) for i in range(len(phi)//2)])

    def optimize_dynamics_distribution(self, budget=1000, epsilon=1e-3, sample_size=10):
        print("Starting optimization with budget:", budget, "epsilon:", epsilon, "sample_size:", sample_size)
        
        dim_task = len(self.sim_env.get_parameters())

        search_space = []

        self.parameter_bounds = np.empty((dim_task, 2, 2), float)
        self.normalized_width = 4

        self.sim_env.set_task_search_bounds()

        for i in range(dim_task):
            width = self.sim_env.max_task[i] - self.sim_env.min_task[i]  # Search interval for this parameter

            # MEAN
            # Normalize parameter mean to interval [0, 4]
            search_space.append(ng.p.Scalar(init=self.normalized_width * 0.5).set_bounds(lower=0, upper=self.normalized_width))

            self.parameter_bounds[i, 0, 0] = self.sim_env.min_task[i]
            self.parameter_bounds[i, 0, 1] = self.sim_env.max_task[i]

            # STANDARD DEVIATION
            initial_std = width / 8  # This may sometimes lead to a stdev smaller than the lower threshold of 0.00001, so take the minimum
            stdev_lower_bound = np.min([0.00001, initial_std - 1e-5])
            stdev_upper_bound = width / 4

            # Normalize parameter stdev to interval [0, 4]
            # Optimize stdevs in log-space
            search_space.append(ng.p.Scalar(init=self.normalized_width / 2).set_bounds(lower=0, upper=self.normalized_width))

            self.parameter_bounds[i, 1, 0] = stdev_lower_bound
            self.parameter_bounds[i, 1, 1] = stdev_upper_bound

        print("Search space initialized with dimensions:", dim_task)

        params = ng.p.Tuple(*search_space)

        instru = ng.p.Instrumentation(bounds=params, sample_size=sample_size, epsilon=epsilon)

        Optimizer = ng.optimizers.CMA
        optim = Optimizer(parametrization=instru, budget=budget)
        
        progressBar = ng.callbacks.ProgressBar()
        optim.register_callback("tell", progressBar)

        start = time.time()

        loss_function = self._L_target_given_phi
        recommendation = optim.minimize(loss_function, verbosity=0)
        
        end = time.time()
        elapsed = end - start

        print("Optimization completed in", round(elapsed, 2), "seconds")
        print("Recommendation value:", recommendation.value)
        print("Bounds:", recommendation.value[1]['bounds'])

        return self._denormalize_bounds(recommendation.value[1]['bounds']), loss_function(**recommendation.kwargs), elapsed

    def _L_target_given_phi(self, bounds, sample_size=100, epsilon=1e-3):
        bounds = self._denormalize_bounds(bounds)

        sample = self.sample_truncnormal(bounds, sample_size)

        r = self.sim_env.reset()
        mapped_sample_per_transition = np.zeros((len(self.transitions), sample_size, r.shape[0]), float)
        target_ob_prime_per_transition = np.zeros((len(self.transitions), r.shape[0]), float)

        lambda_steps = self.t_length

        effective_transitions = []
        first_pass = True

        for i, ss in enumerate(range(sample_size)):
            task = sample[ss]
            self.sim_env.set_parameters(*task)
            reset_next = True
            lambda_count = -1

            for k, t in enumerate(self.transitions[:-self.t_length]):
                lambda_count += 1
                if lambda_count < 0 or lambda_count % lambda_steps != 0:
                    continue

                for l in range(k, k + lambda_steps):
                    
                    if l >= len(self.transitions):
                        print(f"Skipping out-of-bounds transition: l={l}, transitions length={len(self.transitions)}")
                        break
                    
                    if self.T['terminals'][self.transitions[l]] == True:
                        reset_next = True
                        lambda_count = -1
                        break
                if lambda_count == -1:
                    continue
                if first_pass:
                    effective_transitions.append(k)

                ob = self.T['observations'][t]
                target_ob_prime = self.T['next_observations'][t + lambda_steps - 1]

                if reset_next:
                    r = self.sim_env.reset()
                    self.sim_env.set_sim_state(self.sim_env.get_mjstate(ob, self._raw_mjstate))
                    self.sim_env.sim.forward()
                    reset_next = False
                else:
                    self.sim_env.set_sim_state(self.sim_env.get_mjstate(ob, self.sim_env.get_sim_state()))
                    self.sim_env.sim.forward()

                for j in range(t, t + lambda_steps):
                    action = self.T['actions'][j]
                    s_prime, reward, done, _ = self.sim_env.step(action)

                mapped_sample = np.array(s_prime)
                target_ob_prime = self.scaler.transform(target_ob_prime.reshape(1, -1))[0]
                mapped_sample = self.scaler.transform(mapped_sample.reshape(1, -1))[0]

                mapped_sample_per_transition[k, i, :] = mapped_sample
                target_ob_prime_per_transition[k, :] = target_ob_prime

            first_pass = False

        likelihood = 0

        for i, k in enumerate(effective_transitions):
            mapped_sample = mapped_sample_per_transition[k]
            target_ob_prime = target_ob_prime_per_transition[k]

            cov_matrix = np.cov(mapped_sample, rowvar=0)
            mean = np.mean(mapped_sample, axis=0)
            cov_matrix = cov_matrix + np.diag(np.repeat(epsilon, mean.shape[0]))
            multi_normal = multivariate_normal(mean=mean, cov=cov_matrix, allow_singular=True)
            logdensity = multi_normal.logpdf(target_ob_prime)
            likelihood += logdensity

        if np.isinf(likelihood):
            print('WARNING: infinite likelihood encountered.')

        return -1 * likelihood

    def _denormalize_bounds(self, phi):
        new_phi = []

        for i in range(len(phi)//2):
            norm_mean = phi[i*2]
            norm_std = phi[i*2 + 1]

            mean = (norm_mean * (self.parameter_bounds[i, 0, 1] - self.parameter_bounds[i, 0, 0])) / self.normalized_width + self.parameter_bounds[i, 0, 0]
            std = self.parameter_bounds[i, 1, 0] * ((self.parameter_bounds[i, 1, 1] / self.parameter_bounds[i, 1, 0]) ** (norm_std / self.normalized_width))

            new_phi.append(mean)
            new_phi.append(std)

        return new_phi

    def MSE(self, means):
        distance = []
        task = np.array(means)
        self.sim_env.set_parameters(*task)
        reset_next = True

        for k, t in enumerate(self.transitions[:-self.t_length]):
            if self.T['terminals'][t] == True:
                reset_next = True
                continue

            target_s = self.T['observations'][t]
            target_s_prime = self.T['observations'][t + 1]

            if reset_next:
                r = self.sim_env.reset()
                self.sim_env.set_sim_state(self.sim_env.get_mjstate(target_s, self._raw_mjstate))

                self.sim_env.sim.forward()
                reset_next = False
                
            else:
                self.sim_env.set_sim_state(self.sim_env.get_mjstate(target_s, self.sim_env.get_sim_state()))

            action = self.T['actions'][t]
            sim_s_prime, reward, done, _ = self.sim_env.step(action)

            sim_s_prime = np.array(sim_s_prime)
            distance.append(self._distance(sim_s_prime, target_s_prime))

        mse = np.mean(distance)
        print("Calculated MSE:", mse)
        return mse

    def sample_truncnormal(self, phi, size=1):
        a, b = -2, 2
        sample = []

        for i in range(len(phi) // 2):
            mean = phi[i * 2]
            std = phi[i * 2 + 1]

            lower_bound = self.sim_env.get_task_lower_bound(i)
            upper_bound = self.sim_env.get_task_upper_bound(i)

            attempts = 0
            obs = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
            while np.any((obs < lower_bound) | (obs > upper_bound)):
                obs[((obs < lower_bound) | (obs > upper_bound))] = truncnorm.rvs(a, b, loc=mean, scale=std, size=len(obs[((obs < lower_bound) | (obs > upper_bound))]))

                attempts += 1
                if attempts > 20:
                    obs[obs < lower_bound] = lower_bound
                    obs[obs > upper_bound] = upper_bound
                    print(f"Warning - Not all samples were above >= {lower_bound} or below {upper_bound} after 20 attempts. Setting them to their min/max bound values, respectively.")

            sample.append(obs)

        return np.array(sample).T

    def _distance(self, target, sim_state):
        d = np.linalg.norm(self.scaler.transform(target.reshape(1, -1)) - self.scaler.transform(sim_state.reshape(1, -1))) ** 2
        return d
