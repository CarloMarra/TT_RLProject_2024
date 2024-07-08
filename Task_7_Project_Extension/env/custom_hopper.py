"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        self.task_dim = self.original_masses.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dynamics_indexes = {0: 'torsomass', 1: 'thighmass', 2: 'legmass', 3: 'footmass'}
        
        
    def set_task_search_bounds(self):
        
        self.min_task += 0.1
        self.max_task += 8
        
        
    # def set_search_bounds_mean(self):
    #     """Set search bounds for the mean of the parameters optimized,
    #     the stdev bounds are set accordingly in dropo.
    #     """
    #     self.search_bounds_mean = {
    #            'torsomass': (0.5, 10.0),
    #            'thighmass': (0.5, 10.0),
    #            'legmass': (0.5, 10.0),
    #            'footmass': (0.5, 10.0),
    #     }
    #     return
        

    # def get_search_bounds_mean(self, index):
    #     """Get search bounds for the mean of the parameters optimized,
    #     the stdev bounds are set accordingly in dropo.
    #     """
        
    #     return self.search_bounds_mean[self.dyn_ind_to_name[index]]


    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'torsomass': 0.1,
                    'thighmass': 0.1,
                    'legmass': 0.1,
                    'footmass': 0.1
        }

        return lowest_value[self.dynamics_indexes[index]]
    
    def get_task_upper_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'torsomass': 8,
                    'thighmass' :8,
                    'legmass': 8,
                    'footmass': 8
        }

        return lowest_value[self.dynamics_indexes[index]]


    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self, *task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])


    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)
        
    def get_mjstate(self, state, template):
        # Get a new fresh mjstate template
        mjstate = deepcopy(template)

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        return mjstate


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_sim_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)
