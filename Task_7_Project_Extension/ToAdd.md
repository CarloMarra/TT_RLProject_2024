Augment the simulated environment with the following methods to allow Domain Randomization and its optimization:
- `env.set_task(*new_task)` # Set new dynamics parameters **= set_parameters(self, task)**
- `env.get_task()` # Get current dynamics parameters **= get_parameters(self)**
- `mjstate = env.get_sim_state()` # Get current internal mujoco state **OK**
- `env.get_initial_mjstate(state)` and `env.get_full_mjstate(state)` # Get the internal mujoco state from given state **OK**
- `env.set_sim_state(mjstate)` # Set the simulator to a specific mujoco state **OK**

- `env.set_task_search_bounds()` # Set the search bound for the mean of the dynamics parameters
- _(optional)_ `env.get_task_lower_bound(i)` # Get lower bound for i-th dynamics parameter
- _(optional)_ `env.get_task_upper_bound(i)` # Get upper bound for i-th dynamics parameter