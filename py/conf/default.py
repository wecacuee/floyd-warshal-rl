from cog.confutils import (Conf, dict_update_recursive,
                           init_class_stack, NewConfClass)
import numpy as np

class DefaultConf(Conf):
    """
    Follow the principle of delayed execution.

    1. Avoid executing any code untill a property of this conf is accessed.
    2. Do not make any copies of same attributes.
        Use property objects for redirection
    3. Use class stacks for additional properties on the base classes.
    """
    def __init__(self, **kw):
        defaults = self.defaults()
        updated_kw = dict_update_recursive(defaults, kw)
        super().__init__(**updated_kw)
        self.rng = np.random.RandomState(self.seed)

    def _next_seed(self):
        return self.rng.randint(1000)

    def defaults(self):
        from alg.floyd_warshall_grid import (FloydWarshallAlgDiscrete,
                                             FloydWarshallVisualizer)
        from prob.windy_grid_world import (WindyGridWorld, AgentInGridWorld)
        defaults      = dict(
            nepisodes = 2,
            seed      = 0,

            alg_class_stack = [FloydWarshallAlgDiscrete,
                               FloydWarshallVisualizer],

            alg_kwargs_stack            = [
                # FloydWarshallAlgDiscrete
                NewConfClass(
                    "FloydWarshallAlgDiscreteConf",
                    action_space        = lambda s: self.prob.action_space,
                    observation_space   = lambda s: self.prob.observation_space,
                    seed                = lambda s: self._next_seed()
                )(egreedy_epsilon       = 0.1,
                  path_cost_momentum    = 0.9,  # High momemtum changes less frequently
                  action_value_momentum = 0.1, # Low momentum changes more frequently
                  init_value            = 1,
                  top_value_queue_size  = 5,
                  per_edge_reward       = -1),

                # FloydWarshallVisualizer
                NewConfClass(
                    "FloydWarshallVisualizerConf",
                    grid_shape     = lambda s: self.grid_world.shape,
                    goal_pose      = lambda s: self.prob_kwargs.goal_pose
                )(
                )
            ],
            grid_world_maze_string = None,
            grid_world_class       = WindyGridWorld,

            grid_world_kwargs = Conf(
                wind_strength = 0.5),

            prob_class = AgentInGridWorld,

            prob_kwargs     = Conf(
                start_pose  = [2, 3],
                goal_pose   = [3, 4],
                goal_reward = 10,
                max_steps   = 100
            ))

        return defaults

    @property
    def alg(self):
        return init_class_stack(self.alg_class_stack, self.alg_kwargs_stack)

    @property
    def grid_world(self):
        return self.grid_world_class(
            self._next_seed(),
            self.grid_world_maze_string,
            **vars(self.grid_world_kwargs))

    @property 
    def prob(self):
        return self.prob_class(
            self._next_seed(),
            self.grid_world,
            **vars(self.prob_kwargs))


CurrentConf = DefaultConf
