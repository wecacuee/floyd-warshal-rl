from cog.confutils import (Conf, dict_update_recursive,
                           init_class_stack, NewConfClass)
import numpy as np

class PlayConf(Conf):
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
        from alg.qlearning import QLearningDiscrete
        from alg.floyd_warshall_grid import (FloydWarshallAlgDiscrete,
                                             FloydWarshallVisualizer)
        from prob.windy_grid_world import (WindyGridWorld, AgentInGridWorld)
        from game.play import LoggingObserver
        defaults      = dict(
            nepisodes = 10,
            seed      = 0,

            alg_class_stack = [QLearningDiscrete, FloydWarshallAlgDiscrete],

            alg_kwargs_stack            = [

                # QLearningDiscrete
                NewConfClass(
                    "QLearningDiscreteConf",
                    action_space        = lambda s: self.prob.action_space,
                    observation_space   = lambda s: self.prob.observation_space,
                    seed                = lambda s: self._next_seed(),
                )(egreedy_epsilon       = 0.2,
                  action_value_momentum = 0.1, # Low momentum changes more frequently
                  init_value            =   1,
                  discount              = 0.99,
                ),

                # FloydWarshallAlgDiscrete
                Conf(
                  path_cost_momentum    = 0.9,  # High momemtum changes less frequently
                ),
            ],
            grid_world_maze_string = None,
            grid_world_class       = WindyGridWorld,

            grid_world_kwargs = Conf(
                wind_strength = 0.1),

            prob_class = AgentInGridWorld,

            prob_kwargs     = Conf(
                start_pose_gen = lambda prob: prob.grid_world.valid_random_pos(),
                goal_pose_gen  = lambda prob: prob.grid_world.valid_random_pos(),
                goal_reward    = 10,
                max_steps      = 200,
                wall_penality  = 1.0
            ),

            observer_classes = dict(logger = LoggingObserver,
                                    visualizer = FloydWarshallVisualizer),
            observer_classes_kwargs = [dict(), dict()],
        )

        return defaults

    @property
    def alg(self):
        return init_class_stack(self.alg_class_stack, self.alg_kwargs_stack)

    @property
    def observer(self):
        from game.play import MultiObserver
        return MultiObserver(
            { name : obs_class(**kw)
              for (name, obs_class), kw in zip(self.observer_classes.items(), 
                                               self.observer_classes_kwargs)})

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

