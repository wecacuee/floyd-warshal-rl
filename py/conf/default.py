from cog.confutils import (Conf, dict_update_recursive,
                           init_class_stack, NewConfClass)
import numpy as np

def AgentInGridWorldDefaultConfGen(play_conf):
    from prob.windy_grid_world import random_goal_pose_gen, random_start_pose_gen
    return Conf(
        start_pose_gen = random_start_pose_gen,
        goal_pose_gen  = random_goal_pose_gen,
        goal_reward    = 10,
        max_steps      = 200,
        wall_penality  = 1.0)

def QLearningDiscreteDefaultConfGen(play_conf):
    return NewConfClass(
        "QLearningDiscreteConf",
        action_space        = lambda s: play_conf.prob.action_space,
        observation_space   = lambda s: play_conf.prob.observation_space,
        seed                = lambda s: play_conf._next_seed(),
    )(egreedy_epsilon       = 0.2,
      action_value_momentum = 0.1, # Low momentum changes more frequently
      init_value            =   1,
      discount              = 0.99,
    )

def WindyGridWorldDefaultConfGen(play_conf):
    return Conf(wind_strength = 0.1)

class CommonPlayConf(Conf):
    """
    Follow the principle of delayed execution.

    1. Avoid executing any code untill a property of this conf is accessed.
    2. Do not make any copies of same attributes.
        Use property objects for redirection
    3. Use class stacks for additional properties on the base classes.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self._rng = None
        self._alg = None
        self._grid_world = None
        self._prob = None
        self._observer = None

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.RandomState(self.seed)
        return self._rng

    def _next_seed(self):
        return self.rng.randint(1000)

    @property
    def alg(self):
        return init_class_stack(self.alg_class_stack, self.alg_kwargs_stack)

    @property
    def observer(self):
        from game.play import MultiObserver
        if self._observer is None:
            self._observer = MultiObserver(
                { name : obs_class(**vars(kw))
                  for (name, obs_class), kw in zip(
                          self.observer_classes.items(), 
                          self.observer_classes_kwargs) })
        return self._observer

    @property
    def grid_world(self):
        if self._grid_world is None:
            self._grid_world = self.grid_world_class(
                self._next_seed(),
                self.grid_world_maze_string,
                **vars(self.grid_world_kwargs))
        return self._grid_world

    @property 
    def prob(self):
        if self._prob is None:
            self._prob = self.prob_class(
                self._next_seed(),
                self.grid_world,
                **vars(self.prob_kwargs))
        return self._prob

    def defaults(self):
        from prob.windy_grid_world import (WindyGridWorld, AgentInGridWorld)
        defaults      = dict(
            nepisodes = 100,
            seed      = 0,

            grid_world_maze_string = None,
            grid_world_class       = WindyGridWorld,
            grid_world_kwargs      = WindyGridWorldDefaultConfGen(self),

            prob_class = AgentInGridWorld,
            prob_kwargs = AgentInGridWorldDefaultConfGen(self),
        )
        return defaults

class FloydWarshallPlayConf(CommonPlayConf):
    def defaults(self):
        from alg.qlearning import QLearningDiscrete
        from alg.floyd_warshall_grid import (FloydWarshallAlgDiscrete,
                                             FloydWarshallVisualizer)
        from game.play import LoggingObserver
        from game.metrics import LatencyObserver
        defaults = super().defaults()
        defaults.update(dict(
            alg_class_stack = [QLearningDiscrete, FloydWarshallAlgDiscrete],
            alg_kwargs_stack            = [
                QLearningDiscreteDefaultConfGen(self),

                # FloydWarshallAlgDiscrete
                Conf(),
            ],

            observer_classes = dict(logger     = LoggingObserver,
                                    #visualizer = FloydWarshallVisualizer,
                                    metrics    = LatencyObserver),
            observer_classes_kwargs = [Conf(),
                                       #Conf(update_interval = 20),
                                       Conf()],
        ))
        return defaults

class QLearningPlayConf(CommonPlayConf):
    def defaults(self):
        from alg.qlearning import QLearningDiscrete, QLearningVis
        from game.play import LoggingObserver
        from game.metrics import LatencyObserver
        defaults = super().defaults()
        defaults.update(dict(
            alg_class_stack = [QLearningDiscrete],
            alg_kwargs_stack            = [
                QLearningDiscreteDefaultConfGen(self),
            ],

            observer_classes = dict(logger     = LoggingObserver,
                                    #visualizer = QLearningVis,
                                    metrics    = LatencyObserver),
            observer_classes_kwargs = [Conf(),
                                       #Conf(update_interval = 20),
                                       Conf()],
        ))
        return defaults

class MultiPlayConf(QLearningPlayConf):
    def defaults(self):
        from game.play import NoOPObserver
        return dict(trials_classes = [FloydWarshallPlayConf, QLearningPlayConf])

    @property
    def trials(self):
        return [class_(**vars(self)) for class_ in self.trials_classes]


        
