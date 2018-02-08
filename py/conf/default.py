import importlib
from argparse import Namespace, ArgumentParser

import numpy as np

class Conf(Namespace):
    def items(self):
        return vars(self).items()

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    @classmethod
    def parser(cls):
        parser = ArgumentParser()
        parser.add_argument("--config", default="conf.default:CurrentConf")
        return parser

    @classmethod
    def import_class(cls, name):
        module_name, class_ = name.strip().split(":")
        module = importlib.import_module(module_name)
        return getattr(module, class_)

    @classmethod
    def parse_all_args(cls, args):
        c, remargs = cls.parser().parse_known_args(args)
        conf = cls.import_class(c.config)()
        return conf.parse_remargs(remargs)

    def parse_remargs(self, remargs):
        parser = self.parser()
        for k, v in vars(self).items():
            if isinstance(v, Conf):
                conf_from_args = lambda a : v.parse_remargs(a)
                parser.add_argument(
                    f"--{k}", default=None, type=conf_from_args)
            else:
                parser.add_argument(f"--{k}", default=None, type=type(v))
        args = parser.parse_args(remargs)
        self.__init__(
            **{ k : v for k, v in vars(args).items() if v is not None })
        self.run_checks()
        return self

    def run_checks(self):
        return True

def dict_update_recursive(ddest, dsrc):
    for k, v in dsrc.items():
        if hasattr(v, "items"):
            if k in ddest: 
                ddest[k] = dict_update_recursive(ddest[k], v)
            else:
                ddest[k] = v.copy()
        else:
            ddest[k] = v

    return ddest

class DefaultConf(Conf):
    def __init__(self, **kw):
        defaults = self.defaults()
        updated_kw = dict_update_recursive(defaults, kw)
        super().__init__(**updated_kw)
        self.rng = np.random.RandomState(self.seed)

    def _next_seed(self):
        return self.rng.randint(1000)

    def defaults(self):
        from alg.floyd_warshall_grid import FloydWarshallAlgDiscrete
        from prob.windy_grid_world import (WindyGridWorld, AgentInGridWorld)
        defaults      = dict(
            alg_class = FloydWarshallAlgDiscrete,

            alg_kwargs               = Conf(
                egreedy_epsilon      = 0.5,
                path_value_momentum  = 0.9, # High momemtum changes less frequently
                state_value_momentum = 0.1,
                init_value           = 100,
                top_value_queue_size = 5,
                per_edge_reward      = -1),

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
            ),

            nepisodes = 2,
            seed      = 0)
        return defaults

    @property
    def alg(self):
        return self.alg_class(
            self.prob.action_space, self.prob.observation_space,
            self._next_seed(),
            **vars(self.alg_kwargs))

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
