import functools
from functools import partial
from itertools import repeat
import collections

import numpy as np

from umcog.confutils import xargs, xargsonce, extended_kwprop
from ..prob.windy_grid_world import AgentInGridWorld
from .default import (grid_world_play,
                      ql_grid_world_play as _ql_grid_world_play,
                      fw_grid_world_play as _fw_grid_world_play,
                      NoVisMultiObserverXargs)

def transpose_dict(dict_of_lists):
    list_of_dicts = None
    max_len = max(map(len, dict_of_lists.values()))
    for k, list_ in dict_of_lists.items():
        if list_of_dicts is None:
            list_of_dicts = [dict()] * len(list_)
        for i, l in enumerate(list_):
            list_of_dicts[i][k] = l
    return list_of_dicts

def AgentInGridWorlds_from_maze_names(rng = None, max_step = None, maze_names = []):
    return [AgentInGridWorld.from_maze_name(
        rng = rng, max_steps = max_steps, maze_name = mn)
            for mn in maze_names]

def isiterable(e):
    return isinstance(e, collections.Iterable)

def scalars_repeat(values):
    return ((repeat(v) if not isiterable(v) else v) for v in values)

def zip_scalars_repeat(*values):
    return zip(*scalars_repeat(values))

def dictzip(kwiterables):
    keys, values = zip(*kwiterables.items())
    return (dict(zip(keys, v)) for v in zip_scalars_repeat(*values))

def kwmap(function, **kwiterables):
    return (function(**kw) for kw in dictzip(kwiterables))


AgentInGridWorlds_from_maze_names =  partial(kwmap, AgentInGridWorld.from_maze_name)


@extended_kwprop
def tab_algs_grid_worlds(
        seed      = 0,
        nepisodes = 20,
        max_steps = [4000, 400, 400],
        maze_name = ["4-room-lava-world",
                     "4-room-windy-world",
                     "4-room-grid-world"],
        rng       = xargs(np.random.RandomState, ["seed"]),
        probs      = xargsonce(
            AgentInGridWorlds_from_maze_names,
            "rng max_steps maze_name".split()),
        alg_names = ["fw", "ql"],
        gw_plays = [_fw_grid_world_play, _ql_grid_world_play]):

    return_vals = []
    prob_args = list(zip(maze_name, probs, max_steps))
    for alg, play in zip(alg_names, gw_plays):
        print("playing alg {}".format(alg))
        return_vals_per_alg  = []
        for mn, prob, max_stps in prob_args:
            confname  = "{}-{}".format(alg, mn)
            print("playing confname {}".format(confname))
            ret = play(
                prob      = prob,
                seed      = seed,
                max_steps = max_stps,
                rng       = rng,
                nepisodes = nepisodes,
                observer  = NoVisMultiObserverXargs,
                confname = confname)
            return_vals_per_alg.append(ret)
        return_vals.append(return_vals_per_alg)
    return return_vals

main = tab_algs_grid_worlds
