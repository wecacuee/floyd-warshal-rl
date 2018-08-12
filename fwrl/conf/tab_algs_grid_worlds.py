import functools

import numpy as np

from umcog.confutils import xargs, xargsonce, extended_kwprop
from ..prob.windy_grid_world import AgentInGridWorld
from .default import (ql_grid_world_play as _ql_grid_world_play,
                      _fw_grid_world_play as _fw_grid_world_play,
                      NoVisMultiObserverXargs)

@extended_kwprop
def tab_algs_grid_worlds(
        seed      = 0,
        nepisodes = 20,
        max_steps = 400,
        maze_names = ["4-room-grid-world.txt", "4-room-lava-world.txt",
                      "4-room-windy-world"],
        rng       = xargs(np.random.RandomState, ["seed"]),
        probs      = xargsonce(
            lambda s: [AgentInGridWorld.from_maze_name(
                rng = s.rng, max_steps = s.max_steps, mn)
                       for mn in s.maze_names]),
        alg_names = ["ql", "fw"],
        gw_plays = [_ql_grid_world_play, _fw_grid_world_play]):
    return [[
        play(prob      = prob,
             seed      = seed,
             max_steps = max_steps,
             rng       = rng,
             confname  = "-".join(alg, mn))
        for mn, prob in zip(maze_names, probs)]
            for alg, play in zip(alg_names, gw_plays)]

main = tab_algs_grid_worlds
