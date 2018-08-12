import functools

import numpy as np

from umcog.confutils import xargs, xargsonce, extended_kwprop
from ..prob.windy_grid_world import AgentInGridWorld
from .default import ql_grid_world_play, fw_grid_world_play, NoVisMultiObserverXargs

_ql_grid_world_play = functools.partial(
    ql_grid_world_play,
    observer            = NoVisMultiObserverXargs, # if no visualize
    confname = "ql-4-room-grid-world")

_fw_grid_world_play = functools.partial(
    fw_grid_world_play,
    observer            = NoVisMultiObserverXargs, # if no visualize
    confname = "fw-4-room-grid-world")

@extended_kwprop
def multi_four_room_grid_world_play(
        seed      = 0,
        max_steps = 300,
        maze_name = "4-room-grid-world.txt",
        rng       = xargs(np.random.RandomState, ["seed"]),
        prob      = xargsonce(AgentInGridWorld.from_maze_name,
                             "rng max_steps maze_name".split()),
        gw_plays = [_ql_grid_world_play, _fw_grid_world_play]):
    return [p(prob = prob, seed = seed, max_steps = max_steps,
              rng = rng) for p in gw_plays]

main = multi_four_room_grid_world_play
