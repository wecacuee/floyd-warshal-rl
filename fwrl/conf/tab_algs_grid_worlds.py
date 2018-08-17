from functools import partial
from itertools import repeat
import collections

import numpy as np

from umcog.confutils import xargs, xargsonce, xargspartial, extended_kwprop
from umcog.misc import kwmap, kwcompose
from ..alg.qlearning import Renderer as QRenderer, QLearningLogger
from ..alg.floyd_warshall_grid import FloydWarshallLogger
from ..prob.windy_grid_world import AgentInGridWorld, AgentVisObserver
from ..game.play  import Renderer, play_episode
from .default import (grid_world_play,
                      ql_grid_world_play as _ql_grid_world_play,
                      fw_grid_world_play as _fw_grid_world_play,
                      AgentVisMultiObserver,
                      AgentVisMultiObserverXargs,
                      NoVisMultiObserverXargs)


def AgentInGridWorlds_from_maze_names(rng = None, max_step = None, maze_names = []):
    return [AgentInGridWorld.from_maze_name(
        rng = rng, max_steps = max_steps, maze_name = mn)
            for mn in maze_names]


def isiterable(e):
    return isinstance(e, collections.Iterable)


def scalars_repeat(**kw):
    return {k : (v if isiterable(v) else repeat(v))
            for k, v in kw.items()}


AgentInGridWorlds_from_maze_names = partial(kwmap, AgentInGridWorld.from_maze_name)


AgentInGridWorlds_from_maze_names_repeat = kwcompose(
    AgentInGridWorlds_from_maze_names, scalars_repeat)

ql_grid_world_play = partial(_ql_grid_world_play,
                             visualizer_observer = xargs(
                                 partial(QLearningLogger, renderer = QRenderer.human),
                                 "logger image_file_fmt log_file_reader".split()))
fw_grid_world_play = partial(_fw_grid_world_play,
                visualizer_observer = xargs(
                    partial(FloydWarshallLogger, renderer = QRenderer.human),
                    "logger image_file_fmt log_file_reader".split()))

AgentVisHumanMultiObserver = partial(
    AgentVisMultiObserver,
    agent_vis_observer = xargs(
        partial(
            AgentVisObserver,
            show_ax = xargspartial(
                AgentVisObserver.show_ax_human, ["image_file_fmt"])),
        "log_file_path log_file_dir windy_grid_world".split()))

AgentVisHumanMultiObserverXargs = xargs(
    AgentVisHumanMultiObserver,
    """prob logger_factory log_file_path
    logging_encdec log_file_dir
    windy_grid_world visualizer_observer nepisodes""".split())

@extended_kwprop
def tab_algs_grid_worlds(
        seed      = 0,
        nepisodes = 20,
        max_steps = [4000,
                     1000,
                     1000
        ],
        maze_name = ["4-room-lava-world",
                     "4-room-windy-world",
                     "4-room-grid-world"
        ],
        rng       = xargs(np.random.RandomState, ["seed"]),
        probs      = xargsonce(
            AgentInGridWorlds_from_maze_names_repeat,
            "rng max_steps maze_name".split()),
        alg_names = ["ql",
                     "fw"],
        gw_plays = [_ql_grid_world_play,
                    _fw_grid_world_play],
):
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
                #play_episode  = partial(play_episode, renderer = Renderer.human),
                observer  = NoVisMultiObserverXargs,
                #observer = AgentVisHumanMultiObserverXargs,
                confname = confname)
            return_vals_per_alg.append(ret)
        return_vals.append(return_vals_per_alg)
    return return_vals

main = tab_algs_grid_worlds
