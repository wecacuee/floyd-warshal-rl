if not __package__: __package__ = "fwrl.conf"

from functools import partial

import numpy as np

from umcog.confutils import extended_kwprop, KWProp, xargsonce, alias
from umcog.misc import compose

from ..prob.windy_grid_world import AgentInGridWorld, AgentVisObserver
from ..prob.scrolling_grid_world import AgentInScrollingGW

from ..alg.qlnet import QLearningNetAgent, RLinNet, MLP

from ..game.play import play
from ..game.logging import LogFileConf

from .default import grid_world_play, PROJECT_NAME

play_scrolling = partial(
    grid_world_play,
    maze_name         = "4-room-grid-world",
    nepisodes         = 1000,
    seed              = 0,
    max_steps         = 400,
    project_name      = PROJECT_NAME,
    confname          = "qlnet_scrolling",
    rng               = xargsonce(np.random.RandomState, ["seed"]),
    prob              = xargsonce(AgentInScrollingGW, ["maze_name"]),
    action_space      = alias(["prob", "action_space"]),
    observation_space = alias(["prob", "observation_space"]),
    reward_range      = alias(["prob", "reward_range"]),
    log_file_conf     = xargsonce(LogFileConf, ["project_name", "confname"]),
    model_save_dir    = alias(["log_file_conf", "log_file_dir"]),
    qnet              = partial(RLinNet, hiddens = [32, 8, 32], D_out = 16),
    alg               = xargsonce(QLearningNetAgent,
                              """action_space observation_space reward_range
                              rng nepisodes qnet model_save_dir""".split()),
)

def main():
    return play_scrolling()

if __name__ == '__main__':
    main()
