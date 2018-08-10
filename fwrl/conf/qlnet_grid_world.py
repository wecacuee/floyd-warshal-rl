import functools

from umcog.confutils import xargs, KWProp as prop
from .default import grid_world_play, AgentVisMultiObserverXargs, NoVisMultiObserverXargs
from ..alg.qlnet import QLearningNetAgent
from ..game.logging import NPJSONEncDec, LogFileConf
from ..game.play import (MultiObserver, play, LogFileReader, NoOPObserver)

qlnet_grid_world_play = functools.partial(
    grid_world_play,
    confname = "qlnet_grid_world_play",
    alg                 = xargs(QLearningNetAgent,
                                "action_space observation_space reward_range rng".split()),
    action_space        = prop(lambda s : s.prob.action_space),
    observation_space   = prop(lambda s : s.prob.observation_space),
    reward_range        = prop(lambda s : s.prob.reward_range),
    windy_grid_world    = prop(lambda s : s.prob.grid_world),
    #observer            = AgentVisMultiObserverXargs, # if visualize
    observer            = NoVisMultiObserverXargs, # if no visualize
    log_file_dir        = prop(lambda s: s.log_file_conf.log_file_dir),
    logger              = prop(lambda s: s.logger_factory("QLearningLogger")),
    log_file_reader     = xargs(LogFileReader, ["log_file_path"], enc = NPJSONEncDec()),
)
