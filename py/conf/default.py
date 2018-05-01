from argparse import Action, ArgumentParser
import shlex
import hashlib
import pickle
import time
import os
import json
from string import Formatter
from pathlib import Path
import functools

#import torch as tch
import numpy as np

from cog.confutils import (extended_kwprop, KWProp as prop, xargs,
                           xargmem, xargspartial,
                           parse_args_update_kwasattr, KWAsAttr)

from cog.memoize import LambdaMethodMemoizer, MEMOIZE_METHOD

from alg.floyd_warshall_grid import (FloydWarshallAlgDiscrete,
                                     FloydWarshallLogger)
from alg.qlearning import QLearningDiscrete, QLearningLogger
from alg.qlearning import (post_process_from_log_conf as
                           ql_post_process_from_log_conf,
                           post_process_generic, post_process_data_iter)

from alg.floyd_warshall_grid import (post_process_from_log_conf as
                                     fw_post_process_from_log_conf)

from game.play import (MultiObserver, play, LogFileReader, NoOPObserver)
from game.logging import NPJSONEncDec, LogFileConf
from prob.windy_grid_world import (AgentInGridWorld, WindyGridWorld,
                                   DrawAgentGridWorldFromLogs, AgentVisObserver)
from game.vis_imgs_to_video import ImgsToVideoObs

PROJECT_NAME = "floyd_warshall_rl"

class TorchRng:
    @staticmethod
    def rand(size = 1):
        return tch.rand(size)

    @staticmethod
    def randn():
        return tch.randn()

    @staticmethod
    def uniform(size = 1):
        return tch.FloatTensor((size,)).uniform_()

    def randint(self, low, high = None, size = 1):
        if high is None:
            low, high = 0, low
        return tch.LongTensor((size,)).random_(low, high, generator = self.gen)

def random_state_torch(seed):
    rng = TorchRng()
    rng.gen = tch.random.manual_seed(seed)
    return rng


random_state = np.random.RandomState


AgentVisMultiObserver = functools.partial(
    MultiObserver,
    observer_keys = """logging_observer metrics_observers
                       visualizer_observer agent_vis_observer
                       imgs_to_vid_observers""".split(),
    agent_vis_observer = xargs(
        AgentVisObserver,
        "log_file_path log_file_dir maze_file_path".split()),
    imgs_to_vid_observers = xargs(ImgsToVideoObs, ["log_file_dir", "nepisodes"]),
    # Needs: log_file_dir log_file_path maze_file_path nepisodes
)


AgentVisMultiObserverXargs = xargs(
    AgentVisMultiObserver,
    """prob logger_factory log_file_path
    logging_encdec log_file_dir
    maze_file_path visualizer_observer nepisodes""".split())
                                   

@extended_kwprop
def grid_world_play(
        alg            = None,
        observer       = None,
        nepisodes      = 3,
        seed           = 0,
        log_file_conf  = xargmem(LogFileConf,
                               "project_name confname".split()),
        rng            = xargs(random_state, ["seed"]),
        log_file_path  = prop(lambda s: s.log_file_conf.log_file),
        logger_factory = prop(lambda s: s.log_file_conf.logger_factory),
        logging_encdec = prop(lambda s: s.log_file_conf.logging_encdec),
        prob           = xargmem(AgentInGridWorld.from_maze_file_path,
                                  "rng maze_file_path".split()),
        project_name   = PROJECT_NAME,
        maze_file_path = prop(lambda s:
                              s.maze_file_path_t.format(self=s)),
        file_work_dir  = prop(lambda s: Path(__file__).parent),
        maze_file_path_t = "{self.file_work_dir}/maze_5x5_no_wind.txt",
        image_file_fmt   = prop(
            lambda s: s.image_file_fmt_t.format(self=s)),
        log_file_dir     = prop(
            lambda s: s.log_file_conf.log_file_dir),
        image_file_fmt_t = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
        action_value_img_fmt_t = prop(lambda s : s.image_file_fmt_t),
        **kwargs):
    return play(alg, prob, observer, nepisodes, logger_factory)


ql_grid_world_play = functools.partial(
    grid_world_play,
    confname            = "ql_grid_world_play",
    alg                 = xargs(QLearningDiscrete,
                                "action_space observation_space reward_range rng".split()),
    action_space        = prop(lambda s : s.prob.action_space),
    observation_space   = prop(lambda s : s.prob.observation_space),
    reward_range        = prop(lambda s : s.prob.reward_range),
    observer            = AgentVisMultiObserverXargs,
    visualizer_observer = xargs(QLearningLogger,
                                "logger image_file_fmt log_file_reader".split()),
    log_file_dir        = prop(lambda s: s.log_file_conf.log_file_dir),
    logger              = prop(lambda s: s.logger_factory("QLearningLogger")),
    log_file_reader     = xargs(LogFileReader, ["log_file_path"], enc = NPJSONEncDec()),
)


fw_grid_world_play = functools.partial(
    grid_world_play,
    **dict(ql_grid_world_play.keywords,
           confname            = "fw_grid_world_play",
           alg                 = xargs(FloydWarshallAlgDiscrete,
                                       "action_space observation_space reward_range rng".split()),
           logger              = prop(lambda s: s.logger_factory("FloydWarshallLogger")),
           visualizer_observer = xargs(FloydWarshallLogger,
                                       "logger image_file_fmt log_file_reader".split()))
)


fw_post_process_run = functools.partial(
    fw_post_process_from_log_conf,
    project_name            = PROJECT_NAME,
    confname                = "FWPostProc",
    log_file_conf           = xargmem(LogFileConf,
                                      "project_name confname".split()),
    log_file_dir            = prop(lambda s: s.log_file_conf.log_file_dir),
    log_file_reader         = xargs(LogFileReader, ["log_file_path"], enc = NPJSONEncDec()),
    action_value_tag        = "FloydWarshallLogger:action_value",
    # Needs
    # log_file_path
)


ql_post_process_run = functools.partial(
    ql_post_process_from_log_conf,
    **dict(fw_post_process_run.keywords,
           confname                = "QLPostProc",
           action_value_tag        = "QLearningLogger:action_value")
    # Needs:
    # log_file_path
)

AgentVisSessionConf = functools.partial(
    AgentVisObserver,
    log_file_dir_template   = prop(lambda s : str(Path(s.log_file_path).parent)),
    log_file_dir            = prop(lambda s: s.log_file_conf.log_file_dir),
    log_file_conf           = xargmem(LogFileConf,
                                      "project_name confname log_file_dir_template".split()),
    maze_file_path          = prop(lambda s: s.maze_file_path_template.format(self = s)),
    file_work_dir           = prop(lambda s: Path(__file__).parent),
    maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
    project_name            = PROJECT_NAME,
    confname                = "AgentVis",
    # Needs: log_file_path
)

def listget(l, i, default):
    return l[i] if i < len(l) else default

if __name__ == '__main__':
    import sys
    main_func = globals()[listget(sys.argv, 1, 'ql_grid_world_play')]
    kwasattr = parse_args_update_kwasattr(KWAsAttr(main_func),
                                          argv = sys.argv[2:])
    kwasattr()

