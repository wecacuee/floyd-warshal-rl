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
from alg.qlearning import (post_process as ql_post_process,
                           post_process_data_iter as ql_post_process_data_iter,
                           post_process_data_tag as ql_post_process_data_tag)

from alg.floyd_warshall_grid import (post_process as fw_post_process,
                                     post_process_data_iter as fw_post_process_data_iter,
                                     post_process_data_tag as fw_post_process_data_tag)

from game.play import (MultiObserver, play, LogFileReader)
from game.logging import NPJSONEncDec, LogFileConf
from prob.windy_grid_world import (AgentInGridWorld, WindyGridWorld,
                                   DrawAgentGridWorldFromLogs)
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
                              s.maze_file_path_template.format(self=s)),
        file_work_dir  = prop(lambda s: Path(__file__).parent),
        maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
        **kwargs):
    return play(alg, prob, observer, nepisodes, logger_factory)


ql_grid_world_play = functools.partial(
    grid_world_play,
    confname       = "ql_grid_world_play",
    alg            = xargs(
        QLearningDiscrete,
        "action_space observation_space rng".split()),
    action_space   = prop(lambda s : s.prob.action_space),
    observation_space   = prop(lambda s : s.prob.observation_space),
    def_observer   = xargs(MultiObserver,
                           "prob logger_factory log_file_path logging_encdec".split()),
    observer       = prop(lambda s:
                          s.def_observer.add_observer(
                              visualizer_observer = s.qlearning_vis)),
    logger         = prop(lambda s: s.logger_factory("QLearningLogger")),
    qlearning_vis  = xargs(QLearningLogger, ["logger"]),
)


fw_grid_world_play = functools.partial(
    grid_world_play,
    confname = "fw_grid_world_play",
    alg = xargs(FloydWarshallAlgDiscrete,
                "action_space observation_space rng".split()),
    action_space   = prop(lambda s : s.prob.action_space),
    observation_space   = prop(lambda s : s.prob.observation_space),
    def_observer = xargs(
        MultiObserver,
        "prob logger_factory log_file_path logging_encdec".split()),
    observer = prop(lambda s:
                    s.def_observer.add_observer(
                        visualizer_observer = s.fw_vis)),
    logger         = prop(lambda s: s.logger_factory("FloydWarshallLogger")),
    fw_vis = xargs(FloydWarshallLogger, ["logger"])
)

def multiplay(**plays):
    return { k : v() for k, v in plays.items() } 

@extended_kwprop
def fw_post_process_run(
        log_file_path,
        project_name            = PROJECT_NAME,
        confname                = "FWPostProc",
        image_file_fmt_template = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
        cellsize                = 100,
        log_file_conf           = xargmem(LogFileConf,
                                   "project_name confname".split()),
        log_file_dir            = prop(lambda s: s.log_file_conf.log_file_dir),
        image_file_fmt          = prop(lambda s: s.image_file_fmt_template.format(self=s)),
        log_file_reader         = xargs(LogFileReader, ["log_file_path"], enc = NPJSONEncDec()),
        process_data_tag        = xargs(functools.partial,
                                 "cellsize image_file_fmt".split(),
                                 fw_post_process_data_tag),
        filter_criteria         = dict( tag = "FloydWarshallLogger:action_value"),
        data_iter               = xargs(functools.partial,
                          "log_file_reader filter_criteria".split(),
                          fw_post_process_data_iter),
        ):
    return fw_post_process(process_data_tag = process_data_tag, data_iter = data_iter)

@extended_kwprop
def ql_post_process_run(
        log_file_path,
        project_name            = PROJECT_NAME,
        confname                = "QLPostProc",
        image_file_fmt_template = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
        cellsize                = 100,
        log_file_conf           = xargmem(LogFileConf,
                                   "project_name confname".split()),
        log_file_dir            = prop(lambda s: s.log_file_conf.log_file_dir),
        image_file_fmt          = prop(lambda s: s.image_file_fmt_template.format(self=s)),
        log_file_reader         = xargs(LogFileReader, ["log_file_path"], enc = NPJSONEncDec()),
        process_data_tag        = xargs(
            functools.partial,
            "cellsize image_file_fmt".split(),
            ql_post_process_data_tag),
        filter_criteria         = dict( tag = "QLearningLogger:action_value"),
        data_iter               = xargs(
            functools.partial,
            "log_file_reader filter_criteria".split(),
            ql_post_process_data_iter),
        ):
    return ql_post_process(data_iter = data_iter, process_data_tag = process_data_tag)

@extended_kwprop
def AgentVisSessionConf(
        log_file_path,
        windy_grid_world        = xargs(
            WindyGridWorld.from_maze_file_path,
            "rng maze_file_path".split()),
        cellsize                = 100,
        process_data_tag        = xargspartial(
            DrawAgentGridWorldFromLogs(),
            "windy_grid_world cellsize image_file_fmt".split()),
        image_file_fmt_template = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
        image_file_fmt          = prop(lambda s: s.image_file_fmt_template.format(self=s)),
        log_file_dir_template   = prop(lambda s : str(Path(s.log_file_path).parent)),
        log_file_dir            = prop(lambda s: s.log_file_conf.log_file_dir),
        log_file_conf           = xargmem(LogFileConf,
                                   "project_name confname log_file_dir_template".split()),
        log_file_reader         = xargs(LogFileReader, ["log_file_path"], enc = NPJSONEncDec()),
        rng                     = xargmem(random_state, ["seed"]),
        seed                    = 0,
        data_iter               = xargspartial(
            ql_post_process_data_iter,
            "log_file_reader filter_criteria".split()),
        filter_criteria         = dict(tag = 
            ['LoggingObserver:new_episode', 'LoggingObserver:new_step']),
        maze_file_path          = prop(lambda s: s.maze_file_path_template.format(self = s)),
        file_work_dir           = prop(lambda s: Path(__file__).parent),
        maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
        project_name            = PROJECT_NAME,
        confname                = "AgentVis",
):
    return ql_post_process(data_iter, process_data_tag)

def listget(l, i, default):
    return l[i] if i < len(l) else default

if __name__ == '__main__':
    import sys
    main_func = globals()[listget(sys.argv, 1, 'ql_grid_world_play')]
    kwasattr = parse_args_update_kwasattr(KWAsAttr(main_func),
                                          argv = sys.argv[2:])
    kwasattr()

