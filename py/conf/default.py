from argparse import Action, ArgumentParser
import shlex
import numpy as np
import hashlib
import pickle
import time
import os
import json
from string import Formatter
from pathlib import Path
import functools

from cog.confutils import (extended_kwprop, KWProp as prop, xargs,
                           xargmem, FuncAttr, FuncProp)

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

def random_state(seed = 0):
    return np.random.RandomState(seed = 0)

def agent_from_maze_file_path(rng, maze_file_path, **kwargs):
    return AgentInGridWorld.from_maze_file_path(
        rng = rng,
        maze_file_path = maze_file_path, **kwargs)

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
        prob           = xargmem(agent_from_maze_file_path,
                                  "rng maze_file_path".split()),
        project_name   = PROJECT_NAME,
        maze_file_path = prop(lambda s:
                              s.maze_file_path_template.format(self=s)),
        file_work_dir  = prop(lambda s: Path(__file__).parent),
        maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
        **kwargs):
    return play(alg, prob, observer, nepisodes, logger_factory)


def ql_grid_world_play(
        kw = dict(
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
        ),
        **kwargs
):
    return grid_world_play(**dict(kw, **kwargs))


def fw_grid_world_play(
        kw = dict(
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
            fw_vis = xargs(FloydWarshallLogger, ["logger"]),
        ),
        **kwargs):
    return grid_world_play(**dict(kw, **kwargs))


def multiplay(**plays):
    return { k : v() for k, v in plays.items() } 

@extended_kwprop
def fw_post_process_run(
        log_file_path           = None,
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
        log_file_path           = None,
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
    return ql_post_process(process_data_tag = process_data_tag, data_iter = data_iter)

@extended_kwprop
def AgentVisSessionConf(
        log_file_path           = None,
        windy_grid_world        = xargs(
            WindyGridWorld.from_maze_file_path,
            "rng maze_file_path".split()),
        cellsize                = 100,
        process_data_tag        = xargs(
            functools.partial,
            "windy_grid_world cellsize image_file_fmt".split(),
            DrawAgentGridWorldFromLogs()),
        log_file_conf           = xargmem(LogFileConf,
                                   "project_name confname".split()),
        rng                     = xargmem(random_state),
        data_iter               = xargs(
            functools.partial,
            "log_file_reader filter_criteria".split(),
            ql_post_process_data_iter),
        filter_criteria         = dict(tag = 
            ['LoggingObserver:new_episode', 'LoggingObserver:new_step']),
        maze_file_path          = prop(lambda s: s.maze_file_path_template.format(self = s)),
        file_work_dir           = prop(lambda s: Path(__file__).parent),
        maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
        project_name            = PROJECT_NAME,
        confname                = "AgentVis",
):
    return ql_post_process(process_data_tag, data_iter)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=lambda s: globals()[s],
                        default=ql_grid_world_play)
    a, remargs = parser.parse_known_args()
    a.config()
    #import sys
    #c = FloydWarshallPostProcessConf(confname="FloydWarshall")
    #conf = ConfFromDotArgs(c).from_args(sys.argv[1:])
    #conf()
