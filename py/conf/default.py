from argparse import Action
import shlex
import numpy as np
import hashlib
import pickle
import time
import os
import json
from string import Formatter
import logging
import logging.config
from pathlib import Path
import functools

from cog.confutils import (Conf, dict_update_recursive, FallbackConf,
                           func_kwonlydefaults, makeconf,
                           make_fallback_conf)

from cog.memoize import MEMOIZE_METHOD, LambdaMethodMemoizer
from cog.misc import ensuredirs, git_revision, pairwise

from alg.floyd_warshall_grid import (FloydWarshallAlgDiscrete,
                                     FloydWarshallVisualizer,
                                     FloydWarshallLogger)
from alg.qlearning import QLearningDiscrete, QLearningVis, QLearningLogger
from alg.qlearning import post_process as qlearning_post_process
from alg.floyd_warshall_grid import post_process as fw_post_process
from game.metrics import (ComputeMetricsFromLogReplay,
                          LatencyObserver, DistineffObs)
from game.play import (LoggingObserver, MultiObserver, play,
                       NoOPObserver, LogFileWriter, NPJSONEncDec,
                       JSONLoggingFormatter, LogFileReader)
from prob.windy_grid_world import (WindyGridWorld, AgentInGridWorld,
                                   random_goal_pose_gen,
                                   random_start_pose_gen,
                                   maze_from_filepath,
                                   Act2DSpace, Loc2DSpace)
PROJECT_NAME = "floyd_warshall_rl"

def logging_dictConfig(log_file, logging_encdec):
    return dict(
        version = 1,
        formatters = dict(
            json_fmt = {
                '()' : JSONLoggingFormatter,
                'enc' : logging_encdec,
                'sep' : "\t",
                'format' : "%(asctime)s %(name)-15s %(message)s",
                'datefmt' : "%d %H:%M:%S"
            }
        ),
        handlers = dict(
            file = {
                'class' : "logging.FileHandler",
                'filename' : log_file,
                'formatter' : "json_fmt",
                'level' : "DEBUG",
            },
            console = {
                'class' : "logging.StreamHandler",
                'level' : "INFO"
            }
        ),
        root = dict(
            level = 'DEBUG',
            handlers = "console file".split()
        )
    )


WindyGridWorldConf = make_fallback_conf(WindyGridWorld)


def find_latest_file(dir_):
    p_stats = [(p, p.stat()) for p in Path(dir_).iterdir() if p.is_file()]
    return max(p_stats, key = lambda p_stat: p_stat[1].st_mtime)[0]


LogFileConf = FallbackConf(
    props = dict(
        log_file = lambda self: ensuredirs(
            self.log_file_template.format(self=self)),
        log_file_dir = lambda self: ensuredirs(
            self.log_file_dir_template.format(self=self)),
        data_dir     = lambda self: os.environ["MID_DIR"],
        exp_name     = lambda self: self.exp_name_template.format(self=self),
        gitrev       = LambdaMethodMemoizer(
            "gitrev")(lambda self: git_revision(Path(__file__).parent)),
        run_month    = lambda self: self.run_full_time[:6],
        run_time     = lambda self: self.run_full_time[6:],
        run_full_time     = LambdaMethodMemoizer(
            "run_full_time") (lambda self: time.strftime(self.run_full_time_format)),
        image_file_fmt = lambda self: self.image_file_fmt_template.format(self=self),
        log_file_latest = lambda s : find_latest_file(s.log_file_dir),
        retval = lambda s : s
    ),
    attrs = dict(
        __name__                 = "LogFileConf",
        exp_name_template        = "{self.run_month}_{self.gitrev}_{self.confname}",
        log_file_dir_template    = "{self.data_dir}/{self.project_name}/{self.exp_name}",
        log_file_template        = "{self.log_file_dir}/{self.run_time}.log",
        run_full_time_format     = "%Y%m%d-%H%M%S",
        image_file_fmt_template  = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
        #project_name             = PROJECT_NAME,
        #confname                 = "LogFileConf",
    ),
)


Obs2DSpaceConf       = make_fallback_conf(Loc2DSpace)
Act2DSpaceConf       = make_fallback_conf(Act2DSpace)
AgentInGridWorldConf = make_fallback_conf(AgentInGridWorld)
QLearningDiscreteConf = make_fallback_conf(QLearningDiscrete)
LoggingEncdecConf    = make_fallback_conf(NPJSONEncDec)


def setLoggerConfig(confname, log_file, logging_encdec):
    print("Setting dict config from {confname}".format(confname=confname))
    logging.root = logging.RootLogger(logging.WARNING)
    logging.Logger.root = logging.root
    logging.Logger.manager = logging.Manager(logging.Logger.root)
    logging.config.dictConfig(
        logging_dictConfig(log_file,
                           logging_encdec))


def logger_factory(set_logger_conf):
    return (lambda name : logging.getLogger(name))
    

LoggerFactoryConf = make_fallback_conf(logger_factory)
LoggingObserverConf = make_fallback_conf(
    LoggingObserver,
    props = dict(
        logger = lambda s : s.logger_factory("LoggingObserver")
    ),
)
LatencyObserverConf = make_fallback_conf(LatencyObserver)
DistineffObsConf = make_fallback_conf(DistineffObs)
LogFileReaderConf = make_fallback_conf(
    LogFileReader,
    props = dict(
        logfilepath    = lambda s : s.log_file_conf.log_file_latest,
        enc            = lambda s : s.logging_encdec,
    ))

ComputeMetricsFromLogReplayConf = make_fallback_conf(
    ComputeMetricsFromLogReplay,
    props = dict(
        loggingobserver = lambda s: s.logging_observer,
        metrics_observers = lambda s : [s.latency_observer,
                                        s.distineff_observer],
        logfilereader   = lambda s: s.log_file_reader
    ))


VisualizerConf = make_fallback_conf(
    NoOPObserver,
    props = dict(
        logger = lambda s : s.logger_factory("Visualizer")
    ),
    log_interval = 1,
    cellsize = 80)


MultiObserverConf = make_fallback_conf(
    MultiObserver,
    props = dict(
        observers = lambda s: dict(
            logging = s.logging_observer,
            metrics = s.metrics_observer,
            visualizer = s.visualizer_observer)
    ))

QLearningLoggerConf = make_fallback_conf(
    QLearningLogger,
    props = dict(
        logger = lambda self: self.logger_factory("QLearningLogger"),
    ))


QLearningPlayConf = make_fallback_conf(
    play,
    props = dict(
        alg = QLearningDiscreteConf,
        visualizer_observer = QLearningLoggerConf,
    ))

QLearningPostProcessConf = make_fallback_conf(qlearning_post_process)


FloydWarshallAlgDiscreteConf = make_fallback_conf(
    FloydWarshallAlgDiscrete,
        props = dict(
            qlearning = QLearningDiscreteConf,
    ))


FloydWarshallLoggerConf = make_fallback_conf(
    FloydWarshallLogger,
    props = dict(
        logger = lambda self: self.logger_factory("FloydWarshallLogger"),
    ))

FloydWarshallPlayConf = make_fallback_conf(
    play,
    props = dict(
        alg = FloydWarshallAlgDiscreteConf,
        visualizer_observer = FloydWarshallLoggerConf,
    ))

def QLearningPlaySessionConf(
        confname,
        props = dict(
            log_file_conf       = MEMOIZE_METHOD(LogFileConf),
            logging_encdec      = LoggingEncdecConf,
            set_logger_conf     = lambda s : setLoggerConfig(s.log_file_conf.confname,
                                                             s.log_file_conf.log_file,
                                                             s.logging_encdec),
            logger_factory      = MEMOIZE_METHOD(LoggerFactoryConf),
            rng                 = LambdaMethodMemoizer("rng")(
                lambda s: np.random.RandomState(seed=0)),
            maze                = lambda s: maze_from_filepath(
                s.maze_file_path_template.format(self = s)),
            file_work_dir       = lambda s: Path(__file__).parent,
            grid_world          = MEMOIZE_METHOD(WindyGridWorldConf),
            upper_bound         = lambda s: np.array(s.grid_world.shape),
            log_file_dir        = lambda s: s.log_file_conf.log_file_dir,
            action_space        = Act2DSpaceConf,
            observation_space   = Obs2DSpaceConf,
            prob                = MEMOIZE_METHOD(AgentInGridWorldConf),
            logging_observer    = LoggingObserverConf,
            metrics_observer    = ComputeMetricsFromLogReplayConf,
            latency_observer    = LatencyObserverConf,
            distineff_observer  = DistineffObsConf,
            log_file_reader     = LogFileReaderConf,
            observer            = MultiObserverConf,
            play                = QLearningPlayConf,
            visualizer_observer = QLearningLoggerConf,
        ),
        attrs = dict(
            seed                    = 1,
            project_name            = PROJECT_NAME,
            maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
            lower_bound             = np.array([0, 0]),
            nepisodes               = 3,
            log_interval            = 1,
        ),
        retkey = "play",
    ):
    return Conf(props = props, attrs = dict(attrs, confname = confname), retkey=retkey)


def FloydWarshallPlaySessionConf(confname):
    return Conf(
        retkey = "play",
        props = dict(
            play                = FloydWarshallPlayConf,
            visualizer_observer = FloydWarshallLoggerConf,
        ),
        fallback = QLearningPlaySessionConf(confname))
    

def multiplay(plays):
    return { k : v() for k, v in plays.items() } 

MultiPlaySessionConf = makeconf(
    multiplay,
    plays = dict(
        ql = QLearningPlaySessionConf,
        fw = FloydWarshallLoggerConf
    ))

FloydWarshallPostProcessConf = makeconf(fw_post_process,
        props = dict(
            image_file_fmt = lambda s: s.log_file_conf.image_file_fmt
        ))

if __name__ == '__main__':
    import sys
