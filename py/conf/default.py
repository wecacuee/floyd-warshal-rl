from argparse import Action, ArgumentParser
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

from cog.confutils import (Conf, dict_update_recursive, 
                           func_kwonlydefaults, makeconf,
                           ConfFromDotArgs, MEMOIZE_METHOD)

from cog.memoize import LambdaMethodMemoizer
from cog.misc import ensuredirs, git_revision, pairwise

from alg.floyd_warshall_grid import (FloydWarshallAlgDiscrete,
                                     FloydWarshallVisualizer,
                                     FloydWarshallLogger)
from alg.qlearning import QLearningDiscrete, QLearningVis, QLearningLogger
from alg.qlearning import (post_process as ql_post_process,
                           post_process_data_iter as ql_post_process_data_iter,
                           post_process_data_tag as ql_post_process_data_tag)

from alg.floyd_warshall_grid import (post_process as fw_post_process,
                                     post_process_data_iter as fw_post_process_data_iter,
                                     post_process_data_tag as fw_post_process_data_tag)

from game.metrics import (ComputeMetricsFromLogReplay,
                          LatencyObserver, DistineffObs)
from game.play import (LoggingObserver, MultiObserver, play,
                       NoOPObserver, LogFileWriter, NPJSONEncDec,
                       JSONLoggingFormatter, LogFileReader)
from prob.windy_grid_world import (WindyGridWorld, AgentInGridWorld,
                                   random_goal_pose_gen,
                                   random_start_pose_gen,
                                   maze_from_filepath,
                                   Act2DSpace, Loc2DSpace,
                                   DrawAgentGridWorldFromLogs)
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


def WindyGridWorldConf():
    return makeconf(WindyGridWorld)


def find_latest_file(dir_):
    p_stats = [(p, p.stat()) for p in Path(dir_).iterdir() if p.is_file()]
    return max(p_stats, key = lambda p_stat: p_stat[1].st_mtime)[0]


def LogFileConf():
    return Conf(
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
        log_file_latest = lambda s : find_latest_file(s.log_file_dir),
        retval = lambda s : s
    ),
    attrs = dict(
        __name__                 = "LogFileConf",
        exp_name_template        = "{self.run_month}_{self.gitrev}_{self.confname}",
        log_file_dir_template    = "{self.data_dir}/{self.project_name}/{self.exp_name}",
        log_file_template        = "{self.log_file_dir}/{self.run_time}.log",
        run_full_time_format     = "%Y%m%d-%H%M%S",
        #project_name             = PROJECT_NAME,
        #confname                 = "LogFileConf",
    ),
)


Obs2DSpaceConf        = lambda : makeconf(Loc2DSpace)
Act2DSpaceConf        = lambda : makeconf(Act2DSpace)
AgentInGridWorldConf  = lambda : makeconf(AgentInGridWorld)
QLearningDiscreteConf = lambda : makeconf(QLearningDiscrete)
LoggingEncdecConf     = lambda : makeconf(NPJSONEncDec)


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
    

def LoggerFactoryConf():
    return makeconf(logger_factory)

def LoggingObserverConf():
    return makeconf(
        LoggingObserver,
        props = dict(
            logger = lambda s : s.logger_factory("LoggingObserver")
        ),
    )

def LatencyObserverConf():
    return makeconf(LatencyObserver)

def DistineffObsConf():
    return makeconf(DistineffObs)

def LogFileReaderConf():
    return makeconf(
        LogFileReader,
        props = dict(
            logfilepath    = lambda s : s.log_file_conf.log_file_latest,
            enc            = lambda s : s.logging_encdec,
        ))

def ComputeMetricsFromLogReplayConf():
    return makeconf(
        ComputeMetricsFromLogReplay,
        props = dict(
            loggingobserver = lambda s: s.logging_observer,
            metrics_observers = lambda s : [s.latency_observer,
                                            s.distineff_observer],
            logfilereader   = lambda s: s.log_file_reader
        ))


def VisualizerConf():
    return makeconf(
        NoOPObserver,
        props = dict(
            logger = lambda s : s.logger_factory("Visualizer")
        ),
        log_interval = 1,
        cellsize = 80)


def MultiObserverConf():
    return makeconf(
        MultiObserver,
        props = dict(
            observers = lambda s: dict(
                logging = s.logging_observer,
                metrics = s.metrics_observer,
                visualizer = s.visualizer_observer)
        ))

def QLearningLoggerConf():
    return makeconf(
        QLearningLogger,
        props = dict(
            logger = lambda self: self.logger_factory("QLearningLogger"),
        ))


def QLearningPlayConf():
    return makeconf(
        play,
        props = dict(
            alg = QLearningDiscreteConf(),
            visualizer_observer = QLearningLoggerConf(),
        ))

def FloydWarshallAlgDiscreteConf():
    return makeconf(
        FloydWarshallAlgDiscrete,
        props = dict(
            qlearning = QLearningDiscreteConf(),
        ))


def FloydWarshallLoggerConf():
    return makeconf(
        FloydWarshallLogger,
        props = dict(
            logger = lambda self: self.logger_factory("FloydWarshallLogger"),
        ))

def FloydWarshallPlayConf():
    return makeconf(
        play,
        props = dict(
            alg = FloydWarshallAlgDiscreteConf(),
            visualizer_observer = FloydWarshallLoggerConf(),
        ))

def SessionConf(confname,
                props = dict(
                    log_file_conf       = MEMOIZE_METHOD(LogFileConf()),
                    logging_encdec      = LoggingEncdecConf(),
                    set_logger_conf     = lambda s : setLoggerConfig(s.log_file_conf.confname,
                                                                     s.log_file_conf.log_file,
                                                                     s.logging_encdec),
                    logger_factory      = MEMOIZE_METHOD(LoggerFactoryConf()),
                    rng                 = LambdaMethodMemoizer("rng")(
                        lambda s: np.random.RandomState(seed=0)),
                    maze                = lambda s: maze_from_filepath(
                        s.maze_file_path_template.format(self = s)),
                    file_work_dir       = lambda s: Path(__file__).parent,
                    log_file_dir        = lambda s: s.log_file_conf.log_file_dir,
                ),
                attrs = dict(
                    maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
                )
):
    return Conf(props = props, attrs = dict(attrs, confname = confname))

def QLearningPlaySessionConf(
        confname,
        props = dict(
            grid_world          = MEMOIZE_METHOD(WindyGridWorldConf()),
            upper_bound         = lambda s: np.array(s.grid_world.shape),
            action_space        = Act2DSpaceConf(),
            observation_space   = Obs2DSpaceConf(),
            prob                = MEMOIZE_METHOD(AgentInGridWorldConf()),
            logging_observer    = LoggingObserverConf(),
            metrics_observer    = ComputeMetricsFromLogReplayConf(),
            latency_observer    = LatencyObserverConf(),
            distineff_observer  = DistineffObsConf(),
            log_file_reader     = LogFileReaderConf(),
            observer            = MultiObserverConf(),
            play                = QLearningPlayConf(),
            visualizer_observer = QLearningLoggerConf(),
        ),
        attrs = dict(
            seed                    = 1,
            project_name            = PROJECT_NAME,
            lower_bound             = np.array([0, 0]),
            nepisodes               = 3,
            log_interval            = 1,
        ),
        retkey = "play",
    ):
    return Conf(props = props, attrs = dict(attrs), retkey=retkey,
                fallback = SessionConf(confname))

def ql_play(alg = QLearningDiscrete,
            prob = AgentInGridWorld,
            observer = MultiObserver,
            nepisodes = 3):
    return play(alg, prob, observer, nepisodes, logger_factory)
            

def FloydWarshallPlaySessionConf(confname):
    return Conf(
        retkey = "play",
        props = dict(
            play                = FloydWarshallPlayConf(),
            visualizer_observer = FloydWarshallLoggerConf(),
            observer            = MultiObserverConf(),
        ),
        fallback = QLearningPlaySessionConf(confname))
    

def multiplay(plays):
    return { k : v() for k, v in plays.items() } 

def MultiPlaySessionConf():
    return makeconf(
        multiplay,
        plays = dict(
            ql = QLearningPlaySessionConf(confname="QLearning"),
            fw = FloydWarshallPlaySessionConf(confname="FloydWarshall")
        ))

def PostProcDataTagConf(post_process_data_tag_func):
    return makeconf(
        lambda cellsize, image_file_fmt: functools.partial(
            post_process_data_tag_func, cellsize=cellsize, image_file_fmt=image_file_fmt),
        props = dict(
            log_file_dir = lambda self: self.log_file_conf.log_file_dir,
            image_file_fmt = lambda self: self.image_file_fmt_template.format(self=self),
        ),
        image_file_fmt_template = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
        cellsize = 100)

def LogFileReaderPostProcConf():
    return makeconf(
        LogFileReader,
        enc  = NPJSONEncDec())

def PostProcDataIterConf(post_process_iter_func, filter_tag):
    return makeconf(
        lambda log_file_reader, filter_criteria: functools.partial(
            post_process_iter_func, log_file_reader=log_file_reader,
            filter_criteria=filter_criteria),
        props = dict(
            log_file_reader = LogFileReaderPostProcConf(),
            #logfilepath    = "/tmp/need/some/log/file.path",
        ),
        filter_criteria = dict(tag = filter_tag))
                                              

def FloydWarshallPostProcessSessionConf():
    return makeconf(
        fw_post_process,
        props = dict(
            process_data_tag = PostProcDataTagConf(fw_post_process_data_tag),
            log_file_conf    = MEMOIZE_METHOD(LogFileConf()),
            data_iter = PostProcDataIterConf(
                fw_post_process_data_iter,
                'FloydWarshallLogger:action_value'),
        ),
        project_name     = PROJECT_NAME,
        confname         = "FWPostProc",
    )

def QLearningPostProcSessionConf():
    return makeconf(
        ql_post_process,
        props = dict(
            process_data_tag = PostProcDataTagConf(ql_post_process_data_tag),
            log_file_conf    = MEMOIZE_METHOD(LogFileConf()),
            data_iter        = PostProcDataIterConf(
                ql_post_process_data_iter,
                'QLearningLogger:action_value'),
        ),
        project_name     = PROJECT_NAME,
        confname         = "QLPostProc",
    )

def AgentVisSessionConf():
    return makeconf(
        ql_post_process,
        props = dict(
            windy_grid_world = WindyGridWorldConf(),
            process_data_tag = PostProcDataTagConf(None).copy(
                retfunc = DrawAgentGridWorldFromLogs().partial),
            log_file_conf    = MEMOIZE_METHOD(LogFileConf()),
            rng              = LambdaMethodMemoizer("rng")(
                lambda s: np.random.RandomState(seed=0)),
            data_iter        = PostProcDataIterConf(
                ql_post_process_data_iter,
                ['LoggingObserver:new_episode', 'LoggingObserver:new_step']),
            maze                = lambda s: maze_from_filepath(
                s.maze_file_path_template.format(self = s)),
            file_work_dir       = lambda s: Path(__file__).parent,
        ),
        maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
        project_name     = PROJECT_NAME,
        confname         = "AgentVis",
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=lambda s: globals()[s],
                        default=MultiPlaySessionConf())
    a, remargs = parser.parse_known_args()
    a.config()
    #import sys
    #c = FloydWarshallPostProcessConf(confname="FloydWarshall")
    #conf = ConfFromDotArgs(c).from_args(sys.argv[1:])
    #conf()
