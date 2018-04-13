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

from cog.confutils import (WFuncFB, KWProp as KWP,
                           KWFuncExp,
                           props2attrs)

from cog.memoize import LambdaMethodMemoizer, MEMOIZE_METHOD
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


def find_latest_file(dir_):
    if not Path(dir_).exists():
        return None
    p_stats = [(p, p.stat()) for p in Path(dir_).iterdir() if p.is_file()]
    return max(p_stats, key = lambda p_stat: p_stat[1].st_mtime)[0]


def setLoggerConfig(confname, log_file, logging_encdec):
    print("Setting dict config from {confname}".format(confname=confname))
    logging.root = logging.RootLogger(logging.WARNING)
    logging.Logger.root = logging.root
    logging.Logger.manager = logging.Manager(logging.Logger.root)
    logging.config.dictConfig(
        logging_dictConfig(log_file,
                           logging_encdec))


def logger_factory(s):
    _ = s.set_logger_conf
    return (lambda name : logging.getLogger(name))
    

def LogFileConf():
    return WFuncFB(
        lambda  log_file_conf: log_file_conf,
        log_file_conf = KWP(lambda self: self),
        log_file = KWP(lambda self: ensuredirs(
            self.log_file_template.format(self=self))),
        log_file_dir = KWP(lambda self: ensuredirs(
            self.log_file_dir_template.format(self=self))),
        data_dir     = KWP(lambda self: os.environ["MID_DIR"]),
        exp_name     = KWP(lambda self: self.exp_name_template.format(self=self)),
        gitrev       = KWP(LambdaMethodMemoizer(
            "gitrev")(lambda self: git_revision(Path(__file__).parent))),
        run_month    = KWP(lambda self: self.run_full_time[:6]),
        run_time     = KWP(lambda self: self.run_full_time[6:]),
        run_full_time     = KWP(LambdaMethodMemoizer(
            "run_full_time") (lambda self: time.strftime(self.run_full_time_format))),
        log_file_latest = KWP(lambda s : find_latest_file(s.log_file_dir)),
        set_logger_conf = KWP(lambda s: setLoggerConfig(
            s.confname, s.log_file, s.logging_encdec)),
        logger_factory  = KWP(MEMOIZE_METHOD(logger_factory)),
        __name__                 = "LogFileConf",
        logging_encdec           = NPJSONEncDec(),
        exp_name_template        = "{self.run_month}_{self.gitrev}_{self.confname}",
        log_file_dir_template    = "{self.data_dir}/{self.project_name}/{self.exp_name}",
        log_file_template        = "{self.log_file_dir}/{self.run_time}.log",
        run_full_time_format     = "%Y%m%d-%H%M%S",
    ).expects("project_name confname".split())

def AgentInGridWorldConf():
    return WFuncFB(AgentInGridWorld.from_maze_file_path).expects(
        "rng maze_file_path".split())

def QLearningDiscreteConf():
    return WFuncFB(QLearningDiscrete).expects(
                     ["rng", "action_space", "observation_space"])

def LoggingObserverConf():
    return WFuncFB(
        LoggingObserver,
        logger = KWP(lambda s : s.logger_factory("LoggingObserver"))
    ).expects(["logger_factory"])

def LatencyObserverConf():
    return WFuncFB(LatencyObserver).expects(["prob"])

def DistineffObsConf():
    return WFuncFB(DistineffObs).expects(["prob"])

def LogFileReaderConf(
        props = dict(
            logfilepath    = lambda s : s.log_file_conf.log_file_latest,
        )
):
    return WFuncFB(
        LogFileReader,
        enc = NPJSONEncDec(),
        **props2attrs(props)
    ).expects("log_file_conf".split())

def ComputeMetricsFromLogReplayConf(
        props = dict(
            loggingobserver = lambda s: s.logging_observer,
            metrics_observers = lambda s : [s.latency_observer,
                                            s.distineff_observer],
            logfilereader   = lambda s: s.log_file_reader
        )
):
    return WFuncFB(
        ComputeMetricsFromLogReplay,
        **props2attrs(props)
    ).expects(
        "logging_observer latency_observer distineff_observer log_file_reader".split())


def VisualizerConf(
        props = dict(
            logger = lambda s : s.logger_factory("Visualizer")
        ),
):
    return WFuncFB(
        NoOPObserver,
        log_interval = 1,
        cellsize = 80,
        **props2attrs(props)
    ).expects(["logger_factory"])


def MultiObserverConf(
        props = dict(
            observers = lambda s: dict(
                logging = s.logging_observer,
                metrics = s.metrics_observer,
                visualizer = s.visualizer_observer)
        )
):
    return WFuncFB(
        MultiObserver,
        **props2attrs(props)
    ).expects(
        "logging_observer metrics_observer visualizer_observer".split())

def QLearningLoggerConf(
        props = dict(
            logger = lambda self: self.logger_factory("QLearningLogger"),
        )
):
    return WFuncFB(
        QLearningLogger,
        log_interval = 1,
        **props2attrs(props)
    ).expects(
        ["logger_factory"])


def FloydWarshallAlgDiscreteConf():
    return WFuncFB(
        FloydWarshallAlgDiscrete,
        qlearning = KWP(QLearningDiscreteConf()),
    ).expects(["rng", "action_space", "observation_space"])


def FloydWarshallLoggerConf():
    return WFuncFB(
        FloydWarshallLogger,
        logger = KWP(lambda self: self.logger_factory("FloydWarshallLogger")),
        log_interval = 1
        ).expects(["logger_factory"])

def SessionConf(confname,
                props = lambda : dict(
                    log_file_conf       = MEMOIZE_METHOD(LogFileConf()),
                    logger_factory      = lambda s: s.log_file_conf.logger_factory,
                    rng                 = LambdaMethodMemoizer("rng")(
                        lambda s: np.random.RandomState(seed=0)),
                    maze_file_path      = lambda s: s.maze_file_path_template.format(
                        self = s),
                    file_work_dir       = lambda s: Path(__file__).parent,
                    log_file_dir        = lambda s: s.log_file_conf.log_file_dir,
                ),
                attrs = dict(
                    project_name            = PROJECT_NAME,
                    maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
                    seed                    = 1,
                )
):
    attrs = dict(props2attrs(props()), confname = confname, **attrs)
    return WFuncFB(lambda s : s, **attrs)

def QLearningPlaySessionConf(
        confname,
        props = lambda : dict(
            action_space        = lambda s : s.prob.action_space,
            observation_space   = lambda s : s.prob.observation_space,
            prob                = MEMOIZE_METHOD(AgentInGridWorldConf()),
            logging_observer    = LoggingObserverConf(),
            metrics_observer    = ComputeMetricsFromLogReplayConf(),
            latency_observer    = LatencyObserverConf(),
            distineff_observer  = DistineffObsConf(),
            log_file_reader     = LogFileReaderConf(),
            observer            = MultiObserverConf(),
            visualizer_observer = QLearningLoggerConf(),
            alg                 = QLearningDiscreteConf(),
        ),
        attrs = dict(
            nepisodes               = 3,
            log_interval            = 1,
        ),
    ):
    attrs = dict(props2attrs(props()), **attrs)
    wplay = WFuncFB(play, **attrs).expects(
        "maze_file_path log_file_conf logger_factory rng file_work_dir".split())
    wplay.fb = SessionConf(confname)
    return wplay
            

def FloydWarshallPlaySessionConf(confname,
        props = lambda : dict(
            action_space        = lambda s : s.prob.action_space,
            observation_space   = lambda s : s.prob.observation_space,
            prob                = MEMOIZE_METHOD(AgentInGridWorldConf()),
            logging_observer    = LoggingObserverConf(),
            metrics_observer    = ComputeMetricsFromLogReplayConf(),
            latency_observer    = LatencyObserverConf(),
            distineff_observer  = DistineffObsConf(),
            log_file_reader     = LogFileReaderConf(),
            observer            = MultiObserverConf(),
            alg                 = FloydWarshallAlgDiscreteConf(),
            visualizer_observer = FloydWarshallLoggerConf(),
        ),
        attrs = dict(
            nepisodes               = 3,
            log_interval            = 1,
        ),
):
    attrs = dict(props2attrs(props()), **attrs)
    wplay = WFuncFB(play, **attrs).expects(
        "maze_file_path log_file_conf logger_factory rng file_work_dir".split())
    wplay.fb = SessionConf(confname)
    return wplay

def multiplay(plays):
    return { k : v() for k, v in plays.items() } 

def MultiPlaySessionConf():
    return WFuncFB(
        multiplay,
        plays = dict(
            ql = QLearningPlaySessionConf(confname="QLearning"),
            fw = FloydWarshallPlaySessionConf(confname="FloydWarshall")
        ))

def PostProcDataTagConf(post_process_data_tag_func,
        props = dict(
            log_file_dir = lambda self: self.log_file_conf.log_file_dir,
            image_file_fmt = lambda self: self.image_file_fmt_template.format(self=self),
        ),
):
    return WFuncFB(
        lambda cellsize, image_file_fmt: functools.partial(
            post_process_data_tag_func, cellsize=cellsize, image_file_fmt=image_file_fmt),
        image_file_fmt_template = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
        cellsize = 100,
        **props2attrs(props)
    )

def LogFileReaderPostProcConf():
    return WFuncFB(
        LogFileReader,
        enc  = NPJSONEncDec())

def PostProcDataIterConf(post_process_iter_func, filter_tag,
        props = dict(
            log_file_reader = LogFileReaderPostProcConf(),
            #logfilepath    = "/tmp/need/some/log/file.path",
        ),
):
    return WFuncFB(
        lambda log_file_reader, filter_criteria: functools.partial(
            post_process_iter_func, log_file_reader=log_file_reader,
            filter_criteria=filter_criteria),
        filter_criteria = dict(tag = filter_tag),
        **props2attrs(props)
    )
                                              

def FloydWarshallPostProcessSessionConf(
        props = dict(
            process_data_tag = PostProcDataTagConf(fw_post_process_data_tag),
            log_file_conf    = MEMOIZE_METHOD(LogFileConf()),
            data_iter = PostProcDataIterConf(
                fw_post_process_data_iter,
                'FloydWarshallLogger:action_value'),
        ),
):
    return WFuncFB(
        fw_post_process,
        project_name     = PROJECT_NAME,
        confname         = "FWPostProc",
        **props2attrs(props)
    )

def QLearningPostProcSessionConf(
        props = dict(
            process_data_tag = PostProcDataTagConf(ql_post_process_data_tag),
            log_file_conf    = MEMOIZE_METHOD(LogFileConf()),
            data_iter        = PostProcDataIterConf(
                ql_post_process_data_iter,
                'QLearningLogger:action_value'),
        )):
    return WFuncFB(
        ql_post_process,
        project_name     = PROJECT_NAME,
        confname         = "QLPostProc",
        **props2attrs(props)
    )

def AgentVisSessionConf(
        props = dict(
            windy_grid_world = WFuncFB(
                WindyGridWorld.from_maze_file_path).expects(
                    "rng maze_file_path".split()),
            process_data_tag = PostProcDataTagConf(None).copy(
                func = DrawAgentGridWorldFromLogs().partial),
            log_file_conf    = MEMOIZE_METHOD(LogFileConf()),
            rng              = LambdaMethodMemoizer("rng")(
                lambda s: np.random.RandomState(seed=0)),
            data_iter        = PostProcDataIterConf(
                ql_post_process_data_iter,
                ['LoggingObserver:new_episode', 'LoggingObserver:new_step']),
            maze_file_path      = lambda s: s.maze_file_path_template.format(self = s),
            file_work_dir       = lambda s: Path(__file__).parent,
        ),
):
    return WFuncFB(
        ql_post_process,
        maze_file_path_template = "{self.file_work_dir}/maze_5x5_no_wind.txt",
        project_name     = PROJECT_NAME,
        confname         = "AgentVis",
        **props2attrs(props)
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
