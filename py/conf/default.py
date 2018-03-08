import numpy as np
import hashlib
import pickle
import time
import os
from string import Formatter
import logging
import logging.config
from pathlib import Path
import functools

from cog.confutils import (Conf, dict_update_recursive, NewConfClass,
                           format_string_from_obj, apply_conf,
                           serialize_any, LazyApplyable)

from cog.memoize import MEMOIZE_METHOD
from cog.misc import ensuredirs, git_revision

from alg.floyd_warshall_grid import (FloydWarshallAlgDiscrete,
                                     FloydWarshallVisualizer,
                                     FloydWarshallLogger)
from alg.qlearning import QLearningDiscrete, QLearningVis, QLearningLogger
from game.metrics import (ComputeMetricsFromLogReplay,
                          LatencyObserver, DistineffObs)
from game.play import (LoggingObserver, MultiObserver, play,
                       multiplay, NoOPObserver, LogFileWriter,
                       NPJSONEncDec, JSONLoggingFormatter,
                       LogFileReader)
from prob.windy_grid_world import (WindyGridWorld, AgentInGridWorld,
                                   random_goal_pose_gen,
                                   random_start_pose_gen,
                                   maze_from_filepath)

def logging_dictConfig(log_file, logging_encdec):
    return dict(
        version = 1,
        formatters = dict(
            json_fmt = {
                '()' : JSONLoggingFormatter,
                'enc' : logging_encdec,
                'sep' : "\t",
                'fmt' : "%(asctime)s %(name)-15s %(message)s",
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

def AgentInGridWorldDefaultConfGen(play_conf):
    return NewConfClass(
        "AgentInGridWorldConf",
        seed         = lambda s: play_conf._next_seed(),
        grid_world   = lambda s: play_conf.grid_world,
        log_file_dir = lambda s: play_conf.log_file_dir
    ) (
        func           = AgentInGridWorld,
        start_pose_gen = random_start_pose_gen,
        goal_pose_gen  = random_goal_pose_gen,
        goal_reward    = 10,
        max_steps      = 200,
        wall_penality  = 1.0,
        no_render      = False
    )

def QLearningDiscreteDefaultConfGen(play_conf):
    return NewConfClass(
        "QLearningDiscreteConf",
        action_space      = lambda s: play_conf.prob.action_space,
        observation_space = lambda s: play_conf.prob.observation_space,
        seed              = lambda s: play_conf._next_seed(),
    )(
        func                  = QLearningDiscrete,
        egreedy_epsilon       = 0.05,
        action_value_momentum = 0.1, # Low momentum changes more frequently
        init_value            =   1,
        discount              = 0.99,
    )

def WindyGridWorldDefaultConfGen(play_conf):
    return NewConfClass(
        "WindyGridWorldConf",
        seed = lambda s: play_conf._next_seed(),
        maze = lambda s: maze_from_filepath(
            Path(__file__).parent / "maze_5x5_no_wind.txt"),
    ) (
        func          = WindyGridWorld,
        wind_strength = 0.1)

def LoggingObserverConfGen(play_conf):
    return NewConfClass(
        "LoggingObserverConf",
        prob        = lambda s : play_conf.prob,
        logger      = lambda s : play_conf.getLogger("LoggingObserver"))(
            func         = LoggingObserver,
            log_interval = 1)

def ComputeMetricsFromLogReplayConfGen(play_conf):
    return NewConfClass(
        "ComputeMetricsFromLogReplayConf",
        loggingobserver   = lambda s : play_conf.logging_observer,
        metrics_observers = lambda s : [play_conf.latency_observer,
                                        play_conf.distineff_observer],
        logfilereader     = lambda s : play_conf.log_file_reader
    )(
        func              = ComputeMetricsFromLogReplay)


class CommonPlayConf(Conf):
    """
    Follow the principle of delayed execution.

    1. Avoid executing any code untill a property of this conf is accessed.
    2. Do not make any copies of same attributes.
        Use property objects for redirection
    3. Use class stacks for additional properties on the base classes.
    """
    def __init__(self, **kw):
        super().__init__(**kw)

    def getLogger(self, name):
        logging.config.dictConfig(
            logging_dictConfig(self.log_file,
                               self.logging_encdec))
        return logging.getLogger(name)

    @property
    def logger_factory(self):
        return lambda name : self.getLogger(name)

    @property
    @MEMOIZE_METHOD
    def rng(self):
        return np.random.RandomState(self.seed)

    def _next_seed(self):
        return self.rng.randint(1000)

    @property
    @MEMOIZE_METHOD
    def exp_name(self):
        return "{run_month}_{gitrev}_{name}".format(
            name=self.__class__.__name__ ,
            gitrev=git_revision(Path(__file__).parent),
            run_month=self.run_time[:6])

    @property
    @MEMOIZE_METHOD
    def run_time(self):
        return time.strftime(self.run_time_format)

    @property
    def log_file_latest(self):
        return format_string_from_obj(self.log_file_latest_template, self)

    @property
    def log_file_dir(self):
        return format_string_from_obj(self.log_file_dir_template, self)

    @property
    def latency_observer(self):
        return LatencyObserver(self.prob)

    @property
    def distineff_observer(self):
        return DistineffObs(self.prob)

    @property
    @MEMOIZE_METHOD
    def log_file(self):
        log_file = ensuredirs(
            format_string_from_obj(self.log_file_template, self))

        if self.log_file_latest != log_file and not os.path.exists(log_file):
            with open(log_file, "a") as f: pass

            if os.path.exists(self.log_file_latest):
                os.remove(self.log_file_latest)
            os.symlink(log_file, self.log_file_latest)
        return log_file

    @property
    @MEMOIZE_METHOD
    def log_file_reader(self):
        return self.log_file_reader_conf.apply_func()

    @property
    @MEMOIZE_METHOD
    def grid_world(self):
        return self.grid_world_conf.apply_func()

    @property
    @MEMOIZE_METHOD
    def prob(self):
        return self.prob_conf.apply_func()

    @property
    @MEMOIZE_METHOD
    def logging_observer(self):
        return self.logging_observer_conf.apply_func()

    @property
    @MEMOIZE_METHOD
    def compute_metrics_from_replay(self):
        return self.compute_metrics_from_replay_conf.apply_func()

    @property
    @MEMOIZE_METHOD
    def observer(self):
        return self.observer_conf.apply_func()

    @property
    def observers_dict(self):
        return dict(
            logger     = self.logging_observer,
            metrics    = self.compute_metrics_from_replay,
            visualizer = self.visualizer)

    @property
    @MEMOIZE_METHOD
    def visualizer(self):
        return self.visualizer_conf.apply_func()

    @property
    @MEMOIZE_METHOD
    def logging_encdec(self):
        return self.logging_encdec_conf.apply_func()

    def defaults(self):
        defaults      = dict(
            func      = play,
            nepisodes = 3,
            seed      = 0,

            grid_world_conf = WindyGridWorldDefaultConfGen(self),

            prob_conf = AgentInGridWorldDefaultConfGen(self),

            logging_observer_conf = LoggingObserverConfGen(self),

            compute_metrics_from_replay_conf =
                ComputeMetricsFromLogReplayConfGen(self),

            observer_conf = 
                NewConfClass("MultiObserverConf",
                             observers = lambda s : self.observers_dict,
                ) (
                    func = MultiObserver,
                ),

            visualizer_conf = NewConfClass(
                "VisualizerConf",
                logger = lambda s : self.getLogger("Visualizer")
            ) (
                func = NoOPObserver,
                log_interval = 1,
                cellsize = 80
            ),

            logging_encdec_conf = Conf(func = NPJSONEncDec),

            log_file_reader_conf = NewConfClass(
                "LogFileReaderConf",
                logfilepath = lambda s : self.log_file,
                enc = lambda s: self.logging_encdec,
            ) (
                func = LogFileReader,
                sep = "\t"
            ),

            log_file_dir_template = "{data_dir}/{project_name}/{exp_name}",
            log_file_template = "{log_file_dir}/{run_time}.log",
            log_file_latest_template = "{log_file_dir}/latest.log",
            data_dir          = os.environ["MID_DIR"],
            project_name      = "floyd_warshall_rl",
            run_time_format   = "%Y%m%d-%H%M%S"
        )
        return defaults


class FloydWarshallPlayConf(CommonPlayConf):
    @property
    @MEMOIZE_METHOD
    def alg(self):
        return self.alg_conf.apply_func()

    def defaults(self):
        defaults = super().defaults()
        defaults = dict_update_recursive(
            defaults,
            dict(
                alg_conf = NewConfClass(
                    "FloydWarshallAlgDiscreteConf",
                    qlearning = lambda s : s.qlearning_conf.apply_func()
                )(
                    func = FloydWarshallAlgDiscrete,
                    qlearning_conf = QLearningDiscreteDefaultConfGen(self)),
                visualizer_conf = Conf(func = FloydWarshallLogger)
            ))
        return defaults


class QLearningPlayConf(CommonPlayConf):
    @property
    @MEMOIZE_METHOD
    def alg(self):
        return self.alg_conf.apply_func()

    def defaults(self):
        defaults = super().defaults()
        dict_update_recursive(
            defaults,
            dict(alg_conf = QLearningDiscreteDefaultConfGen(self),
                 visualizer_conf = Conf(func = QLearningLogger)
            ))
        return defaults

class MultiPlayConf(CommonPlayConf):
    def defaults(self):
        defaults = super().defaults()
        defaults = dict_update_recursive(
            defaults,
            dict(func = multiplay,
                 trials = [FloydWarshallPlayConf(), QLearningPlayConf()],
            ))
        return defaults


if __name__ == '__main__':
    import sys
    conf = Conf.parse_all_args("conf.default:MultiPlayConf",
                               sys.argv[1:], glbls=globals())
    conf.apply_func()
