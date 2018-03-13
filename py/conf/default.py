"""
Philosophy:
Convert parts of the program into recursive named value pairs. The
values are usually objects. We explicitly avoid sequence of actions as
a model because that makes the replacing of a particular component in
the hierarchy difficult by merging.

The configuration must be lazy. This is to avoid early copying of
shared arguments before the command line parsing has taken place.

Example: log_file_name is usually shared among many objects. Most of
the classes or functions expect log_file_name (or most arguments) as
primitive types rather than functions or lazy configurations.
Conf objects act as thin wrappers over classes or functions that have
most of their arguments as lazy functions/properties that depend upon
a single shared primitive data source.

def foo(a, b, c):
    ...

# Prepare configuration
fooconf = Conf(a=1, b=2, c=3, _call_func = foo)

# Run the function
fooconf()

"""
from argparse import Action
import shlex
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
                           format_string_from_conf, apply_conf,
                           serialize_any, LazyApplyable, MultiConfGen)

from cog.memoize import MEMOIZE_METHOD
from cog.misc import ensuredirs, git_revision, pairwise

from alg.floyd_warshall_grid import (FloydWarshallAlgDiscrete,
                                     FloydWarshallVisualizer,
                                     FloydWarshallLogger)
from alg.qlearning import QLearningDiscrete, QLearningVis, QLearningLogger
from game.metrics import (ComputeMetricsFromLogReplay,
                          LatencyObserver, DistineffObs)
from game.play import (LoggingObserver, MultiObserver, play,
                       NoOPObserver, LogFileWriter, NPJSONEncDec,
                       JSONLoggingFormatter, LogFileReader)
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

class AgentInGridWorldConf(Conf):
    seed =  property(lambda s:s.root_conf._next_seed())
    grid_world = property(lambda s:s.root_conf.grid_world)
    log_file_dir = property(lambda s:s.root_conf.log_file_dir)

    def defaults(self):
        return dict_update_recursive(
            super().defaults(),
            dict(_call_func     = AgentInGridWorld,
                 start_pose_gen = random_start_pose_gen,
                 goal_pose_gen  = random_goal_pose_gen,
                 goal_reward    = 10,
                 max_steps      = 200,
                 wall_penality  = 1.0,
                 no_render      = False))


class QLearningDiscreteConf(Conf):
    action_space = property(lambda s: s.root_conf.action_space)
    observation_space = property(lambda s: s.root_conf.observation_space)
    seed = property(lambda s: s.root_conf._next_seed())

    def defaults(self):
        return dict_update_recursive(
            super().defaults(),
            dict(
                _call_func            = QLearningDiscrete,
                egreedy_epsilon       = 0.05,
                action_value_momentum = 0.1, # Low momentum changes more frequently
                init_value            =   1,
                discount              = 0.99
            ))

class WindyGridWorldConf(Conf):
    seed = property(lambda s: s.root_conf._next_seed())
    maze = property(lambda s: maze_from_filepath(
        Path(__file__).parent / "maze_5x5_no_wind.txt"))

    def defaults(self):
        return dict_update_recursive(
            super().defaults(),
            dict(_call_func          = WindyGridWorld,
                 wind_strength = 0.1))

class LoggingObserverConf(Conf):
    prob        = property(lambda s : play_conf.prob)
    logger      = property(lambda s : play_conf.getLogger("LoggingObserver"))
    def defaults(self):
        return dict_update_recursive(
            super().defaults(),
            dict(_call_func   = LoggingObserver,
                 log_interval = 1))


class ComputeMetricsFromLogReplayConf(Conf):
    loggingobserver   = property(lambda s : play_conf.logging_observer)
    metrics_observers = property(lambda s : [play_conf.latency_observer,
                                             play_conf.distineff_observer])
    logfilereader     = property(lambda s : play_conf.log_file_reader)

    def defaults(self):
        return dict_update_recursive(
            super().defaults(),
            dict(_call_func              = ComputeMetricsFromLogReplay))


def bool_from_str(v):
    try:
        return {"False" : False, "True" : True}[v]
    except KeyError as e:
        raise ValueError(e)


def try_restrictive_types(v, converters=[bool_from_str, int, float]):
    exc = None
    for conv in converters:
        try:
            return conv(v)
        except ValueError as e:
            exc = e
    raise exc


class FilterConf(Conf): 
    def argparse_action(self, key, glbls):
        class DefAction(Action):
            def __call__(s, parser, namespace, values, option_string=None):
                d = {k[2:] : try_restrictive_types(v)
                     for k, v in pairwise(shlex.split(values))}
                setattr(namespace, key, d)
        return DefAction


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

    @MEMOIZE_METHOD
    def setLoggerConfig(self):
        print("Setting dict config from {self.__class__.__name__}".format(self=self))
        logging.root = logging.RootLogger(logging.WARNING)
        logging.Logger.root = logging.root
        logging.Logger.manager = logging.Manager(logging.Logger.root)
        logging.config.dictConfig(
            logging_dictConfig(self.log_file,
                               self.logging_encdec))

    def getLogger(self, name):
        self.setLoggerConfig()
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
        return format_string_from_conf(self.log_file_latest_template, self)

    @property
    def log_file_dir(self):
        log_dir = format_string_from_conf(self.log_file_dir_template, self)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return log_dir

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
            format_string_from_conf(self.log_file_template, self))

        if self.log_file_latest != log_file and not os.path.exists(log_file):
            with open(log_file, "a") as f: pass

            if os.path.exists(self.log_file_latest):
                os.remove(self.log_file_latest)
            os.symlink(log_file, self.log_file_latest)
        return log_file

    @property
    @MEMOIZE_METHOD
    def log_file_reader(self):
        return self.log_file_reader_conf()

    @property
    @MEMOIZE_METHOD
    def grid_world(self):
        return self.grid_world_conf()

    @property
    @MEMOIZE_METHOD
    def prob(self):
        return self.prob_conf()

    @property
    @MEMOIZE_METHOD
    def logging_observer(self):
        return self.logging_observer_conf()

    @property
    @MEMOIZE_METHOD
    def compute_metrics_from_replay(self):
        return self.compute_metrics_from_replay_conf()

    @property
    @MEMOIZE_METHOD
    def observer(self):
        return self.observer_conf()

    @property
    def observers_dict(self):
        return dict(
            logger     = self.logging_observer,
            metrics    = self.compute_metrics_from_replay,
            visualizer = self.visualizer)

    @property
    @MEMOIZE_METHOD
    def visualizer(self):
        return self.visualizer_conf()

    @property
    @MEMOIZE_METHOD
    def logging_encdec(self):
        return self.logging_encdec_conf()

    @property
    def cellsize(self):
        return self.visualizer_conf.cellsize

    @property
    def image_file_fmt(self):
        return str(Path(self.log_file_dir) / self.image_file_fmt_basename)

    @property
    def filter_criteria(self):
        return { k : v for k, v in self.filter_conf.items()
                 if not isinstance(v, (Conf, dict)) }

    def defaults(self):
        defaults      = dict(
            _call_func  = play,
            nepisodes = 3,
            seed      = 0,

            grid_world_conf = WindyGridWorldConf(self),

            prob_conf = AgentInGridWorldConf(self),

            compute_metrics_from_replay_conf =
                ComputeMetricsFromLogReplayConf(self),

            observer_conf = 
                NewConfClass("MultiObserverConf",
                             observers = lambda s : self.observers_dict,
                ) ( root_conf = self,
                    _call_func = MultiObserver,
                ),

            visualizer_conf = NewConfClass(
                "VisualizerConf",
                logger = lambda s : self.getLogger("Visualizer")
            ) ( root_conf = self,
                _call_func = NoOPObserver,
                log_interval = 1,
                cellsize = 80
            ),

            logging_encdec_conf = Conf(root_conf = self,
                                       _call_func = NPJSONEncDec),

            logging_observer_conf = LoggingObserverConf(self),

            log_file_reader_conf = NewConfClass(
                "LogFileReaderConf",
                logfilepath = lambda s : self.log_file_to_process or self.log_file,
                enc = lambda s: self.logging_encdec,
            ) ( root_conf = self,
                _call_func = LogFileReader,
                sep = "\t"
            ),

            filter_conf = FilterConf(root_conf = self),

            log_file_dir_template = "{data_dir}/{project_name}/{exp_name}",
            log_file_template = "{log_file_dir}/{run_time}.log",
            log_file_latest_template = "{log_file_dir}/latest.log",
            data_dir          = os.environ["MID_DIR"],
            project_name      = "floyd_warshall_rl",
            run_time_format   = "%Y%m%d-%H%M%S",
            log_file_to_process  = None,
            image_file_fmt_basename = "{tag}_{episode}_{step}.png"
        )
        return defaults


class FloydWarshallPlayConf(CommonPlayConf):
    @property
    @MEMOIZE_METHOD
    def alg(self):
        return self.alg_conf()

    def defaults(self):
        defaults = super().defaults()
        defaults = dict_update_recursive(
            defaults,
            dict(
                alg_conf = NewConfClass(
                    "FloydWarshallAlgDiscreteConf",
                    qlearning = lambda s : s.qlearning_conf()
                )(
                    _call_func = FloydWarshallAlgDiscrete,
                    qlearning_conf = QLearningDiscreteDefaultConfGen(self)),
                visualizer_conf = Conf(_call_func = FloydWarshallLogger)
            ))
        return defaults


class QLearningPlayConf(CommonPlayConf):
    @property
    @MEMOIZE_METHOD
    def alg(self):
        return self.alg_conf()

    def defaults(self):
        defaults = super().defaults()
        dict_update_recursive(
            defaults,
            dict(alg_conf = QLearningDiscreteDefaultConfGen(self),
                 visualizer_conf = Conf(_call_func = QLearningLogger)
            ))
        return defaults

class MultiPlayConf(Conf):
    def defaults(self):
        return dict_update_recursive(
            super().defaults(),
            dict(fw = FloydWarshallPlayConf(self),
                 ql = QLearningPlayConf(self)))

if __name__ == '__main__':
    import sys
    conf = Conf.parse_all_args("conf.default:MultiPlayConf",
                               sys.argv[1:], glbls=globals())
    conf()
