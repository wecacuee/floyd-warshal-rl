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

from cog.confutils import (Conf, dict_update_recursive, ConfTemplate)

from cog.memoize import MEMOIZE_METHOD, LambdaMethodMemoizer
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


def SeedSourceGen(parent):
    return Conf(
        props = dict(
            rng = LambdaMethodMemoizer(
                "rng")(
                    lambda self: np.random.RandomState(self.seed)),
            next_seed = lambda self: self.rng.randint(1000)
        ),
        from_parent = ("seed",),
        parent = parent,
    )


def WindyGridWorldConf(parent):
    return Conf(
        props = dict(
            seed = lambda s: s.seed_src.next_seed,
            maze_file_path = lambda s: s.maze_file_path_template.expand(s),
            file_work_dir = lambda s: Path(__file__).parent,
            maze = lambda s: maze_from_filepath(s.maze_file_path),
        ),
        attrs = dict(
            _call_func    = WindyGridWorld,
            maze_file_path_template =  ConfTemplate("{file_work_dir}/maze_5x5_no_wind.txt"),
            wind_strength = 0.1
            #seed_src      = SeedSource,
        ),
        from_parent = ("seed_src",),
        parent = parent
        
    )


def LogFileConf(parent):
    return Conf(
        props = dict(
        log_file = lambda self: ensuredirs(
            self.log_file_template.expand(self)),
        log_file_dir = lambda self: ensuredirs(
            self.log_file_dir_template.expand(self)),
        data_dir     = lambda self: os.environ["MID_DIR"],
        exp_name     = lambda self: self.exp_name_template.expand(self),
        gitrev       = LambdaMethodMemoizer(
            "gitrev")(
                lambda self: git_revision(Path(__file__).parent)),
        run_month    = lambda self: self.run_time[:6],
        run_time     = LambdaMethodMemoizer(
            "run_time")(
                lambda self: time.strftime(self.run_time_format)),
        image_file_fmt = lambda self: self.image_file_fmt_template.expand(self)
        ),
        attrs = dict(
            exp_name_template        = ConfTemplate("{run_month}_{gitrev}_{confname}"),
            log_file_dir_template    = ConfTemplate("{data_dir}/{project_name}/{exp_name}"),
            log_file_template        = ConfTemplate("{log_file_dir}/{run_time}.log"),
            run_time_format          = "%Y%m%d-%H%M%S",
            image_file_fmt_template  = ConfTemplate(
                "{log_file_dir}/{{tag}}_{{episode}}_{{step}}.png"),
            #project_name             = PROJECT_NAME,
            #confname                 = "LogFileConf",
        ),
        from_parent = ("project_name", "confname"),
        parent = parent,
    )


def Act2DSpaceConf(parent):
    return Conf(
        props = dict(
            seed = lambda s:s.seed_src.next_seed
        ),
        attrs = dict(
            _call_func = Act2DSpace,
            #seed_src = SeedSource
        ),
        from_parent = ("seed_src",),
        parent = parent
    )


def Obs2DSpaceConf(parent):
    return Conf(
        props = dict(
            seed = lambda s: s.seed_src.next_seed,
            upper_bound = lambda s: np.array(s.grid_world.shape),
            grid_world = lambda s: s.grid_world_conf(),
        ), attrs = dict(
            _call_func = Loc2DSpace,
            lower_bound = np.array([0,0]),
            #grid_world_conf = WindyGridWorldConf,
            #seed_src = SeedSource
        ),
        from_parent = ("seed_src", "grid_world_conf"),
        parent = parent,
    )


def AgentInGridWorldConf(parent):
    return Conf(
        props = dict(
            seed         = lambda s:s.seed_src.next_seed,
            grid_world   = lambda s:s.grid_world_conf(),
            log_file_dir = lambda s:s.log_file_conf.log_file_dir,
            action_space = lambda s:s.action_space_conf(),
            observation_space = lambda s:s.observation_space_conf(),
            grid_world_conf = WindyGridWorldConf,
            action_space_conf = Act2DSpaceConf,
            observation_space_conf = Obs2DSpaceConf,
        ), attrs = dict(
            _call_func      = AgentInGridWorld,
            #log_file_conf   = LogFileConf,
            #seed_src        = SeedSource,
            start_pose_gen  = random_start_pose_gen,
            goal_pose_gen   = random_goal_pose_gen,
            goal_reward     = 10,
            max_steps       = 200,
            wall_penality   = 1.0,
            no_render       = False
        ),
        from_parent = ("seed_src", "log_file_conf"),
        parent = parent,
    )


def QLearningDiscreteConf(parent):
    return Conf(
        props = dict(
            seed              = lambda s: s.seed_src.next_seed
        ), attrs = dict (
            _call_func            = QLearningDiscrete,
            egreedy_epsilon       = 0.05,
            action_value_momentum = 0.1, # Low momentum changes more frequently
            init_value            =   1,
            discount              = 0.99,
            #seed_src              = SeedSource,
        ),
        from_parent = ("seed_src", "action_space", "observation_space"),
        parent = parent
    )


LoggingEncdecConf = Conf(attrs = dict(
    _call_func = NPJSONEncDec))


def setLoggerConfig(confname, log_file, logging_encdec):
    print("Setting dict config from {confname}".format(confname=confname))
    logging.root = logging.RootLogger(logging.WARNING)
    logging.Logger.root = logging.root
    logging.Logger.manager = logging.Manager(logging.Logger.root)
    logging.config.dictConfig(
        logging_dictConfig(log_file,
                           logging_encdec))


def LoggingFactoryConf(parent):
    return Conf(
        props = dict(
            set_logger_conf = lambda s : setLoggerConfig(s.log_file_conf.confname,
                                                         s.log_file_conf.log_file,
                                                         s.logging_encdec_conf()),
        ), attrs = dict(
            _call_func = lambda set_logger_conf: (lambda name : logging.getLogger(name)),
            #log_file_conf = LogFileConf,
            logging_encdec_conf = LoggingEncdecConf,
        ),
        from_parent = ("log_file_conf",),
        parent = parent
    )


def LoggingObserverConf(parent):
    return Conf(
        props = dict(
            logger          = lambda s : s.logger_factory("LoggingObserver"),
        ), attrs = dict (
            _call_func   = LoggingObserver,
            #prob_conf    = AgentInGridWorldConf,
            #logger_factory_conf = LoggingFactoryConf
            log_interval = 1
        ),
        from_parent = ("logger_factory", "prob"),
        parent = parent,
    )


def LatencyObserverConf(parent):
    return Conf(
        attrs = dict(
            _call_func = LatencyObserver,
        ),
        from_parent = ("prob",),
        parent = parent,
    )


def DistineffObsConf(parent):
    return Conf(
        attrs = dict(
            _call_func = DistineffObs,
            #prob_conf = AgentInGridWorldConf
        ),
        from_parent = ("prob", ),
        parent = parent,
    )


def LogFileReaderConf(parent):
    return Conf(
        props = dict(
            logfilepath    = lambda s : s.log_file_conf.log_file,
            enc            = lambda s: s.logging_encdec_conf(),
        ), attrs = dict(
            _call_func = LogFileReader,
            #log_file_conf = LogFileConf,
            logging_encdec_conf = LoggingEncdecConf,
            sep = "\t"
        ),
        from_parent = ("log_file_conf",),
        parent = parent,
    )


def ComputeMetricsFromLogReplayConf(parent):
    return Conf(
        props = dict(
            loggingobserver   = lambda s : s.logging_observer_conf(),
            metrics_observers = lambda s : [s.latency_observer_conf(),
                                            s.distineff_observer_conf()],
            logfilereader     = lambda s : s.log_file_reader_conf()
        ), attrs = dict(
            _call_func              = ComputeMetricsFromLogReplay,
            log_file_reader_conf = LogFileReaderConf(parent),
            logging_observer_conf = LoggingObserverConf(parent),
            latency_observer_conf = LatencyObserverConf(parent),
            distineff_observer_conf = DistineffObsConf(parent)
        ),
        from_parent = ("seed_src", "logger_factory", "project_name",
                       "confname", "log_file_conf", "prob"),
        parent = parent
    )


def VisualizerConf(parent):
    return Conf(
        props = dict(
            logger = lambda s : s.logger_factory("Visualizer")
        ),
        attrs = dict(
            _call_func = NoOPObserver,
            log_interval = 1,
            cellsize = 80,
        ),
        from_parent = ("logger_factory",),
        parent = parent
    )

def MultiObserverConf(parent):
    return Conf( 
        props = dict(
            observers = lambda s: {k : v() for k, v in s.observers_conf.items()}
        ),
        attrs = dict (
            _call_func = MultiObserver,
            observers_conf = dict(
                #logging = LoggingObserverConf(parent),
                metrics = ComputeMetricsFromLogReplayConf(parent),
                visualizer = VisualizerConf(parent))
        ),
        from_parent = ("seed_src", "logger_factory", "project_name",
                       "confname", "log_file_conf", "prob"),
        parent = parent,
    )


def QLearningPlayConf(parent):
    return Conf(
        props = dict(
            alg             = lambda self: self.alg_conf(),
            prob            = lambda self: self.prob_conf(),
            observer        = lambda self: self.observers_conf(),
            action_space    = lambda self: self.prob.action_space,
            observation_space = lambda self: self.prob.observation_space,
            alg_conf       = QLearningDiscreteConf,
            prob_conf      = AgentInGridWorldConf,
            observers_conf = lambda parent: MultiObserverConf(parent).copy(
                attrs = dict(
                    observers_conf = dict(
                        visualizer = Conf(
                            attrs = dict(_call_func = QLearningLogger))))),
        ), attrs = dict (
            _call_func     = play,
            nepisodes            = 3
        ),
        from_parent = ("seed_src", "logger_factory", "project_name",
                       "confname", "log_file_conf"),
        parent = parent,
    )

def FloydWarshallAlgDiscreteConf(parent):
    return Conf(
        props = dict(
            qlearning = lambda s : s.qlearning_conf()
        ), attrs = dict(
            _call_func = FloydWarshallAlgDiscrete,
            qlearning_conf = QLearningDiscreteConf(parent),
        ),
        from_parent = ("seed_src", "action_space", "observation_space"),
        parent = parent
    )


def FloydWarshallPlayConf(parent):
    return Conf(
        props = dict(
            alg             = lambda self: self.alg_conf(),
            prob            = lambda self: self.prob_conf(),
            observer        = lambda self: self.observers_conf(),
            action_space    = lambda self: self.prob.action_space,
            observation_space = lambda self: self.prob.observation_space,
            alg_conf             = FloydWarshallAlgDiscreteConf,
            prob_conf            = AgentInGridWorldConf,
            observers_conf       = lambda self: MultiObserverConf(self).copy(
                attrs = dict(
                    observers_conf = dict(
                        visualizer = Conf(
                            attrs = dict(_call_func = FloydWarshallLogger))))),
        ), attrs = dict (
            _call_func           = play,
            nepisodes            = 3,
        ),
        from_parent = ("seed_src", "logger_factory", "project_name",
                       "confname", "log_file_conf"),
        parent = parent,
    )


def SessionConf(confname, project_name = PROJECT_NAME, seed = 0):
    return Conf(
        props = dict(
            log_file_conf = MEMOIZE_METHOD(LogFileConf),
            logger_factory_conf = LoggingFactoryConf,
            seed_src = MEMOIZE_METHOD(SeedSourceGen),
            logger_factory = LambdaMethodMemoizer("logging_factory")(
                lambda s: s.logger_factory_conf()),
        ),
        attrs = dict(
            seed = seed,
            project_name = project_name,
            confname = confname,
        )
    )


def SessionedConf(conf_gen, **kw):
    return Conf(
        props = dict(
            sessioned_conf = lambda s: s.conf(SessionConf(s.conf.__name__))
        ),
        attrs = dict(
            _call_func = lambda sessioned_conf: sessioned_conf(),
            conf = conf_gen,
            **kw,
        )
    )


def QLearningPlaySessionConf(**kw):
    return QLearningPlayConf(SessionConf(QLearningPlayConf.__name__)).copy(
        attrs = kw)

    
def FloydWarshallPlaySessionConf(**kw):
    return FloydWarshallPlayConf(SessionConf(FloydWarshallPlayConf.__name__)).copy(
        attrs = kw)
    

def MultiPlaySessionConf(**kw):
    return Conf(
        attrs = dict(
            _call_func = lambda multiconf : [c() for c in multiconf],
            multiconf = [FloydWarshallPlaySessionConf(**kw),
                         QLearningPlaySessionConf(**kw)],
        ),
    )


if __name__ == '__main__':
    import sys
    conf = Conf.parse_all_args("conf.default:MultiPlaySessionConf",
                               sys.argv[1:], glbls=globals())
    conf()
