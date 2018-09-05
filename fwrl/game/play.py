from __future__ import absolute_import, division, print_function
import os
import logging
from contextlib import closing
from functools import wraps
import numpy as np
import operator
from collections import OrderedDict
from enum import Enum

from umcog import draw
from umcog.memoize import MEMOIZE_METHOD
from umcog.confutils import extended_kwprop, KWProp as prop, xargs
from .metrics import ComputeMetricsFromLogReplay


class Space(object):
    def values():
        raise NotImplementedError()

    @property
    def size():
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def contains(self, x):
        raise NotImplementedError()

    def to_jsonable(self, sample_n):
        return sample_n

    def from_jsonable(self, sample_n):
        return sample_n


class Problem(object):
    action_space = Space()
    observation_space = Space()
    reward_range = Space()

    def step(self, action):
        raise NotImplementedError()

    def observation(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def episode_reset(self, episode_n):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        raise NotImplementedError()

    def unwrapped(self):
        raise NotImplementedError()

    def done(self):
        raise NotImplementedError()


class Alg(object):
    action_space = Space()
    observation_space = Space()
    reward_range = Space()

    def set_goal_obs(self, goal_obs):
        raise NotImplementedError()

    def update(self, obs, act, reward):
        raise NotImplementedError()

    def policy(self, obs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def episode_reset(self, episode_n):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        raise NotImplementedError()

    def unwrapped(self):
        raise NotImplementedError()

    def done(self):
        raise NotImplementedError()


class NoOPObserver(object):
    def __init__(self):
        self.prob = None
        self.alg = None
        self.parent = None

    def set_prob(self, prob):
        self.prob = prob

    def set_alg(self, alg):
        self.alg = alg

    def __getattr__(self, attr):
        if attr in """on_new_episode on_new_step on_play_end
                      on_goal_hit on_new_goal_obs
                      on_play_start on_episode_end""".split():
            return lambda *args, **kwargs : 0
        else:
            raise AttributeError("attr = {attr} not found".format(attr  = attr))


class LoggingObserver(NoOPObserver):
    @extended_kwprop
    def __init__(self,
                 logger = prop(lambda s: s.logger_factory("LoggingObserver")),
                 log_interval = 1):
        self._logger = logger
        self.log_interval = log_interval
        self.human_tag       = "INFO"
        self.goal_hit_tag    = "{self.__class__.__name__}:goal_hit".format(self=self)
        self.new_step_tag    = "{self.__class__.__name__}:new_step".format(self=self)
        self.new_episode_tag = "{self.__class__.__name__}:new_episode".format(self=self)
        self.new_spawn_tag   = "{self.__class__.__name__}:new_spawn".format(self=self)
        self.play_end_tag    = "{self.__class__.__name__}:play_end".format(self=self)
        super().__init__()

    def info(self, tag, dct):
        self._logger.debug("", extra=dict(tag=tag, data=dct))

    def on_new_episode(self, episode_n=None, obs=None, goal_obs=None):
        self.last_episode_n = episode_n
        self.info(self.human_tag,
                  dict(msg=" +++++++++++++++++ New episode: {episode_n} +++++++++++++".format(
                      episode_n=episode_n)))
        self.info(self.new_episode_tag,
                  dict(episode_n=episode_n,
                       obs=obs.tolist(),
                       goal_obs=self.prob.goal_obs.tolist()))

    def on_goal_hit(self, current_step):
        self.info(self.goal_hit_tag,
                  dict(episode_n=self.last_episode_n, steps=current_step))

    def on_new_spawn(self, obs, rew, action, pose, prob_steps, info):
        if rew >= self.prob.goal_reward:
            self.on_goal_hit(prob_steps)
        else:
            self.info(self.new_spawn_tag,
                      dict(pose      = pose.tolist(),
                           episode_n = int(self.last_episode_n),
                           steps     = int(prob_steps)))

    def on_new_step_with_pose_steps(self, obs, rew, act, pose, steps, info,
                                    **kw):
        if steps % self.log_interval == 0:
            self.info(self.new_step_tag,
                      dict(episode_n = int(self.last_episode_n),
                           steps     = int(steps),
                           obs       = obs.tolist(),
                           rew       = float(rew),
                           act       = int(act),
                           pose      = pose.tolist(),
                           info      = info))

    def on_new_step(self, obs, rew, action, info):
        if info.get("new_spawn", False):
            self.on_new_spawn(obs, rew, action, self.prob.pose, self.prob.steps, info)
        self.on_new_step_with_pose_steps(
            obs, rew, action, self.prob.pose, self.prob.steps, info)

    def on_play_end(self):
        self.info(self.play_end_tag, {})
        self.info(self.human_tag, "End of play")

    def tag_event_map(self):
        return dict([(self.new_step_tag, "on_new_step_with_pose_steps"),
                     (self.goal_hit_tag, "on_goal_hit"),
                     (self.new_episode_tag, "on_new_episode"),
                     (self.play_end_tag, "on_play_end"),
                     (self.new_spawn_tag, "on_new_spawn")])

    def replay_observers_from_logs(self, observers, log_file_reader):
        tag_event_map = self.tag_event_map()
        for dct, tag in log_file_reader.read_data():
            if tag in tag_event_map:
                for obs in observers:
                    if hasattr(obs, tag_event_map[tag]):
                        getattr(obs, tag_event_map[tag])(**dct)
            else:
                #print("Ignoring tag '{}'".format(tag))
                pass


class LogFileReader(object):
    def __init__(self, log_file_path, enc, sep="\t"):
        self.log_file_path = log_file_path
        self.sep         = sep
        self.enc         = enc

    @MEMOIZE_METHOD
    def logfileobj(self):
        return closing( open( self.log_file_path, "r" ) )

    def parse_next_line(self, line):
        pre_tag, tag, len_msg, msg = line.strip().split(self.sep)
        assert int(len_msg) == len(msg)
        return self.enc.loads(msg), tag

    def read_data(self):
        with self.logfileobj() as file_:
            for line in file_:
                yield self.parse_next_line(line)


class MultiObserver(object):
    """
    Class to independently observe the experiments
    """
    @extended_kwprop
    def __init__(self,
                 logging_observer  = xargs(LoggingObserver,
                                           ["logger_factory"]),
                 log_file_reader   = xargs(LogFileReader,
                                           "log_file_path enc".split()),
                 enc               = prop(lambda s: s.logging_encdec),
                 metrics_observers = xargs(ComputeMetricsFromLogReplay,
                                           """logging_observer log_file_reader
                                           prob""".split()),
                 observer_keys     = "logging_observer metrics_observers".split(),
                 observers         = prop(lambda s: OrderedDict([
                     (k , getattr(s, k))
                     for k in s.observer_keys])),
    ):
        self.observers = observers
        for o in self.observers.values():
            o.parent = self

    def add_observer(self, **kw):
        self.observers.update(kw)
        for o in kw.values():
            o.parent = self
        return self

    def __getattr__(self, attr):
        if attr in """set_prob set_alg on_new_episode on_episode_end
                        on_new_step on_play_start on_play_end""".split():
            childattr = [(k, getattr(o, attr))
                         for k, o in self.observers.items()]
            def wrapper(*args, **kwargs):
                for k, a in childattr:
                    a(*args, **kwargs)

            return wrapper
        else:
            super().__getattr__(attr)


class LogFileWriter(object):
    def __init__(self, logger):
        self._logger      = logger
        self.linesep     = "\n"

    def write_data(self, dct, tag):
        self._logger.debug("", extra=dict(tag=tag, data=dct))


NOOP_OBSERVER = NoOPObserver()


def default_logger_factory(name):
    return logging.getLogger(name)


def call_sometimes(func, call_prob = 1.0, rng = np.random.RandomState()):
    @wraps(func)
    def wrapper(*a, **kw):
        if rng.rand() < call_prob:
            return func(*a, **kw)
    return wrapper


def show_ax_log(ax, data, tag = "c", image_file_fmt = "/tmp/{tag}.png"):
    fname = image_file_fmt.format(
        tag = tag,
        episode=data["episode_n"], step=data["steps"])
    img_dir = os.path.dirname(fname)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    print("Writing img to: {}".format(fname))
    return draw.imwrite(fname, ax)


def show_ax_human(ax, data, tag = "c"):
    return draw.imshow(tag, ax)


class Renderer(Enum):
    noop = lambda prob: 0
    gym = lambda prob: prob.render()
    human = lambda prob: prob.render(mode = 'human')
    sometimes = call_sometimes(lambda prob: prob.render(mode = 'human'),
                               call_prob = 0.1)
    log = lambda prob: prob.render(mode = 'log')


def play_episode(alg, prob, observer, episode_n,
                 renderer = Renderer.noop,
                 # 5 million steps in one episode is a lot
                 max_steps = 5000000):
    prob.episode_reset(episode_n)
    alg.episode_reset(episode_n)
    obs = prob.observation()
    observer.on_new_episode(episode_n=episode_n, obs=obs,
                            goal_obs=prob.goal_obs)
    act = alg.update(obs, None, None, None, dict())
    reward = 0
    for step_n in range(max_steps):
        obs, rew, done, info = prob.step(act)
        reward += rew
        observer.on_new_step(obs=obs, rew=rew, action=act, info=info)

        act = alg.update(obs, act, rew, done, info)

        renderer(prob)
        if done:
            break

    # Record end of episode
    observer.on_episode_end(episode_n)
    return reward


def play(alg,
         prob,
         observer = NOOP_OBSERVER,
         nepisodes = 1,
         logger_factory = default_logger_factory,
         play_episode_ = play_episode):
    """
    renderer: Options are:
      - Renderer.noop
      - Renderer.gym
      - Renderer.human
      - Renderer.sometimes
    """
    logger = logger_factory(__name__)
    if len(logging.root.handlers) >= 2 and hasattr(logging.root.handlers[1], "baseFilename"):
        logger.info("Logging to file : {}".format(logging.root.handlers[1].baseFilename))
    observer.set_prob(prob)
    observer.set_alg(alg)
    observer.on_play_start()
    for n in range(nepisodes):
        play_episode_(alg, prob, observer, n)

    observer.on_play_end()
    return observer


def condition_by_type_map():
    return {str: operator.eq,
            list: operator.contains}


def comparison_op(criteria_val, data_val):
    return condition_by_type_map()[type(criteria_val)](criteria_val, data_val)


def data_cmp_criteria(data, criteria,
                      cmp_op = comparison_op):
    return all(cmp_op(v, data.get(k, None)) for k, v in criteria.items())


def filter_by_tag_data(data_cmp_criteria = data_cmp_criteria,
                       **criteria):
    def func(data_tag):
        data, tag = data_tag
        if not isinstance(data, dict):
            return False

        if "tag" in data:
            raise NotImplementedError("Data should not have tag")
        data["tag"] = tag
        if data_cmp_criteria(data, criteria):
            return True
        else:
            return False
    return func


def post_process_data_iter(log_file_reader = None, filter_criteria = dict()):
    return filter(filter_by_tag_data(**filter_criteria) ,
                  log_file_reader.read_data())


def post_process_generic(data_iter, process_data_tag):
    return [process_data_tag(data=data, tag=tag)
            for data, tag in data_iter()] 
