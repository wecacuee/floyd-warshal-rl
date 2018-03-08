from __future__ import absolute_import, division, print_function
import logging
import json
from contextlib import closing
from functools import lru_cache

from cog.misc import NumpyEncoder
from cog.memoize import MEMOIZE_METHOD

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
                      on_goal_hit on_new_goal_pose""".split():
            return lambda *args, **kwargs : 0
        else:
            raise AttributeError("attr = {attr} not found".format(attr  = attr))


class MultiObserver(object):
    """
    Class to independently observe the experiments
    """
    def __init__(self, observers):
        self.observers = observers
        for o in self.observers.values():
            o.parent = self

    def set_prob(self, prob):
        for o in self.observers.values():
            o.set_prob(prob)

    def set_alg(self, alg):
        for o in self.observers.values():
            o.set_alg(alg)

    def on_new_episode(self, episode_n):
        for o in self.observers.values():
            o.on_new_episode(episode_n)

    def on_new_step(self, obs, rew, action):
        for o in self.observers.values():
            o.on_new_step(obs, rew, action)

    def on_play_end(self):
        for o in self.observers.values():
            o.on_play_end()

class NPJSONEncDec(object):
    def dumps(self, dct):
        return json.dumps(dct, cls=NumpyEncoder)

    def loads(self, str_):
        return json.loads(str_, object_hook=NumpyEncoder.loads_hook)

class JSONLoggingFormatter(logging.Formatter):
    def __init__(self, enc, sep = "\t", **kwargs) :
        self.enc  = enc
        self.sep  = sep
        super().__init__(**kwargs)

    def format(self, record):
        str_ = super().format(record)
        record_str = self.enc.dumps(getattr(record, "data", {}))
        str_ = self.sep.join((str_, getattr(record, "tag", ""),
                              str(len(record_str)), record_str))
        return str_

class LogFileWriter(object):
    def __init__(self, logger):
        self._logger      = logger
        self.linesep     = "\n"

    def write_data(self, dct, tag):
        self._logger.debug("", extra=dict(tag=tag, data=dct))

class LogFileReader(object):
    def __init__(self, logfilepath, enc, sep="\t"):
        self.logfilepath = logfilepath
        self.sep         = sep
        self.enc         = enc

    @MEMOIZE_METHOD
    def logfileobj(self):
        return closing( open( self.logfilepath, "r" ) )

    def parse_next_line(self, line):
        pre_tag, tag, len_msg, msg = line.strip().split(self.sep)
        assert int(len_msg) == len(msg)
        return self.enc.loads(msg), tag

    def read_data(self):
        with self.logfileobj() as file_:
            for line in file_:
                yield self.parse_next_line(line)


class LoggingObserver(NoOPObserver):
    def __init__(self, logger, log_interval):
        self._logger = logger
        self.log_interval = log_interval
        self.human_tag       = "INFO"
        self.goal_hit_tag    = "{self.__class__.__name__}:goal_hit".format(self=self)
        self.new_step_tag    = "{self.__class__.__name__}:new_step".format(self=self)
        self.new_episode_tag = "{self.__class__.__name__}:new_episode".format(self=self)
        self.play_end_tag    = "{self.__class__.__name__}:play_end".format(self=self)
        super().__init__()

    def info(self, tag, dct):
        self._logger.debug("", extra=dict(tag=tag, data=dct))

    def on_new_episode(self, episode_n):
        self.last_episode_n = episode_n
        self.info(self.human_tag, 
                  " +++++++++++++++++ New episode: {episode_n} +++++++++++++".format(
                      episode_n=episode_n))
        self.info(self.new_episode_tag,
                  dict(episode_n=episode_n, goal_pose=self.prob.goal_pose.tolist()))

    def on_goal_hit(self, current_step):
        self.info(self.goal_hit_tag,
                  dict(episode_n=self.last_episode_n, steps=current_step))

    def on_new_step_with_pose_steps(self, obs,rew, act, pose, steps):
        if self.prob.hit_goal(): self.on_goal_hit(self.prob.steps)
        if steps % self.log_interval == 0:
            self.info(self.new_step_tag,
                      dict(episode_n = int(self.last_episode_n),
                           steps     = int(steps),
                           obs       = obs.tolist(),
                           rew       = float(rew),
                           act       = int(act),
                           pose      = pose.tolist()))

    def on_new_step(self, obs, rew, act):
        self.on_new_step_with_pose_steps(
            obs, rew, act, self.prob.pose, self.prob.steps)

    def on_play_end(self):
        self.info(self.play_end_tag, {})
        self.info(self.human_tag, "End of play")

    def tag_event_map(self):
        return dict([(self.new_step_tag, "on_new_step_with_pose_steps"),
                     (self.goal_hit_tag, "on_goal_hit"),
                     (self.new_episode_tag, "on_new_episode"),
                     (self.play_end_tag, "on_play_end")])

    def replay_observers_from_logs(self, observers, logfilereader):
        tag_event_map = self.tag_event_map()
        for dct, tag in logfilereader.read_data():
            if tag in tag_event_map:
                for obs in observers:
                    if hasattr(obs, tag_event_map[tag]):
                        getattr(obs, tag_event_map[tag])(**dct)

# Sample refrees
def play(alg, prob, observer, nepisodes):
    observer.set_prob(prob)
    observer.set_alg(alg)
    for n in range(nepisodes):
        play_episode(alg, prob, observer, n)

    observer.on_play_end()
    return observer

def play_episode(alg, prob, observer, episode_n):
    prob.episode_reset(episode_n)
    alg.episode_reset(episode_n)
    observer.on_new_episode(episode_n)
    obs = prob.observation()
    rew = prob.reward()
    action = prob.action_space.sample()
    while not (prob.done() or alg.done()):
        observer.on_new_step(obs=obs, rew=rew, action=action)
        alg.update(obs, action, rew)
        action = alg.egreedy(alg.policy(obs))
        obs, rew = prob.step(action)
        # prob.render(None, 100, wait_time=0)

        
def multiplay(trials, logger_factory):
    logger = logger_factory(__name__)
    for conf in trials:
        logger.info("Running trial {conf.__class__.__name__}".format(conf=conf))
        play(conf.alg, conf.prob, conf.observer, conf.nepisodes)
