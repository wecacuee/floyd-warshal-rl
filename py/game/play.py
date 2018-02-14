from __future__ import absolute_import, division, print_function
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    def episode_reset(self):
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

    def episode_reset(self):
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

    def getattr(self, attr):
        return lambda *args, **kwargs : 0


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
        

class LoggingObserver(NoOPObserver):
    def __init__(self):
        self.episode_n = 0
        self.total_rew = 0
        self.all_episode_rew = []
        super().__init__()
        
    def on_new_episode(self, episode_n):
        logger.debug(
            f" +++++++++++++++++ New episode: {episode_n} +++++++++++++")
        self.episode_n = episode_n
        self.all_episode_rew.append(self.total_rew)
        self.total_rew = 0

    def on_new_step(self, obs, rew, action):
        prob = self.prob
        self.total_rew += rew
        if prob.steps % 5 == 0:
            logger.debug(
                f"episode = {self.episode_n}, step = {prob.steps}, obs = {obs}; action = {action}; rew = {rew}; total_episode_reward = {self.total_rew}")

    def on_play_end(self):
        logger.info(f"all_episode_rew : {sum(self.all_episode_rew)}")

# Sample refrees
def play(alg, prob, observer, nepisodes):
    observer.set_prob(prob)
    observer.set_alg(alg)
    for n in range(nepisodes):
        observer.on_new_episode(n)
        play_episode(alg, prob, observer, n)

    observer.on_play_end()
    return observer

def play_episode(alg, prob, observer, episode_n):
    obs = prob.observation()
    rew = prob.reward()
    action = prob.action_space.sample()
    while not (prob.done() or alg.done()):
        observer.on_new_step(obs=obs, rew=rew, action=action)
        alg.update(obs, action, rew)
        action = alg.egreedy(alg.policy(obs))
        obs, rew = prob.step(action)
        #prob.render(None, 100, wait_time=1)

    prob.episode_reset()
    alg.episode_reset()
        
def play_from_conf(conf):
    play(conf.alg, conf.prob, conf.observer, conf.nepisodes)

def multiplay(multiconf):
    for conf in multiconf.trials:
        print(f"Running trial {conf.__class__.__name__}")
        play_from_conf(conf)

if __name__ == '__main__':
    import sys
    from cog.confutils import Conf
    conf = Conf.parse_all_args("conf.default:MultiPlayConf", sys.argv[1:])
    multiplay(conf)
