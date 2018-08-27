import numpy as np

from collections import namedtuple
from ..game.play import (Problem)

EpisodeData = namedtuple('EpisodeData', "obs reward done info".split())

class GymProblem:
    def __init__(self, gym, seed = 0):
        self._gym              = gym
        self.action_space      = gym.action_space
        self.observation_space = gym.observation_space
        self.reward_range      = gym.reward_range
        self._episode_n        = 0
        self._episode_data     = None

        self._gym.seed(seed)
        self.reset()

    def reward(self):
        return self._episode_data.reward

    def observation(self):
        return self._episode_data.obs

    def done(self):
        return self._episode_data.done

    def reset(self):
        obs = self._gym.reset()
        self._episode_data = EpisodeData(obs, 0, False, dict())
        return obs

    def step(self, a):
        x = self._gym.step(a)
        self._episode_data = EpisodeData(*x)
        return x

    def render(self, *a, **kw):
        self._gym.render(*a, **kw)

    def episode_reset(self, episode_n):
        self._episode_n = episode_n
        return self.reset()

    def __getattr__(self, a):
        return getattr(self._gym, a)

