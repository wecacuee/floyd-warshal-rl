
from ..game.play import (Problem)

class GymProblem:
    def __init__(self, gym):
        self._gym = gym
        self.action_space = gym.action_space
        self.observation_space = gym.observation_space
        self.reward_range = gym.reward_range
        self._episode_n = 0
        self._reward = None
        self._done = False

    def reward(self):
        return self._reward

    def observation(self):
        return self._obs

    def done(self):
        return self._done

    def reset(self):
        self._obs = self._gym.reset()
        return self._obs

    def step(self, a):
        x = self._gym.step(a)
        self._obs, self._reward, self._done, info = x
        return self._obs, self._reward

    def episode_reset(self, episode_n):
        self._episode_n = episode_n
        return self.reset()

    def __getattr__(self, a):
        return getattr(self._gym, a)

