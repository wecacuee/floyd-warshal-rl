
from ..game.play import (Problem)

class GymProblem:
    def __init__(self, gym):
        self._gym = gym
        self.action_space = gym.action_space
        self.observation_space = gym.observation_space
        self.reward_range = gym.reward_range

    def observation(self):
        return self._obs

    def reset(self):
        self._obs = self._gym.reset()
        return self._obs

    def step(self, a):
        x = self._gym.step(a)
        self._obs = x[0]
        return x

    def episode_reset(self):
        return self.reset()

    def __getattr__(self, a):
        return getattr(self._gym, a)

