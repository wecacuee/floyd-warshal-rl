# Makes relative imports possible
if not __package__ : __package__ = "fwrl.conf"
import logging
from functools import partial

import numpy as np
import torch as t
import gym

from ..game.play import play, NoOPObserver, play_episode
from ..prob.gym import GymProblem
from ..alg.qlnet import QLearningNetAgent

import random, torch

def demo(nepisodes = 500, seed = 0, max_steps = 5000):
    cartpole = GymProblem(gym.make("CartPole-v0").unwrapped, seed = seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    qlnet = QLearningNetAgent(observation_space = cartpole.observation_space,
                              action_space = cartpole.action_space,
                              reward_range = cartpole.reward_range,
                              rng = np.random.RandomState(seed = 0),
                              nepisodes = nepisodes)

    play(qlnet, cartpole, nepisodes = nepisodes,
         play_episode_ = partial(play_episode, max_steps = max_steps))
    return qlnet, cartpole


if __name__ == "__main__":
    demo()
