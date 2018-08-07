from ..game.play import play, NoOPObserver
from ..prob.gym import GymProblem
from ..alg.qlnet import QLearningNetAgent
import gym
import logging
import numpy as np
import torch as t

def demo():
    cartpole = GymProblem(gym.make("CartPole-v0"))
    qlnet = QLearningNetAgent(observation_space = cartpole.observation_space,
                              action_space = cartpole.action_space,
                              reward_range = cartpole.reward_range,
                              rng = np.random.RandomState(seed = 0))
    play(qlnet, cartpole, NoOPObserver(), 1000000, lambda x: logging.getLogger(__name__))

