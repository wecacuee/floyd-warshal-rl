from ..game.play import play, NoOPObserver, play_episode
from ..prob.gym import GymProblem
from ..alg.qlnet import QLearningNetAgent
import gym
import logging
import numpy as np
import torch as t

def demo():
    cartpole = GymProblem(gym.make("CartPole-v0").unwrapped)
    qlnet = QLearningNetAgent(observation_space = cartpole.observation_space,
                              action_space = cartpole.action_space,
                              reward_range = cartpole.reward_range,
                              rng = np.random.RandomState(seed = 0))

    return cartpole, qlnet
    #play(qlnet, cartpole, NoOPObserver(), 1000000, lambda x: logging.getLogger(__name__))

