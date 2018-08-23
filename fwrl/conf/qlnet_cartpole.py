# Makes relative imports possible
if not __package__ : __package__ = "fwrl.conf"
import logging
from functools import partial
import random
from datetime import datetime

import numpy as np
import torch as t
import gym

from ..game.play import play, NoOPObserver, play_episode, Renderer
from ..game.logging import LogFileConf
from ..prob.gym import GymProblem
from ..alg.qlnet import QLearningNetAgent
from .default import PROJECT_NAME


def demo(confname = "qlnet_cartpole", nepisodes = 1000, seed = 0, max_steps = 5000):
    cartpole = GymProblem(gym.make("CartPole-v0").unwrapped, seed = seed)
    random.seed(seed)
    np.random.seed(seed)
    if seed: t.manual_seed(seed)
    log_file_conf = LogFileConf(project_name = PROJECT_NAME, confname = confname)
    no_train_args = dict(batch_update_prob = 0,
                         target_update_prob = 0,
                         egreedy_prob = 0,
                         model_save_prob = 0)

    qlnet = QLearningNetAgent(observation_space = cartpole.observation_space,
                              action_space = cartpole.action_space,
                              reward_range = cartpole.reward_range,
                              rng = np.random.RandomState(seed = 0),
                              nepisodes = nepisodes,
                              model_save_dir = log_file_conf.log_file_dir)

    play(qlnet, cartpole, nepisodes = nepisodes,
         play_episode_ = partial(play_episode, max_steps = max_steps))

    qlnet_test = qlnet.test_mode()
    play(qlnet, cartpole, nepisodes = nepisodes / 10,
         play_episode_ = partial(play_episode, max_steps = max_steps, renderer = Renderer.human))

    return qlnet, cartpole


if __name__ == "__main__":
    demo()
