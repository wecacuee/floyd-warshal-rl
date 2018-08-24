# Makes relative imports possible
if not __package__ : __package__ = "fwrl.conf"
import logging
from functools import partial
import random
from datetime import datetime

import numpy as np
import torch
import gym

import gym_moving_dot

from umcog.confutils import extended_kwprop, KWProp, xargsonce, alias
from umcog.misc import compose

from ..alg.qlnet import QLearningNetAgent, RLinNet, MLP, QConvNet
from ..game.logging import LogFileConf
from ..game.play import play, NoOPObserver, play_episode, Renderer
from ..prob.gym import GymProblem
from ..prob.scrolling_grid_world import AgentInScrollingGW
from ..prob.windy_grid_world import AgentInGridWorld, AgentVisObserver
from .default import grid_world_play, PROJECT_NAME

play_qlnet_pong = partial(
    grid_world_play,
    nepisodes         = 1000,
    seed              = 0,
    max_steps         = 400,
    project_name      = PROJECT_NAME,
    confname          = "qlnet_pong",
    rng               = xargsonce(np.random.RandomState, ["seed"]),
    prob              = xargsonce(partial(GymProblem, gym.make("Pong-v0")),
                                  ["seed"]),
    action_space      = alias(["prob", "action_space"]),
    observation_space = alias(["prob", "observation_space"]),
    reward_range      = alias(["prob", "reward_range"]),
    log_file_conf     = xargsonce(LogFileConf, ["project_name", "confname"]),
    model_save_dir    = alias(["log_file_conf", "log_file_dir"]),
    qnet              = partial(QConvNet, hiddens = [32, 64]),
    alg               = xargsonce(partial(QLearningNetAgent,
                                          no_display = True),
                              """action_space observation_space reward_range
                              rng nepisodes qnet model_save_dir""".split()),
)

play_qlnet_moving_dot = partial(
    play_qlnet_pong,
    confname          = "qlnet_moving_dot",
    prob              = xargsonce(partial(GymProblem, gym.make("Pong-v0")),
                                  ["seed"]),
    qnet              = partial(QConvNet, hiddens = [8, 16]),
    alg               = xargsonce(partial(QLearningNetAgent,
                                          batch_update_prob = 0.1,
                                          no_display = True),
                              """action_space observation_space reward_range
                              rng nepisodes qnet model_save_dir""".split()),
)

def demo(confname = "qlnet_cartpole", nepisodes = 1000, seed = 0, max_steps = 5000):
    cartpole = GymProblem(gym.make("CartPole-v0").unwrapped, seed = seed)
    random.seed(seed)
    np.random.seed(seed)
    if seed: torch.manual_seed(seed)
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


def listget(l, i, default=None):
    return l[i] if len(l) > i else default

if __name__ == "__main__":
    import sys
    entry_point = listget(sys.argv, 1, "demo")
    globals()[entry_point]()
