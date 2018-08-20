# stdlib
# -*- coding: utf-8 -*-
from pathlib import Path
import os
from queue import PriorityQueue
import logging
import operator
from functools import reduce, lru_cache, partial
from collections import namedtuple

import numpy as np

# installed
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# project
from umcog.misc import NumpyEncoder
from umcog.confutils import xargs, xargspartial, xargmem, KWProp, extended_kwprop
import umcog.draw as draw
from ..game.play import Space, Alg, NoOPObserver, post_process_data_iter
from .qlearning import egreedy_prob_exp

LOG = logging.getLogger(__name__)

# if gpu is to be used
@lru_cache()
def get_device():
    return t.device("cuda" if t.cuda.is_available() else "cpu")

def logger():
    return logging.getLogger(__name__)


def notnone(x):
    return x is not None

def moving_average(er, window):
    #er = np.ascontiguousarray(er).reshape(-1)
    #erunroll = np.lib.stride_tricks.as_strided(er, shape = (er.shape[0]-window+1, window),
    #                                strides = (er.dtype.itemsize, er.dtype.itemsize),
    #                                writeable = False)
    #ma = np.mean(erunroll, axis=-1)
    window = min(len(er), window)
    cum = np.cumsum(er)
    cum[window:] = cum[window:] - cum[:-window]
    #cum[:window-1] = 0
    return cum / window

class QConvNet(nn.Module):
    def __init__(self, D_in, D_out, H):
        super(QConvNet, self).__init__()
        self.D_in = D_in
        self.D_out= D_out
        self.H = H
        self.conv1 = nn.Conv2d(1, H, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(H)
        self.conv2 = nn.Conv2d(H, 2*H, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(2*H)
        self.head = nn.Linear(prod(D_in), D_out)

    def forward(self, x):
        x = t.as_tensor(x).to(get_device())
        x = self.bn2(self.conv2(self.bn1(self.conv1(x))))
        return self.head(x.view(x.size(0), -1))

    def __call__(self, state_batch, act_batch=None):
        # Create a parameteric function that takes state and action and returns
        # the Q-value
        act_values = super(QConvNet, self).__call__(state_batch)
        return act_values if act_batch is None else act_values.gather(1, act_batch)


def apply_layer(x, layer, activation = F.relu):
    return activation(layer(x))

class MLP(nn.Module):
    def __init__(self, observation_size, hiddens, action_size):
        super(MLP, self).__init__()
        ph = observation_size
        for i, h in enumerate(hiddens):
            self.add_module("lin{}".format(i), nn.Linear(ph, h))
            ph = h
        self.add_module("lin{}".format(len(hiddens)), nn.Linear(ph, action_size))

    def forward(self, x):
        return reduce(apply_layer, self.children(), x)

class QLinearNet(nn.Module):
    def __init__(self, D_in, D_out, H):
        super(QLinearNet, self).__init__()
        self.D_in = D_in
        self.D_out= D_out
        self.H = H
        self.lin1 = nn.Linear(prod(D_in), H)
        self.t1 = nn.Tanh()
        self.head = nn.Linear(H, D_out)

    def forward(self, x):
        x = self.t1(self.lin1(x))
        return self.head(x)

    def __call__(self, state_batch, act_batch=None):
        # Create a parameteric function that takes state and action and returns
        # the Q-value
        state_batch = t.as_tensor(state_batch).to(
            device=get_device(), dtype=t.float32)
        if state_batch.shape[0] == 0:
            return t.empty(0)
        if state_batch.ndimension() < 2:
            state_batch.unsqueeze_(dim=0)
        act_values = super(QLinearNet, self).__call__(state_batch)
        return (act_values
                if act_batch is None
                else act_values.gather(1, act_batch.reshape(-1, 1)))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.memory = np.empty(capacity, dtype=np.object)
        self.position = 0
        self.rng = rng

    def push(self, *args):
        """Saves a transition."""
        self.memory[self.position % self.capacity] = Transition(*args)
        self.position = self.position + 1

    def sample(self, batch_size):
        valid_memory = (self.memory[:self.position]
                        if self.position < self.capacity
                        else self.memory)
        return self.rng.choice(valid_memory, batch_size).tolist()

    def __len__(self):
        return len(self.memory)


def prod(seq):
    return reduce(operator.mul, seq, 1)

def egreedy_prob_tut(steps_done, start_eps = 0.9, end_eps = 0.05, eps_decay = 200):
    return  end_eps + (start_eps - end_eps) * \
        np.exp(-1. * steps_done / eps_decay)

class QLearningNetAgent:
    egreedy_prob_exp = egreedy_prob_exp
    egreedy_prob_tut = egreedy_prob_tut
    def __init__(self,
                 action_space,
                 observation_space,
                 reward_range,
                 rng,
                 nepisodes,
                 egreedy_prob = egreedy_prob_tut,
                 discount              = 0.999, # step cost
                 hidden_state_size     = 64,
                 qnet                  = MLP,
                 batch_size            = 128,
                 batch_update_prob     = 0.1,
                 target_update_prob    = 0.1,
                 goal_reward           = 10,
                 learning_rate         = 1e-2,
                 memory_size           = 10000,
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self.reward_range         = reward_range
        self.rng                  = rng
        self.egreedy_prob         = egreedy_prob
        self.init_value           = discount * reward_range[0]
        self.discount             = discount
        self.hidden_state_size    = hidden_state_size
        self.qnet                 = qnet
        self.batch_size           = batch_size
        self.batch_update_prob    = batch_update_prob
        self.target_update_prob   = target_update_prob
        self.goal_reward          = goal_reward
        self.learning_rate        = learning_rate
        self.memory_size          = memory_size
        self.reset()

    @property
    def egreedy_epsilon(self):
        return self.egreedy_prob(self._episode_count)

    def episode_reset(self, episode_n):
        # Reset parameters
        #self.action_value[:]= self.init_value
        self._last_state_idx = None
        self._episode_count = episode_n
        self.episode_rewards.append(self._this_episode_reward)
        self._this_episode_reward = 0
        if np.random.rand() < self.target_update_prob:
            self._action_value_target.load_state_dict(
                self._action_value_online.state_dict())
            self._plot_rewards()


    def _default_action_value(self):
        return self.qnet(self.observation_space.shape[0],
                         [self.hidden_state_size],
                         self.action_space.n)

    def reset(self):
        self._action_value_online = self._default_action_value()
        self._action_value_online.to(device = get_device())
        self._action_value_target = self._default_action_value().to(device = get_device())
        self._action_value_online.to(device = get_device())
        self._action_value_target.load_state_dict(self._action_value_online.state_dict())
        # set target Q function in eval mode (opposite of training mode)
        self._action_value_target.eval()
        self._optimizer = optim.RMSprop(self._action_value_online.parameters())
        # running mean and std
        self._obs_n_mean_std = (0, None, None)
        self._episode_count = 0
        self.episode_rewards = []
        self._this_episode_reward = 0
        self._memory = ReplayMemory(self.memory_size, self.rng)
        self.episode_reset(0)

    def egreedy(self, greedy_act):
        sample_greedy = (self.rng.rand() >= self.egreedy_epsilon)
        #return greedy_act if sample_greedy else t.random.randint(
        #    self.action_space.n, size=(1,1), device=get_device(), dtype=t.long)
        return greedy_act if sample_greedy else np.random.randint(self.action_space.n)

    def _state_from_obs_noop(self, obs):
        return t.as_tensor(obs, device=get_device(), dtype=t.float).view(1, -1)

    def _state_from_obs(self, obs):
        n, mean, std = self._obs_n_mean_std
        if notnone(mean) and notnone(std):
            mean = (n * mean + obs) / (n+1)
            std = np.sqrt(((n * std**2) + (obs-mean)**2) / (n+1))
            state = (obs - mean) / np.where(std == 0, 1, std)
        else:
            mean = obs
            std = 0
            state = obs

        self._obs_n_mean_std = (n+1, mean, std)
        return t.as_tensor(state, device=get_device(), dtype=t.float).view(1, -1)

    def policy(self, obs, usetarget=False):
        state = self._state_from_obs_noop(obs)
        Q = self._action_value_target if usetarget else self._action_value_online
        with t.no_grad():
            return Q(state).argmax(dim = -1).item()

    def _hit_goal(self, rew):
        return rew >= self.goal_reward

    def on_hit_goal(self, obs, act, rew):
        self._last_state_idx = None

    def update(self, obs, act, rew, done, info):
        # Protocol defined by: game.play:play_episode()
        # - act = alg.policy(obs)
        # - obs_plus_1, rew_plus_1 = the prob.step(act)
        # - the alg.update(obs, act, rew)
        # or
        # obs_m_1 --alg--> act --prob--> obs, rew # # # obs, rew = prob.step(action)
        if not self.observation_space.contains(obs):
            raise ValueError("Bad observation {obs}".format(obs=obs))

        st = self._state_from_obs_noop(obs)
        stm1 = self._last_state_idx
        self._last_state_idx = st
        if stm1 is None:
            return self.egreedy(self.policy(obs))

        self._this_episode_reward += rew

        # Encoding state_hash from observation
        if self._hit_goal(rew):
            self.on_hit_goal(obs, act, rew)

        transition = (stm1,
                      t.as_tensor([act], device=get_device(), dtype=t.long).view(1,1),
                      st,
                      t.as_tensor([rew], device=get_device()),
                      t.as_tensor([done], device=get_device(), dtype=t.uint8))
        #print(", ".join(map(str, transition)))
        self._memory.push(*transition)
        if self.rng.rand() < self.batch_update_prob:
            self.batch_update()

        return self.egreedy(self.policy(obs))

    def _plot_rewards(self):
        ax = draw.backend.mplfig().add_subplot(111)
        ax.set_title("Training ...")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rewards")
        ax.plot(self.episode_rewards)
        ax.plot(moving_average(self.episode_rewards, 100))
        draw.imshow("rewards", ax)

    def batch_update(self):
        # 1. L = rₜ + γ Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a)) - Q(sₜ, aₜ)
        # 2. θₜ₊₁ = θₜ - α ∇L
        # Abbreviate the variables
        # Q(s, a)
        Q = self._action_value_online
        # Q'(s, a)
        Q_target = self._action_value_target
        # γ
        d = self.discount

        transitions = self._memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # 1-m
        done_batch = t.cat(batch.done)
        next_state_batch = t.cat(batch.next_state)
        state_batch = t.cat(batch.state)
        action_batch = t.cat(batch.action).view(-1, 1)
        reward_batch = t.cat(batch.reward)

        # Q(sₜ, aₜ)
        # Compute Q(s_t, aₜ) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = Q(state_batch).gather(1, action_batch)

        # next_state_best_actions = argmaxₐ Q(sₜ₊₁, a)
        non_final_next_states = next_state_batch[~done_batch]
        next_state_best_actions = Q(non_final_next_states).argmax(dim=-1)
        # next_state_values = Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a))
        next_state_values = t.zeros(self.batch_size, device=get_device())
        #action_0_choice = (next_state_best_actions == 0).sum().tolist() / next_state_best_actions.shape[0]
        #print("best action prob: 0:{}, 1:{}".format(action_0_choice, 1- action_0_choice))
        next_state_values[~done_batch] = Q_target(
            #non_final_next_states).gather(1, next_state_best_actions).detach().squeeze()
            non_final_next_states).max(dim=1)[0].detach()
        # Note: detach() de-attaches the Tensor from the graph, hence avoid
        # gradient back propagation.

        # Compute the expected Q values
        # rₜ + γ Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a))
        expected_state_action_values = (next_state_values * d) + reward_batch
        #print("r+max_a Q_target(s, a): {}".format(expected_state_action_values))

        # Compute Huber loss
        # loss = rₜ + γ Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a)) - Q(sₜ, aₜ)
        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1))

        # Optimize the model
        # Reset the _optimizer
        self._optimizer.zero_grad()
        # Compute ∇ L
        loss.backward()
        # trim the gradients
        # ∇ L <- max(min(∇L, 1)), -1)
        for param in Q.parameters():
            param.grad.data.clamp_(-1, 1)
        # 2. θₜ₊₁ = θₜ - α max(min(∇L, 1)), -1)
        self._optimizer.step()

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False
