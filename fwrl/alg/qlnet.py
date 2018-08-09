# stdlib
# -*- coding: utf-8 -*-
from pathlib import Path
import os
from queue import PriorityQueue
import logging
import operator
import functools
from functools import reduce, lru_cache
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

LOG = logging.getLogger(__name__)

# if gpu is to be used
@lru_cache()
def device():
    return t.device("cuda" if t.cuda.is_available() else "cpu")

def logger():
    return logging.getLogger(__name__)


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
        x = t.as_tensor(x).to(device())
        x = self.bn2(self.conv2(self.bn1(self.conv1(x))))
        return self.head(x.view(x.size(0), -1))

    def __call__(self, state_batch, act_batch=None):
        # Create a parameteric function that takes state and action and returns
        # the Q-value
        act_values = super(QConvNet, self).__call__(state_batch)
        return act_values if act_batch is None else act_values.gather(1, act_batch)


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
            device=device(), dtype=t.float32)
        if state_batch.shape[0] == 0:
            return t.empty(0)
        if state_batch.ndimension() < 2:
            state_batch.unsqueeze_(dim=0)
        act_values = super(QLinearNet, self).__call__(state_batch)
        return (act_values
                if act_batch is None
                else act_values.gather(1, act_batch.reshape(-1, 1)))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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


class QLearningNetAgent:
    def __init__(self,
                 action_space,
                 observation_space,
                 reward_range,
                 rng,
                 egreedy_start         = 0.90,
                 egreedy_end           = 0.01,
                 egreedy_episodes      = 1000,
                 discount              = 0.99, # step cost
                 hidden_state_size     = 20,
                 qnet                  = QLinearNet,
                 batch_size            = 128,
                 batch_update_prob     = 0.1,
                 target_update_prob    = 0.1,
                 goal_reward           = 10,
                 learning_rate         = 1e-5,
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self.reward_range         = reward_range
        self.rng                  = rng
        self.egreedy_start        = egreedy_start
        self.egreedy_end          = egreedy_end
        self.egreedy_episodes     = egreedy_episodes
        self.init_value           = discount * reward_range[0]
        self.discount             = discount
        self.hidden_state_size    = hidden_state_size
        self.qnet                 = qnet
        self.batch_size           = batch_size
        self.batch_update_prob    = batch_update_prob
        self.target_update_prob   = target_update_prob
        self.goal_reward          = goal_reward
        self._action_value_online = self._default_action_value()
        self._action_value_online.to(device = device())
        self._action_value_target = self._default_action_value().to(device = device())
        self._action_value_online.to(device = device())
        self._action_value_target.load_state_dict(self._action_value_online.state_dict())
        self._optimizer = optim.RMSprop(self._action_value_online.parameters(),
                                        lr=learning_rate)
        self._episode_count = 0
        self.reset()

    @property
    def egreedy_epsilon(self):
        n = min(self._episode_count, self.egreedy_episodes)
        return (self.egreedy_end - self.egreedy_start) * n / self.egreedy_episodes + self.egreedy_start

    def episode_reset(self, episode_n):
        # Reset parameters
        #self.action_value[:]= self.init_value
        self._last_state_idx_act = None
        self._episode_count = episode_n


    def _default_action_value(self):
        return self.qnet(self.observation_space.shape,
                         self.action_space.n,
                         self.hidden_state_size)

    def reset(self):
        self.episode_reset(0)
        self._memory = ReplayMemory(10000, self.rng)

    def egreedy(self, greedy_act):
        sample_greedy = (self.rng.rand() >= self.egreedy_epsilon)
        return greedy_act if sample_greedy else self.action_space.sample()

    def _state_from_obs(self, obs):
        state = obs # fully observed system
        return state

    def policy(self, obs, usetarget=False):
        state = self._state_from_obs(obs)
        Q = self._action_value_target if usetarget else self._action_value_online
        return t.argmax(Q(state)).tolist()

    def _hit_goal(self, rew):
        return rew >= self.goal_reward

    def on_hit_goal(self, obs, act, rew):
        self._last_state_idx_act = None

    def update(self, obs, act, rew):
        # Protocol defined by: game.play:play_episode()
        # - act = alg.policy(obs)
        # - obs_plus_1, rew_plus_1 = the prob.step(act)
        # - the alg.update(obs, act, rew)
        # or
        # obs_m_1 --alg--> act --prob--> obs, rew # # # obs, rew = prob.step(action)
        if obs is None:
            stm1, am1 = self._last_state_idx_act
            self._memory.push(t.as_tensor(stm1), act, None, -100)
            return

        if not self.observation_space.contains(obs):
            raise ValueError("Bad observation {obs}".format(obs=obs))

        # Encoding state_hash from observation
        st = self._state_from_obs(obs) # does nothing

        if self._last_state_idx_act is None:
            self._last_state_idx_act = st, act
            return
        stm1, am1 = self._last_state_idx_act

        if self._hit_goal(rew):
            self.on_hit_goal(obs, act, rew)

        stm1 = t.as_tensor(stm1)
        act = act
        st = t.as_tensor(st)
        rew = rew
        self._memory.push(stm1, act, st, rew)
        if self.rng.rand() < self.batch_update_prob:
            self.batch_update()

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
        non_final_mask = t.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device(), dtype=t.uint8)
        # sₜ₊₁
        non_none = [s
                    for s in batch.next_state
                    if s is not None]
        if non_none:
            non_final_next_states = t.cat(non_none).reshape(-1, non_none[0].shape[0])
        else:
            non_final_next_states = t.empty(0)
        # sₜ
        state_batch = t.cat(batch.state).reshape(-1, batch.state[0].shape[0])
        # aₜ
        action_batch = t.tensor(batch.action, device=device())
        action_0_batch = sum(batch.action) / len(batch.action)
        print("actual action prob: 0:{}, 1:{}".format(action_0_batch, 1- action_0_batch))
        # rₜ
        reward_batch = t.tensor(batch.reward, device=device())

        # Q(sₜ, aₜ)
        # Compute Q(s_t, aₜ) - the model computes Q(s_t), then we select the
        # columns of actions taken
        # print("a : {}".format(action_batch))
        # print("Q(s, :) {}".format(Q(state_batch)))
        state_action_values = Q(state_batch, action_batch)

        # next_state_best_actions = argmaxₐ Q(sₜ₊₁, a)
        next_state_best_actions = Q(non_final_next_states).argmax(dim=-1)
        # next_state_values = Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a))
        next_state_values = t.zeros(self.batch_size, device=device())
        action_0_choice = (next_state_best_actions == 0).sum().tolist() / next_state_best_actions.shape[0]
        print("best action prob: 0:{}, 1:{}".format(action_0_choice, 1- action_0_choice))
        next_state_values[non_final_mask] = Q_target(
            non_final_next_states, next_state_best_actions).detach().squeeze()
            #non_final_next_states).max(1)[0].detach().squeeze()
        # Note: detach() de-attaches the Tensor from the graph, hence avoid
        # gradient back propagation.

        # Compute the expected Q values
        # rₜ + γ Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a))
        #print("max_a Q_target(s', a): {}".format(next_state_values))
        expected_state_action_values = (next_state_values * d) + reward_batch
        #print("r+max_a Q_target(s, a): {}".format(expected_state_action_values))
        #print("Q(s, a): {}".format(state_action_values))

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
        print("Q_(t+1)(s, a): {}".format(Q(state_batch, action_batch).mean()))
        final_mask = ~non_final_mask
        if final_mask.any():
            print("terminal Q_target(s, a): {}".format(
                expected_state_action_values[final_mask].mean()))
            print("terminal Q_t(s, a): {}".format(
                Q(state_batch, action_batch)[final_mask].mean()))


        if self.rng.rand() < self.target_update_prob:
            self._action_value_target.load_state_dict(self._action_value_online.state_dict())
        print("\nBatch update loss: {}".format(loss))

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False
