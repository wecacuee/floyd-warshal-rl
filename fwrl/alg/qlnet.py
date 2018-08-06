# stdlib
# -*- coding: utf-8 -*-
from pathlib import Path
import os
from queue import PriorityQueue
import logging
import operator
import functools
from collections import namedtuple

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


# if gpu is to be used
device = t.device("cuda" if torch.cuda.is_available() else "cpu")

def logger():
    return logging.getLogger(__name__)


class QNet(nn.Module):
    def __init__(self, D_in, D_out, H):
        super(QNet, self).__init__()
        self.D_in = D_in
        self.D_out= D_out
        self.H = H
        self.conv1 = nn.Conv2d(1, H, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(H)
        self.conv2 = nn.Conv2d(H, 2*H, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(2*H)
        self.head = nn.Linear(D_in.cumprod(), D_out)

    def forward(self, x):
        x = self.bn2(self.conv2(self.bn1(self.conv1(x))))
        return self.head(x.view(x.size(0), -1))

    def __call__(self, state_batch, act_batch=None):
        # Create a parameteric function that takes state and action and returns
        # the Q-value
        act_values = super(QNet, self).__call__(state_batch)
        return act_values if act is None else act_values.gather(1, act_batch)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QLearningNetAgent:
    def __init__(self,
                 action_space,
                 observation_space,
                 reward_range,
                 rng,
                 egreedy_epsilon       = 0.05,
                 discount              = 0.99, # step cost
                 hidden_state_size     = 10,
                 qnet                  = QNet,
                 batch_size            = 16,
                 batch_update_prob     = 0.1,
                 goal_reward           = 10,
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self.reward_range         = reward_range
        self.rng                  = rng
        self.egreedy_epsilon      = egreedy_epsilon
        assert reward_range[0] >= 0, "Reward range"
        self.init_value           = discount * reward_range[0]
        self.discount             = discount
        self.hidden_state_size    = hidden_state_size
        self.qnet                 = qnet
        self.batch_size           = batch_size
        self.batch_update_prob    = batch_update_prob
        self.reset()

    def episode_reset(self, episode_n):
        # Reset parameters
        #self.action_value[:]= self.init_value
        self.action_value_online = self._default_action_value()
        self.action_value_target = self._default_action_value()
        self.action_value_target.load_state_dict(self.action_value_online.state_dict())
        self.optimizer = optim.RMSprop(self.action_value_online.parameters())
        self.memory = ReplayMemory(10000)


    def _default_action_value(self):
        return self.qnet(self.observation_space.size, self.action_space.size,
                         self.hidden_state_size)

    def reset(self):
        self.episode_reset(0)

    def egreedy(self, greedy_act):
        sample_greedy = (self.rng.rand() >= self.egreedy_epsilon)
        return greedy_act if sample_greedy else self.action_space.sample()

    def _state_from_obs(self, obs):
        state = obs # fully observed system
        return state

    def policy(self, obs, usetarget=False):
        state = self._state_from_obs(obs)
        Q = self.action_value_target if usetarget else self.action_value_online
        return t.argmax(Q(state))

    def _hit_goal(self, rew):
        return rew >= self.goal_reward

    def on_hit_goal(self, obs, act, rew):
        self.last_state_idx_act = None

    def update(self, obs, act, rew):
        # Protocol defined by: game.play:play_episode()
        # - act = alg.policy(obs)
        # - obs_plus_1, rew_plus_1 = the prob.step(act)
        # - the alg.update(obs, act, rew)
        # or
        # obs_m_1 --alg--> act --prob--> obs, rew # # # obs, rew = prob.step(action)
        if not self.observation_space.contains(obs):
            raise ValueError("Bad observation {obs}".format(obs=obs))

        # Encoding state_hash from observation
        st = self._state_from_obs(obs) # does nothing

        if self.last_state_idx_act is None:
            self.last_state_idx_act = st, act
            return
        stm1, am1 = self.last_state_idx_act

        if self._hit_goal(rew):
            self.on_hit_goal(obs, act, rew)
        self.memory.add(stm1, act, st, act)
        if random.rand() < self.batch_update_prob:
            self.batch_update()

    def batch_update(self):
        # 1. L = rₜ + γ Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a)) - Q(sₜ, aₜ)
        # 2. θₜ₊₁ = θₜ - α ∇L
        # Abbreviate the variables
        # α
        qm = self.action_value_momentum
        # Q(s, a)
        Q = self.action_value_online
        # Q'(s, a)
        Q_target = self.action_value_target
        # γ
        d = self.discount

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # 1-m
        non_final_mask = t.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8)
        # sₜ₊₁
        non_final_next_states = t.cat([s for s in batch.next_state
                                       if s is not None])
        # sₜ
        state_batch = t.cat(batch.state)
        # aₜ
        action_batch = t.cat(batch.action)
        # rₜ
        reward_batch = t.cat(batch.reward)

        # Q(sₜ, aₜ)
        # Compute Q(s_t, aₜ) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = Q(state_batch, action_batch)

        # next_state_best_actions = argmaxₐ Q(sₜ₊₁, a)
        next_state_best_actions = Q(non_final_next_states).argmax(dim=-1)
        # next_state_values = Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a))
        next_state_values = t.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = Q_target(
            non_final_next_states, next_state_best_actions).detach()
        # Note: detach() de-attaches the Tensor from the graph, hence avoid
        # gradient back propagation.

        # Compute the expected Q values
        # rₜ + γ Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a))
        expected_state_action_values = (next_state_values * d) + reward_batch

        # Compute Huber loss
        # loss = rₜ + γ Q'(sₜ₊₁, argmaxₐ Q(sₜ₊₁, a)) - Q(sₜ, aₜ)
        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1))

        # Optimize the model
        # Reset the optimizer
        self.optimizer.zero_grad()
        # Compute ∇ L
        loss.backward()
        # trim the gradients
        # ∇ L <- max(min(∇L, 1)), -1)
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # 2. θₜ₊₁ = θₜ - α max(min(∇L, 1)), -1)
        self.optimizer.step()

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False
