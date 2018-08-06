# stdlib
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class QLearningNet:
    def __init__(self, qnet):
        self.qnet = qnet
    def __call__(self, state, act=None):
        #return self.init_value * t.ones((state_size, self.action_space.size))
        # Create a parameteric function that takes state and action and returns
        # the Q-value
        act_values = self.qnet(state)
        return act_values if act is None else act_values[act]

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
                 egreedy_epsilon       = 0.00,
                 action_value_momentum = 0.1, # Low momentum changes more frequently
                 discount              = 0.99, # step cost
                 hidden_state_size     = 10,
                 target_update_m       = 0.6,
                 qnet                  = QLearningNet,
                 batch_size            = 16,
                 batch_update_prob     = 0.1,
                 gamma                 = 0.99,
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self.reward_range         = reward_range
        self.rng                  = rng
        self.egreedy_epsilon      = egreedy_epsilon
        self.action_value_momentum= action_value_momentum
        assert reward_range[0] >= 0, "Reward range"
        self.init_value           = discount * reward_range[0]
        self.discount             = discount
        self.hidden_state_size    = hidden_state_size
        self.target_update_m      = target_update_m
        self.qnet                 = qnet
        self.batch_size           = batch_size
        self.batch_update_prob    = batch_update_prob
        self.gamma                = gamma
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
        return np.argmax(Q(state))

    def _hit_goal(self, rew):
        return rew >= 9

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
        st = self._state_from_obs(obs)

        stm1, am1 = self.last_state_idx_act or (None, None)
        self.last_state_idx_act = st, act
        if stm1 is None:
            return

        if self._hit_goal(rew):
            self.on_hit_goal(obs, act, rew)
        self.memory.add(stm1, act, st, act)
        if random.rand() < self.batch_update_prob:
            self.batch_update()

    def batch_update(self):
        # Abbreviate the variables
        qm = self.action_value_momentum
        Q = self.action_value_online
        Q_t = self.action_value_target
        d = self.discount

        # Update step from online observed reward
        #Q(stm1, act) = (1-qm) * (rew + d * np.max(Q(st, :))) + qm * Q(stm1, act)
        #which is equivalent to loss.update()

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = t.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8)
        non_final_next_states = t.cat([s for s in batch.next_state
                                       if s is not None])
        state_batch = t.cat(batch.state)
        action_batch = t.cat(batch.action)
        reward_batch = t.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.action_value_online(state_batch).gather(
            1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = t.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.action_value_target(
            non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1))

        # Optimize the model
        # Reset the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # WE are not updating a single transition at a time but a batch at a time.
        #loss = (rew + d * Q_t(st, t.argmax(Q(st)))) - Q(stm1, act)

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False
