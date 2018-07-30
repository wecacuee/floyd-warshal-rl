from pathlib import Path
import torch as t
import os
from queue import PriorityQueue
import logging
import operator
import functools

from cog.misc import NumpyEncoder
from cog.confutils import xargs, xargspartial, xargmem, KWProp, extended_kwprop
import cog.draw as draw
from game.play import Space, Alg, NoOPObserver, post_process_data_iter

def logger():
    return logging.getLogger(__name__)


class QNet:
    def __init__(self, D_in, D_out, H):
        self.D_in = D_in
        self.D_out= D_out
        self.H = H
        self.qnet = t.nn.Sequential(
            t.nn.Linear(D_in, H),
            t.nn.ReLU(),
            t.nn.Linear(H, D_out))

    def __call__(self, state, act=None):
        #return self.init_value * t.ones((state_size, self.action_space.size))
        # Create a parameteric function that takes state and action and returns
        # the Q-value
        act_values = self.qnet(state)
        return act_values if act is None else act_values[act]


class QLearningNet:
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
                 qnet                  = QNet
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
        self.reset()

    def episode_reset(self, episode_n):
        # Reset parameters
        #self.action_value[:]= self.init_value
        self.action_value_online = self._default_action_value()
        self.action_value_target = self._default_action_value()

    def _default_action_value(self):
        return self.qnet(self.observation_space.size, self.action_space.size,
                         self.hidden_state_size)

    def reset(self):
        self.action_value_online = self._default_action_value()
        self.action_value_target = self._default_action_value()
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

        # Abbreviate the variables
        qm = self.action_value_momentum
        Q = self.action_value_online
        Q_t = self.action_value_target
        d = self.discount

        # Update step from online observed reward
        #Q(stm1, act) = (1-qm) * (rew + d * np.max(Q(st, :))) + qm * Q(stm1, act)
        #which is equivalent to loss.update()
        loss = (rew + d * Q_t(st, t.argmax(Q(st)))) - Q(stm1, act)

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False
