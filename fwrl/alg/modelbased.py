from functools import partial

import torch

from umcog.memoize import MEMOIZE_METHOD

from .qlearning import egreedy_prob_exp
from ..game.play import Alg


class ModelBasedTabular(Alg):
    egreedy_prob_exp = egreedy_prob_exp

    def __init__(self,
                 action_space,
                 observation_space,
                 reward_range,
                 rng,
                 egreedy_prob = partial(egreedy_prob_exp, nepisodes = 200),
                 discount = 1.00):  # step cost
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = reward_range
        self.rng = rng
        self.egreedy_prob = self.egreedy_prob
        self.discount = discount
        self.reset()

    def reset(self):
        self.step = 0
        self.hash_state = dict()
        self.dynamics_model = self._defaut_dynamics(0)
        self.rewards = self._default_rewards(0)

    def _default_rewards(self, state_size):
        return torch.ones((state_size, self.action_space.n))

    def _resize_rewards(self, new_size):
        olds = self.rewards.shape[0]
        self.rewards.resize_(new_size, self.action_space.n, new_size)
        self.rewards[olds:new_size, :, olds:new_size] = 1
        return self.rewards

    def _defaut_dynamics(self, state_size):
        return torch.zeros((state_size, self.action_space.n, state_size))

    def _resize_dynamics(self, new_size):
        olds = self.dynamics_model.shape[0]
        self.dynamics_model.resize_(new_size, self.action_space.n, new_size)
        self.dynamics_model[olds:new_size, :, olds:new_size] = 1
        return self.dynamics_model

    def _state_from_obs(self, obs):
        obs = tuple(obs)
        if obs not in self.hash_state:
            # A new state has been visited
            state_idx = self.hash_state[obs] = max(
                self.hash_state.values(), default=-1) + 1
            self.dynamics_model = self._resize_dynamics(state_idx + 1)
            self.rewards = self._resize_rewards(state_idx + 1)
        else:
            state_idx = self.hash_state[obs]
        return state_idx

    def egreedy(self, greedy_act):
        sample_greedy = (self.rng.rand() >= self.egreedy_prob(self.step))
        return greedy_act if sample_greedy else self.action_space.sample()

    def _exploration_policy(self, state):
        nstates = self.dynamics_model.shape[0]
        experience_count = self.dynamics_model.view(nstates, -1).sum(dim = -1)
        goal_state = experience_count.argmin()
        return self._exploitation_policy(state,
                                         tuple(goal_state.tolist()))

    def _expected_reward(self, st):
        counts = self.dynamics_model[:, :, st]
        prob = counts / counts.sum(dim = -1)
        return (prob * self.rewards).sum(dim = -1)

    @MEMOIZE_METHOD
    def _exploitation_policy(self, state, goal_state):
        if self.dynamics_model[state, : goal_state].max() > 0:
            act = self.dynamics_model[state, :, goal_state].argmax()
            return act, self.rewards[state, act]
        else:
            nbr_states = self.dynamics_model[:, :,
                                             goal_state].max(
                                                 dim=-1)[0].nonzero()
            exp_rew = self._expected_reward(goal_state)[nbr_states]
            act_rewards = (self._exploitation_policy(state, nbr_st)
                           for nbr_st in nbr_states)
            act_cumrewards = (
                (act, rew + erew)
                for (act, rew), erew in zip(act_rewards, exp_rew))
            return min(act_cumrewards, key = lambda e: e[1])


    def policy(self, obs):
        state = self._state_from_obs(obs)
        if self.goal_state is None:
            return self._exploration_policy(state)[0]
        else:
            return self._exploitation_policy(tuple(state.tolist()),
                                             tuple(self.goal_state.tolist()))[0]

    def _hit_goal(self, obs, act, rew, done, info):
        return info.get("hit_goal", False)

    def on_hit_goal(self, obs, act, rew):
        self.last_state = None

    def is_terminal_step(self, obs, act, rew, done, info):
        return done or info.get("new_spawn", False)

    def episode_reset(self, episode_n):
        self.goal_state = None
        self.last_state = None

    def update(self, obs, act, rew, done, info):
        self.step += 1
        # Protocol defined by: game.play:play_episode()
        # - act = alg.policy(obs)
        # - obs_plus_1, rew_plus_1 = the prob.step(act)
        # - the alg.update(obs, act, rew)
        # or
        # obs_m_1 --alg--> act --prob--> obs, rew
        # obs, rew = prob.step(action)
        if not self.observation_space.contains(obs):
            raise ValueError("Bad observation {obs}".format(obs=obs))
        # Encoding state_hash from observation
        st = self._state_from_obs(obs, act, rew)
        stm1 = self.last_state
        self.last_state = st
        if stm1 is None:
            return self.egreedy(self.policy(obs))

        if self._hit_goal(obs, act, rew, done, info):
            self.goal_state = stm1
            self.on_hit_goal(obs, act, rew)
            return self.egreedy(self.policy(obs))

        T = self.dynamics_model
        R = self.rewards
        T[stm1, act, st] += 1
        R[stm1, act] = rew

        return self.egreedy(self.policy(obs))


