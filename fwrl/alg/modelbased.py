import math

import torch

from umcog.confutils import xargspartial, extended_kwprop

from .qlearning import egreedy_prob_exp
from ..game.play import Alg


def safe_div(num, den):
    return num / torch.where(den == 0, torch.ones_like(den), den)


def one_step_exp_reward(dynamics_model, rewards, st):
    """Returns one-step expected reward to state `st` from all states

    >>> dynamics = torch.zeros((3, 2, 3))
    >>> dynamics[0, 0, 2] = 1
    >>> dynamics[0, 1, 1] = 1
    >>> dynamics[2, 1, 1] = 1
    >>> rewards = torch.ones((3, 2))
    >>> rewards[0, 0] = -1e-4
    >>> rewards[0, 1] = -1e-4
    >>> rewards[2, 1] = 10
    >>> one_step_exp_reward(dynamics, rewards, 1)
    tensor([[-0.0000, -0.0001],
            [ 0.0000,  0.0000],
            [ 0.0000, 10.0000]])

    >>> rewards = torch.ones((3, 2))
    >>> rewards[0, 0] = -1e-4
    >>> rewards[0, 1] = 10
    >>> rewards[2, 1] = 10
    >>> one_step_exp_reward(dynamics, rewards, 1)
    tensor([[-0., 10.],
            [ 0.,  0.],
            [ 0., 10.]])
    """
    counts = dynamics_model[:, :, st]
    state_count = counts.sum(dim = -1, keepdim = True)
    prob = safe_div(counts, state_count)
    return (prob * rewards)


def default_value_fun(rewards, init):
    return rewards.new_ones(rewards.shape) * init


def out_neighbors(dynamics_model, state, action):
    state_to_goal_prob = dynamics_model[state, action, :]
    nbr_states = state_to_goal_prob.nonzero().squeeze(-1)
    return nbr_states


def in_neighbors(dynamics_model, goal_state):
    state_to_goal_prob = dynamics_model[:, :, goal_state].max(dim=-1)[0]
    nbr_states = state_to_goal_prob.nonzero().squeeze(-1)
    return nbr_states


def exp_action_value(dynamics_model, rewards, state, action, goal_state,
                     value_fun = None,
                     visited = None,
                     value_fun_gen = default_value_fun,
                     value_fun_init = float('-inf'),
                     discount = 1):
    """Returns the best action and expected reward

    >>> dynamics = torch.zeros((3, 2, 3))
    >>> dynamics[0, 0, 2] = 1
    >>> dynamics[2, 1, 1] = 1
    >>> dynamics[0, 1, 1] = 1
    >>> dynamics[2, 1, 1] = 1
    >>> rewards = torch.ones((3, 2))
    >>> rewards[0, 0] = -1e-4
    >>> rewards[0, 1] = -1e-4
    >>> rewards[2, 1] = 10
    >>> exp_action_value(dynamics, rewards, 0, 0, 1)
    tensor(9.9999)
    >>> exp_action_value(dynamics, rewards, 0, 1, 1)
    tensor(-0.0001)

    >>> rewards = torch.ones((3, 2))
    >>> rewards[0, 0] = -1e-4
    >>> rewards[0, 1] = 10
    >>> rewards[2, 1] = 10
    >>> exp_action_value(dynamics, rewards, 0, 0, 1)
    tensor(9.9999)
    >>> exp_action_value(dynamics, rewards, 0, 1, 1)
    tensor(10.)
    """
    nstates = dynamics_model.shape[0]
    nactions = dynamics_model.shape[1]
    assert nstates == dynamics_model.shape[2]
    assert dynamics_model.dim() == 3
    assert nstates == rewards.shape[0]
    assert nactions == rewards.shape[1]
    assert rewards.dim() == 2

    if value_fun is None:
        value_fun = value_fun_gen(rewards, value_fun_init)

    if visited is None:
        visited = set()
    visited.add(state)

    # Memoization
    if value_fun[state, action] != value_fun_init:
        return value_fun[state, action]

    # Exp_action_value algorithm
    if state == goal_state:
        # zero step
        return rewards.new_zeros(1)
    else:
        # There is a trade off in choosing expected reward vs max reward.
        nbr_states = out_neighbors(dynamics_model, state, action)
        nbr_states_set = set(nbr_states.tolist()) - visited
        if not len(nbr_states_set):
            # we have run into a dead end
            # Do not prefer this one
            return rewards.new_ones(1) * float('-inf')

        for nbr_st in nbr_states_set:
            for a in torch.arange(nactions):
                rew = exp_action_value(dynamics_model, rewards, nbr_st, a,
                                       goal_state, value_fun = value_fun,
                                       visited = visited)
                assert not math.isnan(rew)
                # Memoize
                value_fun[nbr_st, a] = rew

        nbr_states_t = torch.tensor(list(nbr_states_set))
        counts = dynamics_model[state, action, nbr_states_t]
        probs = safe_div(counts, counts.sum(dim = -1, keepdim = True)) + 1e-8
        # -inf * 0 -> nan; Handle nan
        future_rewards = (value_fun[nbr_states_t, :].max(dim = -1)[0] * probs)
        max_cumrew = (rewards[state, action] + future_rewards.sum())
        assert not math.isnan(max_cumrew)

        # Memoize
        value_fun[state, action] = max_cumrew
        return max_cumrew


class ModelBasedTabular(Alg):
    egreedy_prob_exp = egreedy_prob_exp

    @extended_kwprop
    def __init__(self,
                 action_space,
                 observation_space,
                 reward_range,
                 rng,
                 start_eps = 0.9, end_eps = 0.01, alpha = -0.1,
                 egreedy_prob = xargspartial(
                     egreedy_prob_exp,
                     dict(start_eps="start_eps",
                          end_eps="end_eps",
                          alpha="alpha",
                          nepisodes="max_steps")),
                 discount = 1.00):  # step cost
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = reward_range
        self.rng = rng
        self.egreedy_prob = egreedy_prob
        print("egreedy_prob {}".format(egreedy_prob.keywords))
        self.discount = discount
        self.reset()

    def reset(self):
        self.step = 0
        self.episode_n = 0
        self.hash_state = dict()
        self.dynamics_model = self._defaut_dynamics(0)
        self.rewards = self._default_rewards(0)

    def _default_rewards(self, state_size):
        return torch.ones((state_size, self.action_space.n))

    def _resize_rewards(self, new_size):
        olds = self.rewards.shape[0]
        newrews = self.rewards.new_ones((new_size, self.action_space.n))
        newrews[:olds, :] = self.rewards
        self.rewards = newrews
        assert not torch.isnan(self.rewards).any()
        return self.rewards

    def _defaut_dynamics(self, state_size):
        return torch.zeros((state_size, self.action_space.n, state_size))

    def _resize_dynamics(self, new_size):
        olds = self.dynamics_model.shape[0]
        dynm = self.dynamics_model.new_zeros((new_size, self.action_space.n, new_size))
        dynm[:olds, :, :olds] = self.dynamics_model
        self.dynamics_model = dynm
        assert not torch.isnan(self.dynamics_model).any()
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
        r = self.rng.rand()
        return_greedy = (r >= self.egreedy_prob(self.episode_n))
        rnd_act = self.action_space.sample()
        ret_act = greedy_act if return_greedy else rnd_act
        return ret_act

    def _exploration_policy(self, state):
        return self.action_space.sample()

    def _exploitation_policy(self, state, goal_state):
        act, exp_rew = max(
            [(act, exp_action_value(self.dynamics_model, self.rewards, state,
                                    act, goal_state))
             for act in range(self.action_space.n)],
            key = lambda x: x[1])
        return act, exp_rew

    def policy(self, obs):
        state = self._state_from_obs(obs)
        if self.goal_state is None:
            return self._exploration_policy(state)
        else:
            act, exp_rew = self._exploitation_policy(state, self.goal_state)
            if math.isinf(exp_rew) or math.isnan(exp_rew):
                return self._exploration_policy(state)
            else:
                print("exploitation, exp_rew {}".format(exp_rew))
                return act

    def _hit_goal(self, obs, act, rew, done, info):
        return info.get("hit_goal", False)

    def on_hit_goal(self, obs, act, rew):
        self.last_state = None

    def is_terminal_step(self, obs, act, rew, done, info):
        return done or info.get("new_spawn", False)

    def episode_reset(self, episode_n):
        self.step = 0
        self.episode_n = episode_n
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
        st = self._state_from_obs(obs)
        stm1 = self.last_state
        self.last_state = st
        if stm1 is None:
            return self.egreedy(self.policy(obs))
        print("{stm1} -- {act} --> {st}, {rew}".format(stm1 = stm1, act = act,
                                                       st = st, rew = rew))

        if self._hit_goal(obs, act, rew, done, info):
            self.goal_state = stm1
            self.on_hit_goal(obs, act, rew)
            return self.egreedy(self.policy(obs))

        T = self.dynamics_model
        R = self.rewards
        T[stm1, act, st] += 1
        R[stm1, act] = rew

        return self.egreedy(self.policy(obs))

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False

