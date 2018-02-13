from game.play import Space, Alg, NoOPObserver
import numpy as np
from queue import PriorityQueue
import cog.draw as draw
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QLearningDiscrete(Alg):
    def __init__(self,
                 action_space,
                 observation_space,
                 seed,
                 egreedy_epsilon,
                 action_value_momentum,
                 init_value,
                 discount
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self._seed                = seed
        self.egreedy_epsilon      = egreedy_epsilon
        self.action_value_momentum= action_value_momentum
        self.init_value           = init_value
        self.discount             = discount
        self.reset()

    def episode_reset(self):
        self.action_value[:]= self.init_value
        self.last_state_idx_act = None

    def reset(self):
        self.rng            = np.random.RandomState()
        self.rng.seed(self._seed)
        self.action_value    = self._default_action_value(0)
        self.hash_state     = dict()
        self.episode_reset()

    def _default_action_value(self, state_size):
        return self.init_value * np.ones((state_size, self.action_space.size))

    def _resize_action_value(self, new_size):
        new_action_value = self._default_action_value(new_size)
        if self.action_value.size:
            new_action_value[:self.action_value.shape[0], :] = self.action_value
        return new_action_value

    def egreedy(self, greedy_act):
        sample_greedy = (self.rng.rand() >= self.egreedy_epsilon)
        return greedy_act if sample_greedy else self.action_space.sample()

    def _state_from_obs(self, obs):
        state = obs # fully observed system
        return state

    def _state_idx_from_obs(self, obs, act, rew):
        state = tuple(self._state_from_obs(obs))
        if state not in self.hash_state:
            # A new state has been visited
            state_idx = self.hash_state[state] = max(
                self.hash_state.values(), default=-1) + 1
            self.action_value = self._resize_action_value(state_idx + 1)
        else:
            state_idx = self.hash_state[state]
        return state_idx

    def _value(self, state_s, act_s, state_g):
        return (self.path_cost[state_s, act, state_g]
                + self.action_value[state_g])

    def policy(self, obs):
        state = self._state_from_obs(obs)
        state_idx = self.hash_state[tuple(state)]
        # desirable_dest = max(
        #     self.top_m_states.queue,
        #     key = lambda s: self.action_value[s[1]])[1]
        logger.debug(
            f"state = {state}; action_values = {self.action_value[state_idx, :]}")
        return np.argmax(self.action_value[state_idx, :])

    def _hit_goal(self, rew):
        return rew >= 9

    def update(self, obs, act, rew):
        if not self.observation_space.contains(obs):
            raise ValueError(f"Bad observation {obs}")

        # Encoding state_hash from observation
        state_idx = self._state_idx_from_obs(obs, act, rew)
        stm1, am1 = self.last_state_idx_act or (None, None)
        st = state_idx
        self.last_state_idx_act = state_idx, act
        if stm1 is None:
            return

        if self._hit_goal(rew):
            self.last_state_idx_act = None

        # Abbreviate the variables
        qm = self.action_value_momentum
        Q = self.action_value
        d = self.discount

        # Update step from online observed reward
        Q[stm1, act] = (1-qm) * (rew + d * np.max(Q[st, :])) + qm * Q[stm1, act]

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False
