from game.interface import Space, Alg
import numpy as np
from queue import PriorityQueue

class FloydWarshallAlgDiscrete(Alg):
    def __init__(self,
                 action_space,
                 observation_space,
                 seed,
                 egreedy_epsilon,
                 path_value_momentum,
                 state_value_momentum,
                 init_value,
                 top_value_queue_size,
                 per_edge_reward
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self.seed                 = seed
        self.egreedy_epsilon      = egreedy_epsilon
        self.path_value_momentum  = path_value_momentum
        self.state_value_momentum = state_value_momentum
        self.init_value           = init_value
        self.top_value_queue_size = top_value_queue_size
        self.per_edge_reward      = per_edge_reward
        self.reset()

    def episode_reset(self):
        self.state_value[:]    = 0
        self.top_m_states   = PriorityQueue(maxsize=self.top_value_queue_size)
        self.last_state_idx = None

    def reset(self):
        self.rng            = np.random.RandomState()
        self.rng.seed(self.seed)
        self.path_value     = self._default_path_value(
            (0, self.action_space.size, 0))
        self.state_value    = self._default_state_value((0,))
        self.hash_state     = dict()
        self.episode_reset()

    def _default_path_value(self, shape):
        return -0.1 * np.ones(shape)

    def _default_state_value(self, shape):
        return self.init_value * np.ones(shape)

    def _resize_path_value(self, new_state_size):
        new_path_value = self._default_path_value(
            (new_state_size, self.action_space.size, new_state_size))
        if self.path_value.size:
            new_path_value[:self.path_value.shape[0],
                           : ,
                           :self.path_value.shape[2]] = self.path_value
        return new_path_value

    def _resize_state_value(self, new_size):
        new_state_value = self._default_state_value((new_size, ))
        if self.state_value.size:
            new_state_value[:self.state_value.shape[0]] = self.state_value
        return new_state_value
    
    def egreedy(self, greedy_act):
        sample_egreedy = (self.rng.rand() <= self.egreedy_epsilon)
        return self.action_space.sample() if sample_egreedy else greedy_act

    def _state_from_obs(self, obs):
        state = obs # fully observed system
        return state

    def _state_idx_from_obs(self, obs):
        state = tuple(self._state_from_obs(obs))
        if state not in self.hash_state:
            # A new state has been visited
            state_idx = self.hash_state[state] = max(
                self.hash_state.values(), default=-1) + 1
            self.path_value = self._resize_path_value(state_idx + 1)
            self.state_value = self._resize_state_value(state_idx + 1)
        else:
            state_idx = self.hash_state[state]
        return state_idx

    def _value(self, state_s, act_s, state_g):
        return self.path_value[state_s, act_s, state_g] + self.state_value[state_g]

    def policy(self, obs):
        state = self._state_from_obs(obs)
        state_idx = self.hash_state[tuple(state)]
        value_per_action = lambda a : max(
            self.top_m_states.queue, key = lambda s: self._value(state_idx, a, s[1]))
        return max(self.action_space.values(), key = value_per_action)

    def update(self, obs, act, rew):
        if not self.observation_space.contains(obs):
            raise ValueError(f"Bad observation {obs}")

        # Encoding state_hash from observation
        state_idx = self._state_idx_from_obs(obs)

        # Update step from online observed reward
        pm = self.path_value_momentum
        vm = self.state_value_momentum
        if self.last_state_idx is not None:
            pv = self.path_value[self.last_state_idx, act, state_idx] = (
                (1-pm) * rew + pm * self.path_value[
                    self.last_state_idx, act, state_idx] + self.per_edge_reward)
        else:
            pv = 0

        self.state_value[state_idx] = (
            (1-vm)*max(0, rew - pv) + vm * self.state_value[state_idx])

        # Update the top m states
        if not self.top_m_states.full():
            self.top_m_states.put((self.state_value[state_idx], state_idx))

        top_value, top_state_idx = self.top_m_states.get()
        if self.state_value[state_idx] > top_value:
            self.top_m_states.put((self.state_value[state_idx], state_idx))
        else:
            self.top_m_states.put((top_value, top_state_idx))

        # Update step from Floyd Warshall algorithm
        # This is an expensive step depending up on the number of states
        self.path_value = np.maximum(
            self.path_value,
            self.path_value[..., -1:] + self.path_value[-1:, :, :])
        self.last_state_idx = state_idx

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False


