from game.interface import Space, Alg
import numpy as np
from queue import PriorityQueue
import cog.draw as draw

class FloydWarshallAlgDiscrete(Alg):
    def __init__(self,
                 action_space,
                 observation_space,
                 seed,
                 egreedy_epsilon,
                 path_cost_momentum,
                 action_value_momentum,
                 init_value,
                 top_value_queue_size,
                 per_edge_reward
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self.seed                 = seed
        self.egreedy_epsilon      = egreedy_epsilon
        self.path_cost_momentum   = path_cost_momentum
        self.action_value_momentum = action_value_momentum
        self.init_value           = init_value
        self.top_value_queue_size = top_value_queue_size
        self.per_edge_reward      = per_edge_reward
        self.reset()

    def episode_reset(self):
        self.action_value[:]    = 0
        self.top_m_states   = PriorityQueue(maxsize=self.top_value_queue_size)
        self.last_state_idx = None

    def reset(self):
        self.rng            = np.random.RandomState()
        self.rng.seed(self.seed)
        self.path_cost     = self._default_path_cost(
            (0, 0))
        self.action_value    = self._default_action_value(0)
        self.hash_state     = dict()
        self.episode_reset()

    def _default_path_cost(self, shape):
        path_cost = -self.per_edge_reward * np.ones(shape)
        # make staying at the same place very expensive
        np.fill_diagonal(path_cost, 100)
        return path_cost

    def _default_action_value(self, state_size):
        return self.init_value * np.ones((state_size, self.action_space.size))

    def _resize_path_cost(self, new_state_size):
        new_path_cost = self._default_path_cost(
            (new_state_size, new_state_size))
        if self.path_cost.size:
            new_path_cost[:self.path_cost.shape[0],
                           :self.path_cost.shape[1]] = self.path_cost
        return new_path_cost

    def _resize_action_value(self, new_size):
        new_action_value = self._default_action_value(new_size)
        if self.action_value.size:
            new_action_value[:self.action_value.shape[0], :] = self.action_value
        return new_action_value
    
    def egreedy(self, greedy_act):
        sample_greedy = (self.rng.rand() >= self.egreedy_epsilon)
        return self.action_space.sample() if sample_greedy else greedy_act

    def _state_from_obs(self, obs):
        state = obs # fully observed system
        return state

    def _state_idx_from_obs(self, obs):
        state = tuple(self._state_from_obs(obs))
        if state not in self.hash_state:
            # A new state has been visited
            state_idx = self.hash_state[state] = max(
                self.hash_state.values(), default=-1) + 1
            self.path_cost = self._resize_path_cost(state_idx + 1)
            self.action_value = self._resize_action_value(state_idx + 1)
        else:
            state_idx = self.hash_state[state]
        return state_idx

    def _value(self, state_s, act_s, state_g):
        return (self.path_cost[state_s, act_s, state_g]
                + self.action_value[state_g])

    def policy(self, obs):
        state = self._state_from_obs(obs)
        state_idx = self.hash_state[tuple(state)]
        # desirable_dest = max(
        #     self.top_m_states.queue,
        #     key = lambda s: self.action_value[s[1]])[1]
        return np.argmax(self.action_value[state_idx, :])

    def state_pairs_iter(self):
        for si in self.hash_state.values():
            for sj in self.hash_state.values():
                yield (si, sj)

    def update(self, obs, act, rew):
        if not self.observation_space.contains(obs):
            raise ValueError(f"Bad observation {obs}")

        # Encoding state_hash from observation
        state_idx = self._state_idx_from_obs(obs)
        if self.last_state_idx is None:
            self.last_state_idx = state_idx
            return

        # Abbreviate the variables
        pm = self.path_cost_momentum
        qm = self.action_value_momentum
        F = self.path_cost
        Q = self.action_value
        stm1 = self.last_state_idx
        st = state_idx

        # Update step from online observed reward
        F[stm1, st] = (1-pm)* (-rew) + pm * F[stm1, st]
        Q[stm1, act] = (1-qm) * (rew + np.max(Q[st, :])) + qm * Q[stm1, act]

        # # TODO: Disabled for small state spaces. Re-enable for larger ones
        # # Update the top m states
        # if not self.top_m_states.full():
        #     self.top_m_states.put((self.action_value[state_idx], state_idx))

        # top_value, top_state_idx = self.top_m_states.get()
        # if self.action_value[state_idx] > top_value:
        #     self.top_m_states.put((self.action_value[state_idx], state_idx))
        # else:
        #     self.top_m_states.put((top_value, top_state_idx))

        # Update step from actual Floyd Warshall algorithm
        # This is an expensive step depending up on the number of states
        # Linear in the number of states

        #for (si, sj) in self.state_pairs_iter():
        # O(n^2) step to update all path_costs and action_values
        self.path_cost = np.minimum(
            self.path_cost,
            self.path_cost[..., -1:] + self.path_cost[-1:, ...])
        V = np.max(self.action_value, axis=1)
        self.action_value = np.maximum(
            self.action_value,
            np.max(V - self.path_cost, axis=1, keepdims=True))

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False

class FloydWarshallVisualizer(object):
    def __init__(self, fwalg, goal_pose, grid_shape):
        self.fwalg = fwalg
        self.goal_pose = goal_pose
        self.grid_shape = grid_shape
        self._inverted_hash_state = None
        
    def _invert_hash_state(self, hash_state):
        self._inverted_hash_state = { state_idx : state_pose
            for state_pose, state_idx in hash_state.items() }
        return self._inverted_hash_state

    def _state_idx_to_pose(self, hash_state, state_idx):
        return self._invert_hash_state(hash_state)[state_idx]

    def _path_cost_to_mat(self, path_cost, hash_state):
        mat = np.zeros(self.grid_shape)
        if path_cost.size:
            goal_state_idx = hash_state.get(tuple(self.goal_pose), 0)
            for state_pose, state_idx in hash_state.items():
                mat[state_pose] = path_cost[state_idx, goal_state_idx]
        return mat

    def _action_value_to_mat(self, action_value, hash_state):
        mat = np.zeros(self.grid_shape)
        if action_value.size:
            for state_pose, state_idx in hash_state.items():
                mat[state_pose] = np.max(action_value[state_idx, :])
        return mat
        
    def visualize_path_cost(self, ax, path_cost, hash_state, wait_time):
        if ax is None:
            ax = draw.white_img(self.grid_shape)
        path_cost_mat = self._path_cost_to_mat(path_cost, hash_state)
        ax.matshow(path_cost_mat)
        for i, j in np.ndindex(path_cost_mat.shape):
            draw.putText(ax, f"{path_cost_mat[i, j]}",
                         np.array((i*100, j*100)), None, None,
                         (0, 0, 0))
        draw.imshow("path_cost", ax)

    def visualize_action_value(self, ax, action_value, hash_state, wait_time):
        if ax is None:
            ax = draw.white_img(self.grid_shape)
        ax.matshow(self._action_value_to_mat(action_value, hash_state))
        draw.imshow("action_value", ax)

    def visualize(self, action_value, path_cost, hash_state, wait_time):
        self.visualize_action_value(None, action_value, hash_state, wait_time)
        self.visualize_path_cost(None, path_cost, hash_state, wait_time)

    def update(self, obs, act, rew):
        self.fwalg.update(obs, act, rew)
        self.visualize(self.fwalg.action_value,
                       self.fwalg.path_cost, self.fwalg.hash_state, 1)

    def __getattr__(self, attr):
        return getattr(self.fwalg, attr)

