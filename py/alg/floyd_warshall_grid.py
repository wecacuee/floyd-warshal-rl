from game.interface import Space, Alg
import numpy as np
from queue import PriorityQueue
import cog.draw as draw
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
                 discount
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self.seed                 = seed
        self.egreedy_epsilon      = egreedy_epsilon
        self.path_cost_momentum   = path_cost_momentum
        self.action_value_momentum= action_value_momentum
        self.init_value           = init_value
        self.top_value_queue_size = top_value_queue_size
        self.per_edge_cost        = init_value / 20
        self.discount             = discount
        self.reset()

    def episode_reset(self):
        self.action_value[:]= self.init_value
        self.top_m_states   = PriorityQueue(maxsize=self.top_value_queue_size)
        self.last_state_idx_act = None

    def reset(self):
        self.rng            = np.random.RandomState()
        self.rng.seed(self.seed)
        self.path_cost     = self._default_path_cost(0)
        self.action_value    = self._default_action_value(0)
        self.hash_state     = dict()
        self.episode_reset()

    def _default_path_cost(self, new_state_size):
        shape = (new_state_size, self.action_space.size, new_state_size)
        path_cost = 10 * np.ones(shape)
        # make staying at the same place very expensive
        # for act in self.action_space.values():
        #     np.fill_diagonal(path_cost[:, act, :], 100)
        return path_cost

    def _default_action_value(self, state_size):
        return self.init_value * np.ones((state_size, self.action_space.size))

    def _resize_path_cost(self, new_state_size):
        new_path_cost = self._default_path_cost(new_state_size)
        if self.path_cost.size:
            new_path_cost[:self.path_cost.shape[0],
                          :self.path_cost.shape[1],
                          :self.path_cost.shape[2],
            ] = self.path_cost
        return new_path_cost

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
            self.path_cost = self._resize_path_cost(state_idx + 1)
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

    def state_pairs_iter(self):
        for si in self.hash_state.values():
            for sj in self.hash_state.values():
                yield (si, sj)

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
            #import pdb; pdb.set_trace()
            self.last_state_idx_act = None

        # Abbreviate the variables
        pm = self.path_cost_momentum
        qm = self.action_value_momentum
        F = self.path_cost
        Q = self.action_value

        # Update step from online observed reward
        Q[stm1, act] = (1-qm) * (
            (rew-self.per_edge_cost) + np.max(Q[st, :])
        ) + qm * Q[stm1, act]
        # Make a conservative estimate of differential
        F[stm1, act, st] = self.per_edge_cost
        F[:, :, st] = np.minimum(F[:, :, st], F[:, :, stm1] + F[stm1, act, st])

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
            F,
            F[:, :, st:st+1] + np.min(F[st:st+1, :, :], axis=1, keepdims=True))
        V = np.max(Q, axis=-1)
        self.action_value = np.maximum(
            Q,
            np.max(V[None, :] - self.path_cost , axis=-1))

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False

class FloydWarshallVisualizer(object):
    def __init__(self, fwalg, prob, grid_shape):
        if not isinstance(fwalg, FloydWarshallAlgDiscrete):
            raise NotImplementedError()
        self.fwalg = fwalg
        self.prob = prob
        self.grid_shape = grid_shape
        self._inverted_hash_state = None
        self.update_steps = 0

    @property
    def goal_pose(self):
        return self.prob.goal_pose
        
    def _invert_hash_state(self, hash_state):
        self._inverted_hash_state = { state_idx : state_pose
            for state_pose, state_idx in hash_state.items() }
        return self._inverted_hash_state

    def _state_idx_to_pose(self, hash_state, state_idx):
        return self._invert_hash_state(hash_state)[state_idx]

    def _path_cost_to_mat(self, path_cost, hash_state):
        big_mat = np.zeros(np.array(self.grid_shape) * 2)
        if self.action_space.size != 4:
            raise NotImplementedError("Do not know how to visualize")
        for act in range(4):
            mat = np.zeros(self.grid_shape)
            if path_cost.size:
                goal_state_idx = hash_state.get(tuple(self.goal_pose), -1)
                for state_pose, state_idx in hash_state.items():
                    mat[state_pose] = path_cost[state_idx, act, goal_state_idx]
            big_r = act // 2
            big_c = act % 2
            big_mat[big_r::2, big_c::2] = mat
        return big_mat

    def _action_value_to_mat(self, action_value, hash_state):
        mat = np.ones(self.grid_shape) * self.fwalg.init_value
        if action_value.size:
            for state_pose, state_idx in hash_state.items():
                mat[state_pose] = np.max(action_value[state_idx, :])
        return mat

    def _action_value_to_matrices(self, action_value, hash_state):
        big_mat = np.zeros(np.array(self.grid_shape) * 2)
        if self.action_space.size != 4:
            raise NotImplementedError("Do not know how to visualize")

        for act in range(4):
            mat = np.ones(self.grid_shape) * self.fwalg.init_value
            if action_value.size:
                for state_pose, state_idx in hash_state.items():
                    mat[state_pose] = action_value[state_idx, act]
            big_r = act // 2
            big_c = act % 2
            big_mat[big_r::2, big_c::2] = mat
        return big_mat
        
        
    def visualize_path_cost(self, ax, path_cost, hash_state):
        if ax is None:
            ax = draw.white_img(np.array(self.grid_shape)*100)
        path_cost_mat = self._path_cost_to_mat(path_cost, hash_state)
        max_val = np.max(path_cost_mat[:])
        second_max = np.max(path_cost_mat[path_cost_mat != max_val])
        path_cost_mat[path_cost_mat == max_val] = second_max
        draw.matshow(ax, path_cost_mat)
        for i, j in np.ndindex(path_cost_mat.shape):
            cellsize = ax.get_xlim()[1] / path_cost_mat.shape[1]
            center = (np.array((i, j)) + 0.5) * cellsize
            if i % 2 == 0 and j % 2 == 0:
                draw.rectangle(ax, center - cellsize/2, center + cellsize*3/2,
                               (0, 0, 0))
            draw.putText(ax, f"{path_cost_mat[i, j]:.3}",
                         center, fontScale=2)

    def visualize_action_value(self, ax, action_value, hash_state):
        if ax is None:
            ax = draw.white_img(np.array(self.grid_shape) * 100)
        cellsize = ax.get_xlim()[1] / self.grid_shape[0]
        action_value_mat = self._action_value_to_matrices(
            action_value, hash_state)
        action_value_mat += np.min(action_value_mat[:])
        draw.matshow(ax, action_value_mat)
        for i, j in np.ndindex(action_value_mat.shape):
            center = np.array((i*50 + 25, j*50 + 25))
            if i % 2 == 0 and j % 2 == 0:
                draw.rectangle(ax, center - 25, center + 75, (0, 0, 0))
            draw.putText(ax, f"{action_value_mat[i, j]:.3}",
                         center, fontScale=2)

    def _policy_to_mat(self, policy_func):
        return mat

    def visualize_policy(self, ax, policy_func, hash_state):
        if ax is None:
            ax = draw.white_img(np.asarray(self.grid_shape) * cellsize)
        cellsize = ax.get_xlim()[1] / self.grid_shape[0]

        for state in hash_state.keys():
            act = policy_func(state)
            act_vec = self.action_space.tovector(act)
            center = (np.array(state) + 0.5) * cellsize
            draw.arrowedLine(ax,
                             center - act_vec * cellsize / 4,
                             center + act_vec * cellsize / 4,
                             (0, 0, 0),
                             thickness=5,
                             tipLength=10)

    def visualize(self, action_value, path_cost, hash_state, wait_time):
        cellsize = 100
        ax = draw.white_img(
            (self.grid_shape[1]*cellsize, self.grid_shape[0]*3*cellsize),
            dpi=4*cellsize)
        ax.set_position([0, 0, 0.33, 1])
        ax.clear()
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, self.grid_shape[1]*cellsize])
        ax.set_ylim([0, self.grid_shape[0]*cellsize])
        self.visualize_action_value(ax, action_value, hash_state)
        ax2 = ax.figure.add_axes([0.33, 0, 0.33, 1], frameon=False)
        ax2.set_position([0.33, 0, 0.33, 1])
        ax2.clear()
        ax2.axis('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim([0, self.grid_shape[1]*cellsize])
        ax2.set_ylim([0, self.grid_shape[0]*cellsize])
        self.visualize_path_cost(ax2, path_cost, hash_state)
        ax3 = ax.figure.add_axes([0.67, 0, 0.33, 1], frameon=False)
        ax3.set_position([0.67, 0, 0.33, 1])
        ax3.clear()
        ax3.axis('equal')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_xlim([0, self.grid_shape[1]*cellsize])
        ax3.set_ylim([0, self.grid_shape[0]*cellsize])
        self.visualize_policy(ax3, self.policy, hash_state)
        draw.imshow("action_value", ax)
        draw.waitKey(1)


    def update(self, obs, act, rew):
        self.fwalg.update(obs, act, rew)
        if self.update_steps % 20 == 0:
            self.visualize(self.fwalg.action_value,
                        self.fwalg.path_cost, self.fwalg.hash_state, 1)
        self.update_steps += 1

    def __getattr__(self, attr):
        return getattr(self.fwalg, attr)

