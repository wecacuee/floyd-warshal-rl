from pathlib import Path
import numpy as np
from queue import PriorityQueue
import logging

import cog.draw as draw
from game.play import Space, Alg, NoOPObserver

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    def episode_reset(self, episode_n):
        self.action_value[:]= self.init_value
        self.last_state_idx_act = None

    def reset(self):
        self.rng            = np.random.RandomState()
        self.rng.seed(self._seed)
        self.action_value    = self._default_action_value(0)
        self.hash_state     = dict()
        self.episode_reset(0)

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

    def on_hit_goal(self, obs, act, rew):
        self.last_state_idx_act = None

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
            self.on_hit_goal(obs, act, rew)

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

class QLearningVis(NoOPObserver):
    def __init__(self, update_interval, cellsize, log_file_dir):
        self.update_interval = update_interval
        self.cellsize = cellsize
        self.log_file_dir = log_file_dir

        # State variables
        self.grid_shape = None
        self.goal_pose = None
        self._inverted_hash_state = None
        self.update_steps = 0
        self.nepisodes = 0
        super().__init__()

    @property
    def action_space(self):
        return self.alg.action_space
        
    def _invert_hash_state(self, hash_state):
        self._inverted_hash_state = { state_idx : state_pose
            for state_pose, state_idx in hash_state.items() }
        return self._inverted_hash_state

    def _state_idx_to_pose(self, hash_state, state_idx):
        return self._invert_hash_state(hash_state)[state_idx]

    def _action_value_to_mat(self, action_value, hash_state):
        mat = np.ones(self.grid_shape) * self.alg.init_value
        if action_value.size:
            for state_pose, state_idx in hash_state.items():
                mat[state_pose] = np.max(action_value[state_idx, :])
        return mat

    def _action_value_to_matrices(self, action_value, hash_state):
        big_mat = np.zeros(np.array(self.grid_shape) * 2)
        if self.action_space.size != 4:
            raise NotImplementedError("Do not know how to visualize")

        for act in range(4):
            mat = np.ones(self.grid_shape) * self.alg.init_value
            for state_pose, state_idx in hash_state.items():
                mat[state_pose] = action_value(state_idx, act) or mat[state_pose]
            big_r = act // 2
            big_c = act % 2
            big_mat[big_r::2, big_c::2] = mat
        return big_mat
        
        
    def visualize_action_value(self, ax, action_value, hash_state):
        cellsize = ax.get_xlim()[1] / self.grid_shape[1]
        action_value_mat = self._action_value_to_matrices(
            action_value, hash_state)
        action_value_mat += np.min(action_value_mat[:])
        draw.matshow(ax, self.normalize_by_std(action_value_mat))
        c50 = cellsize/2
        c25 = cellsize/4
        c75 = cellsize*3/4
        for i, j in np.ndindex(action_value_mat.shape):
            center = np.array((i*c50 + c25, j*c50 + c25))
            if i % 2 == 0 and j % 2 == 0:
                draw.rectangle(ax, center - c25, center + c75, (0, 0, 0))
            draw.putText(ax, f"{action_value_mat[i, j]:.3}",
                         center, fontScale=2)

    def _policy_to_mat(self, policy_func):
        return mat

    def visualize_policy(self, ax, policy_func, hash_state):
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

    def visualize(self, ax, action_value, hash_state, wait_time, cellsize):
        ax.set_position([0, 0, 0.5, 1])
        ax.clear()
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, self.grid_shape[1]*cellsize])
        ax.set_ylim([0, self.grid_shape[0]*cellsize])
        self.visualize_action_value(
            ax,
            lambda s, a: (action_value[s, a] if action_value.size else None),
            hash_state)
        ax2 = ax.figure.add_axes([0.5, 0, 0.5, 1], frameon=False)
        ax2.set_position([0.5, 0, 0.5, 1])
        ax2.clear()
        ax2.axis('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim([0, self.grid_shape[1]*cellsize])
        ax2.set_ylim([0, self.grid_shape[0]*cellsize])
        self.visualize_policy(ax2, self.alg.policy, hash_state)

    def normalize_by_std(self, mat):
        if not np.any(mat):
            return mat
        median_mat = np.median(mat[:])
        std_mat = np.std(mat[:])
        norm_mat = (mat - median_mat)
        if std_mat:
             norm_mat = norm_mat / (1.5*std_mat)
        norm_mat[norm_mat < -1] = -1
        norm_mat[norm_mat > +1] = +1
        return norm_mat

    def on_new_goal_pose(self, goal_pose):
        cellsize = self.cellsize
        ax = draw.white_img(
            (self.grid_shape[1]*cellsize, self.grid_shape[0]*2*cellsize),
            dpi=cellsize)
        self.visualize(ax, self.alg.action_value, self.alg.hash_state, 1, cellsize)
        self.goal_pose = goal_pose
        return ax

    def on_new_step(self, obs, act, rew):
        if not np.any(self.goal_pose == self.prob.goal_pose):
            self.on_new_goal_pose(self.prob.goal_pose)
            
        if self.update_steps % self.update_interval == 0:
            ax = self.on_new_goal_pose(self.goal_pose)
            draw.imwrite(
                str(
                    Path(self.log_file_dir) / "action_value_{episode}_{step}.pdf".format(
                        episode=self.nepisodes, step=self.update_steps)),
                ax)
        self.update_steps += 1

    def on_new_episode(self, n):
        self.grid_shape = self.prob.grid_shape
        self.goal_pose = self.prob.goal_pose
        self.nepisodes = n
        self.update_steps = 0

    def on_play_end(self):
        pass
