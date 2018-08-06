import os
import functools
import logging
from collections import namedtuple

import numpy as np

import umcog.draw as draw
from ..game.play import Space, Alg, NoOPObserver
from .qlearning import (QLearningDiscrete, QLearningVis,
                        post_process_from_log_conf as ql_post_process_from_log_conf,
                        post_process_data_iter, post_process_generic)
from umcog.confutils import xargs, xargspartial, xargmem, KWProp as prop, extended_kwprop

def logger():
    return logging.getLogger(__name__)

class FloydWarshallAlgDiscrete(object):
    @extended_kwprop
    def __init__(self,
                 qlearning = xargs(
                     QLearningDiscrete,
                     "action_space observation_space reward_range rng".split()), 
    ):
        self.qlearning            = qlearning
        self.reset()

    @property
    def per_edge_cost(self):
        return 10 * self.qlearning.reward_range[1] * (1-self.qlearning.discount)

    @property
    def path_cost_init(self):
        return 10 * self.qlearning.reward_range[1]

    def episode_reset(self, episode_n):
        self.qlearning.episode_reset(episode_n)

    def reset(self):
        self.qlearning.reset()
        self.path_cost     = self._default_path_cost(0)

    def _default_path_cost(self, new_state_size):
        shape = (new_state_size, self.action_space.size, new_state_size)
        path_cost = self.path_cost_init * np.ones(shape)
        return path_cost

    def _resize_path_cost(self, new_state_size):
        new_path_cost = self._default_path_cost(new_state_size)
        if self.path_cost.size:
            new_path_cost[
                tuple(map(lambda s : slice(None, s),
                          self.path_cost.shape))] = self.path_cost
        return new_path_cost

    def _state_idx_from_obs(self, obs, act, rew):
        state_idx = self.qlearning._state_idx_from_obs(obs, act, rew)
        if state_idx >= self.path_cost.shape[0]:
            self.path_cost = self._resize_path_cost(state_idx + 1)
        return state_idx

    def update(self, obs, act, rew):
        stm1, am1 = self.last_state_idx_act or (None, None)
        st = self._state_idx_from_obs(obs, act, rew)
        self.qlearning.update(obs, act, rew)
        if stm1 is None:
            return

        # Abbreviate the variables
        F = self.path_cost
        Q = self.qlearning.action_value

        # Make a conservative estimate of differential
        F[stm1, act, st] = max(np.max(Q[st, :]) - Q[stm1, act],
                               self.per_edge_cost)
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
        assert np.all(self.path_cost >= 0), "The Floyd cost should be positive at all times"

    def net_value(self, state_idx):
        Q = self.qlearning.action_value
        V = np.max(Q, axis=-1)
        state_action_values = np.maximum(
            Q[state_idx, :],
            np.max(V[None, :] - self.path_cost[state_idx, :, :] , axis=-1))
        return state_action_values

    def policy(self, obs):
        state = self._state_from_obs(obs)
        state_idx = self.hash_state[tuple(state)]
        return np.argmax(self.net_value(state_idx))

    def __getattr__(self, attr):
        if attr in """action_space done action_value hash_state
                      grid_shape init_value last_state_idx_act egreedy
                      _state_from_obs""".split():
            return getattr(self.qlearning, attr)
        else:
            raise AttributeError(f"No attribute {attr}")

class FloydWarshallVisualizer(QLearningVis):
    def _path_cost_to_mat(self, path_cost, hash_state, goal_state_idx, grid_shape):
        big_mat = np.zeros(np.array(grid_shape) * 2)
        for act in range(4):
            mat = np.zeros(grid_shape)
            if path_cost.size:
                for state_pose, state_idx in hash_state.items():
                    mat[state_pose] = path_cost[state_idx, act, goal_state_idx]
            big_r = act // 2
            big_c = act % 2
            big_mat[big_r::2, big_c::2] = mat
        return big_mat

    def visualize_path_cost(self, ax, path_cost, hash_state, goal_pose, grid_shape):
        if not len(hash_state):
            return
        vis_goal_pose = np.asarray(goal_pose
                         if tuple(goal_pose) in hash_state
                         else list(hash_state.keys())[-1])
        goal_state_idx = hash_state[tuple(vis_goal_pose)]
        path_cost_mat = self._path_cost_to_mat(path_cost, hash_state,
                                               goal_state_idx, grid_shape)
        max_val = np.max(path_cost_mat[:])
        if max_val > 0:
            second_max = np.max(path_cost_mat[path_cost_mat != max_val])
            path_cost_mat[path_cost_mat == max_val] = second_max
        draw.matshow(ax, self.normalize_by_std(path_cost_mat))
        cellsize = ax.get_xlim()[1] / path_cost_mat.shape[1]
        draw.rectangle(ax, vis_goal_pose * 2*cellsize,
                       (vis_goal_pose + 1)*2*cellsize,
                       (0, 255, 0), thickness=-1)
        for i, j in np.ndindex(path_cost_mat.shape):
            center = (np.array((i, j)) + 0.5) * cellsize
            if i % 2 == 0 and j % 2 == 0:
                draw.rectangle(ax, center - cellsize/2, center + cellsize*3/2,
                               (0, 0, 0))
            draw.putText(ax, f"{path_cost_mat[i, j]:.3}",
                         center, fontScale=4)

    def visualize_all(self, ax, action_value, policy, path_cost,
                      net_value, hash_state, grid_shape, goal_pose,
                      cellsize):
        ax = draw.white_img(
            (grid_shape[1]*2*cellsize, grid_shape[0]*2*cellsize),
            dpi=cellsize*2)
        ax.set_position([0, 0, 0.5, 0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, grid_shape[1]*cellsize])
        ax.set_ylim([0, grid_shape[0]*cellsize])
        self.visualize_action_value(ax, action_value, hash_state, grid_shape)

        ax2 = ax.figure.add_axes([0.5, 0, 0.5, 0.5], frameon=False)
        ax2.set_position([0.5, 0, 0.5, 0.5])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim([0, grid_shape[1]*cellsize])
        ax2.set_ylim([0, grid_shape[0]*cellsize])
        self.visualize_policy(ax2, policy, hash_state, grid_shape)

        ax3 = ax.figure.add_axes([0, 0.5, 0.5, 0.5], frameon=False)
        ax3.set_position([0, 0.5, 0.5, 0.5])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_xlim([0, grid_shape[1]*cellsize])
        ax3.set_ylim([0, grid_shape[0]*cellsize])
        self.visualize_path_cost(ax3, path_cost, hash_state, goal_pose, grid_shape)

        ax4 = ax.figure.add_axes([0.5, 0.5, 0.5, 0.5], frameon=False)
        ax4.set_position([0.5, 0.5, 0.5, 0.5])
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_xlim([0, grid_shape[1]*cellsize])
        ax4.set_ylim([0, grid_shape[0]*cellsize])
        self.visualize_action_value(
            ax4,
            net_value,
            hash_state,
            grid_shape)
        self.goal_pose = goal_pose
        return ax


    def on_new_goal_pose(self, goal_pose):
        self.visualize_all(ax,
                           action_value = lambda s, a: (
                               action_value[s, a] if action_value.size else None),
                           policy = self.alg.policy,
                           path_cost = self.alg.path_cost,
                           net_value = lambda s, a: (
                               self.alg.net_value(s)[a] if action_value.size else None),
                           hash_state = self.alg.hash_state)



    def on_play_end(self):
        pass


def visualize_action_value(action_value, policy, path_cost, net_value,
                           hash_state, grid_shape, goal_pose, cellsize):
    vis = FloydWarshallVisualizer(update_interval = 1, cellsize = cellsize, log_file_dir = None)
    ax = draw.white_img(
        (grid_shape[1] * vis.cellsize, grid_shape[0] * 2 * vis.cellsize),
        dpi = cellsize)
    return vis.visualize_all(ax,
                             action_value = lambda s, a: action_value[s, a],
                             policy       = lambda obs: policy[hash_state[obs]],
                             path_cost    = path_cost,
                             net_value    = lambda s, a: net_value[s, a],
                             hash_state   = hash_state,
                             grid_shape   = grid_shape,
                             goal_pose    = goal_pose,
                             cellsize     = cellsize)

def post_process_data_tag(data, tag, cellsize, image_file_fmt):
    ax = visualize_action_value(
        action_value = data["action_value"],
        policy       = data["policy"],
        path_cost    = data["path_cost"],
        net_value    = data["net_value"],
        hash_state   = data["hash_state"],
        grid_shape   = data["grid_shape"],
        goal_pose    = data["goal_pose"],
        cellsize     = cellsize)
    fname = image_file_fmt.format(
        tag = "action_value",
        episode=data["episode_n"], step=data["steps"])
    img_dir = os.path.dirname(fname)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    print("Writing img to: {}".format(fname))
    draw.imwrite(fname, ax)

post_process_from_log_conf = functools.partial(
    ql_post_process_from_log_conf,
    process_data_tag = xargspartial(
        post_process_data_tag,
        "cellsize image_file_fmt".split()),
)

class FloydWarshallLogger(NoOPObserver):
    @extended_kwprop
    def __init__(self, logger, log_interval = 1,
                 action_value_tag = "FloydWarshallLogger:action_value",
                 post_process     = xargspartial(
                     post_process_from_log_conf,
                     "image_file_fmt log_file_reader action_value_tag".split()),
    ):
        self.logger           = logger
        self.log_interval     = log_interval
        self.human_tag        = "INFO"
        self.action_value_tag = action_value_tag
        self.episode_n        = None
        self.post_process     = post_process
        super().__init__()

    def info(self, tag, dct):
        self.logger.debug("", extra=dict(data=dct, tag=tag))

    def policy(self):
        policy = np.zeros(len(self.alg.hash_state.keys()), dtype='i8')
        for obs, k in self.alg.hash_state.items():
            policy[k] = self.alg.policy(obs)
        return policy

    def net_value(self):
        net_value = np.zeros((len(self.alg.hash_state.keys()),
                              self.alg.action_space.size))
        for obs, k in self.alg.hash_state.items():
            net_value[k, :] = self.alg.net_value(k)
        return net_value

    def on_new_step_with_pose_steps(self, obs, rew, act, pose, steps):
        if steps % self.log_interval == 0:
            self.info(self.action_value_tag,
                      dict(episode_n = int(self.episode_n),
                           steps     = int(steps),
                           obs       = obs.tolist(),
                           rew       = float(rew),
                           act       = int(act),
                           pose      = pose.tolist(),
                           action_value = self.alg.action_value,
                           policy       = self.policy(),
                           path_cost    = self.alg.path_cost,
                           net_value    = self.net_value(),
                           grid_shape   = self.prob.grid_shape,
                           goal_pose    = self.prob.goal_pose,
                           hash_state   = self.alg.hash_state
                      ))

    def on_new_step(self, obs, rew, action):
        self.on_new_step_with_pose_steps(
            obs, rew, action, self.prob.pose, self.prob.steps)

    def on_new_episode(self, episode_n):
        self.episode_n = episode_n

    def on_play_end(self):
        logging.shutdown()
        self.post_process()

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
            # increase the size of memory by one
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




class FloydWarshallAlgDiscrete(object):
    @extended_kwprop
    def __init__(self,
                 max_steps   = None,
                 goal_reward = 10,
                 init_fw     = -np.finfo('f8').min,
                 step_cost   = prop(lambda s: s.goal_reward / 10*s.max_steps ),
                 replay_memory = ReplayMemory,
                 key_frames  = [],
    ):
        assert max_steps is not None, "max_steps is required"
        self.replay_memory = qlearning
        self.reset()

    @property
    def per_edge_cost(self):
        return 10 * self.qlearning.reward_range[1] * (1-self.qlearning.discount)

    @property
    def path_cost_init(self):
        return 10 * self.qlearning.reward_range[1]

    def episode_reset(self, episode_n):
        self.qlearning.episode_reset(episode_n)

    def reset(self):
        self.qlearning.reset()
        self.path_cost     = self._default_path_cost(0)

    def _default_path_cost(self, new_state_size):
        shape = (new_state_size, self.action_space.size, new_state_size)
        path_cost = self.path_cost_init * np.ones(shape)
        return path_cost

    def _resize_path_cost(self, new_state_size):
        new_path_cost = self._default_path_cost(new_state_size)
        if self.path_cost.size:
            new_path_cost[
                tuple(map(lambda s : slice(None, s),
                          self.path_cost.shape))] = self.path_cost
        return new_path_cost

    def _state_idx_from_obs(self, obs, act, rew):
        state_idx = self.qlearning._state_idx_from_obs(obs, act, rew)
        if state_idx >= self.path_cost.shape[0]:
            self.path_cost = self._resize_path_cost(state_idx + 1)
        return state_idx

    def update(self, obs, act, rew):
        stm1, am1 = self.last_state_idx_act or (None, None)
        st = self._state_idx_from_obs(obs, act, rew)
        self.qlearning.update(obs, act, rew)
        if stm1 is None:
            return

        # Abbreviate the variables
        F = self.path_cost
        Q = self.qlearning.action_value

        # Make a conservative estimate of differential
        F[stm1, act, st] = max(np.max(Q[st, :]) - Q[stm1, act],
                               self.per_edge_cost)
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
        assert np.all(self.path_cost >= 0), "The Floyd cost should be positive at all times"

    def net_value(self, state_idx):
        Q = self.qlearning.action_value
        V = np.max(Q, axis=-1)
        state_action_values = np.maximum(
            Q[state_idx, :],
            np.max(V[None, :] - self.path_cost[state_idx, :, :] , axis=-1))
        return state_action_values

    def policy(self, obs):
        state = self._state_from_obs(obs)
        state_idx = self.hash_state[tuple(state)]
        return np.argmax(self.net_value(state_idx))

    def __getattr__(self, attr):
        if attr in """action_space done action_value hash_state
                      grid_shape init_value last_state_idx_act egreedy
                      _state_from_obs""".split():
            return getattr(self.qlearning, attr)
        else:
            raise AttributeError(f"No attribute {attr}")