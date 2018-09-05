import os
from functools import partial
import logging
from collections import namedtuple

import numpy as np

import umcog.draw as draw
from ..game.play import Space, Alg, NoOPObserver
from .qlearning import (QLearningDiscrete, QLearningVis,
                        post_process_from_log_conf as ql_post_process_from_log_conf,
                        post_process_data_iter, post_process_generic,
                        rand_argmax, Renderer)
from umcog.confutils import xargs, xargspartial, xargmem, KWProp as prop, extended_kwprop

def logger():
    return logging.getLogger(__name__)


class FloydWarshallAlgDiscrete(object):
    @extended_kwprop
    def __init__(self,
                 qlearning = xargs(
                     QLearningDiscrete,
                     "action_space observation_space reward_range rng".split()),
                 consistency_update_prob = 0.1
    ):
        self.qlearning     = qlearning
        self.path_cost     = np.zeros((0, self.action_space.size, 0))
        self.consistency_update_prob = consistency_update_prob
        self.reset()

    @property
    def path_cost_init(self):
        #return 10 * self.qlearning.reward_range[1]
        return np.inf

    def episode_reset(self, episode_n):
        self.qlearning.episode_reset(episode_n)
        self.goal_state    = None
        self.last_state_idx = None

    def reset(self):
        self.qlearning.reset()
        self.path_cost     = self._default_path_cost(0)

    def _default_path_cost(self, new_state_size):
        shape = (new_state_size, self.action_space.size, new_state_size)
        path_cost = self.path_cost_init * np.ones(shape)
        #path_cost[np.arange(shape[0]), :, np.arange(shape[0])] = 0
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

    def get_action_value_dct(self):
        print("episode_n {}".format(self.qlearning.episode_n))
        return dict(net_value = -self.path_cost[:, :, self.goal_state],
                    hash_state = self.hash_state,
                    episode_n = self.qlearning.episode_n,
                    steps = self.qlearning.step)

    def update(self, obs, act, rew, done, info):
        stm1 = self.last_state_idx
        st = self._state_idx_from_obs(obs, act, rew)
        self.last_state_idx = st

        if self.qlearning._hit_goal(obs, act, rew, done, info):
            self.goal_state = stm1
            self.consistency_update()
            return self.qlearning.egreedy(self.policy(obs))

        self.qlearning.update(obs, act, rew, done, info)

        if stm1 is None:
            return self.qlearning.egreedy(self.policy(obs))

        # Abbreviate the variables
        F = self.path_cost
        F[stm1, act, st] = max(-rew, 0)

        if self.qlearning.rng.rand() <= self.consistency_update_prob:
            self.consistency_update()

        return self.qlearning.egreedy(self.policy(obs))

    def consistency_update(self):
        F = self.path_cost
        np.minimum(
            F,
            np.min(
                F[:, :,  :, np.newaxis] + np.min(F[:, :, :], axis=1),
                axis=2),
            out=F)
        assert np.all(self.path_cost >= 0), "The Floyd cost should be positive at all times"

    def random_state(self):
        return self.qlearning.rng.randint(self.path_cost.shape[2])

    def net_value(self, state_idx):
        if self.goal_state is None \
           or np.all(self.path_cost[state_idx, :, self.goal_state] == self.path_cost_init):
            return self.qlearning.action_value[state_idx, :]
            #return self.path_cost[self.spawn_state, :, state_idx]
        else:
            #Q = self.qlearning.action_value
            return - self.path_cost[state_idx, :, self.goal_state]

    def policy(self, obs):
        state_idx = self._state_idx_from_obs(obs, None, None)
        net_val = self.net_value(state_idx)
        #print("st:{} -> {}".format(state_idx, net_val))
        return rand_argmax(net_val, self.qlearning.rng)

    def __getattr__(self, attr):
        if attr in """action_space done action_value hash_state
                      grid_shape init_value egreedy
                      _state_from_obs """.split():
            return getattr(self.qlearning, attr)
        else:
            raise AttributeError("No attribute {attr}".format(attr=attr))

    def set_goal_obs(self, goal_obs):
        self.qlearning.set_goal_obs(goal_obs)
        self.goal_state = self._state_idx_from_obs(goal_obs, None, None)


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
            draw.putText(ax, "{:.3}".format(path_cost_mat[i, j]),
                         center, fontScale=4)

    def visualize_all(self, ax, action_value, policy, path_cost,
                      net_value, hash_state, grid_shape, goal_pose,
                      cellsize):
        ax = draw.white_img(
            #(grid_shape[1]*2*cellsize, grid_shape[0]*2*cellsize),
            (grid_shape[1]*cellsize, grid_shape[0]*2*cellsize),
            dpi=cellsize*2)
        ax.set_position([0, 0, 0.5, 1.0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, grid_shape[1]*cellsize])
        ax.set_ylim([0, grid_shape[0]*cellsize])
        #self.visualize_action_value(ax, action_value, hash_state, grid_shape)

        #ax2 = ax.figure.add_axes([0.5, 0, 0.5, 0.5], frameon=False)
        #ax2.set_position([0.5, 0, 0.5, 0.5])
        #ax2.set_xticks([])
        #ax2.set_yticks([])
        #ax2.set_xlim([0, grid_shape[1]*cellsize])
        #ax2.set_ylim([0, grid_shape[0]*cellsize])
        #self.visualize_policy(ax2, policy, hash_state, grid_shape)
        self.visualize_policy(ax, policy, hash_state, grid_shape, goal_pose)

        #ax3 = ax.figure.add_axes([0, 0.5, 0.5, 0.5], frameon=False)
        #ax3.set_position([0, 0.5, 0.5, 0.5])
        #ax3.set_xticks([])
        #ax3.set_yticks([])
        #ax3.set_xlim([0, grid_shape[1]*cellsize])
        #ax3.set_ylim([0, grid_shape[0]*cellsize])
        #self.visualize_path_cost(ax3, path_cost, hash_state, goal_pose, grid_shape)

        ax4 = ax.figure.add_axes([0.5, 0.0, 0.5, 1.0], frameon=False)
        ax4.set_position([0.5, 0.0, 0.5, 1.0])
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


def post_process_data_tag(data, tag, cellsize, renderer):
    ax = visualize_action_value(
        action_value = data["action_value"],
        policy       = data["policy"],
        path_cost    = data["path_cost"],
        net_value    = data["net_value"],
        hash_state   = data["hash_state"],
        grid_shape   = data["grid_shape"],
        goal_pose    = data["goal_pose"],
        cellsize     = cellsize)
    return renderer(ax, data)


post_process_from_log_conf = partial(
    ql_post_process_from_log_conf,
    process_data_tag = xargspartial(
        post_process_data_tag,
        "cellsize renderer".split()))

class FloydWarshallLogger(NoOPObserver):
    @extended_kwprop
    def __init__(self, logger, log_interval = 1,
                 action_value_tag = "FloydWarshallLogger:action_value",
                 renderer         = prop(lambda s: partial(
                     Renderer.log,
                     image_file_fmt = s.image_file_fmt)),
                 post_process     = xargspartial(
                     post_process_from_log_conf,
                     "log_file_reader renderer action_value_tag".split()),
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
                           goal_pose    = self.prob.goal_obs,
                           hash_state   = self.alg.hash_state
                      ))

    def on_new_step(self, obs, rew, action, info):
        self.on_new_step_with_pose_steps(
            obs, rew, action, self.prob.pose, self.prob.steps)

    def on_new_episode(self, episode_n, obs=None, goal_obs=None):
        self.episode_n = episode_n
        self.goal_pose = goal_obs

    def on_play_end(self):
        logging.shutdown()
        self.post_process()



