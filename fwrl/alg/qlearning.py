from pathlib import Path
import numpy as np
#import torch as tch
import os
from queue import PriorityQueue
import logging
import operator
from functools import partial

from umcog.misc import NumpyEncoder
from umcog.confutils import xargs, xargspartial, xargmem, KWProp, extended_kwprop
import umcog.draw as draw
from ..game.play import Space, Alg, NoOPObserver, post_process_data_iter, show_ax_human, show_ax_log

def logger():
    return logging.getLogger(__name__)


def rand_argmax(arr, rng):
    """
    Chose the a random arg among maxes
    """
    val = np.max(arr, keepdims=True)
    idx = np.arange(arr.shape[0])[arr == val]
    return rng.choice(idx)

def q_policy(state_idx, action_value, rng):
    # desirable_dest = max(
    #     self.top_m_states.queue,
    #     key = lambda s: self.action_value[s[1]])[1]
    #logger().debug(
    #    "state = {state}; action_values = {av}".format(
    #        av=self.action_value[state_idx, :], state=state))
    return rand_argmax(action_value[state_idx, :], rng)

def linscale(x, src_range, target_range):
    ss, se = src_range
    ts, te = target_range
    return (x - ss) / (se - ss) * (te - ts) + ts

def egreedy_prob_exp(step, start_eps = 0.5, end_eps = 0.001, nepisodes = None, alpha = -20):
    """
    >>> egreedy_prob_exp(np.array([0, 500, 1000]), start_eps = 0.8, end_eps = 0.001,
    ...                  nepisodes = 1000, alpha = np.log(0.001 / 0.8))
    array([ 0.8       ,  0.02828427,  0.001     ])
    """
    assert nepisodes is not None, "nepisodes is required"
    # scale later
    return linscale(np.exp( alpha * np.minimum(step, nepisodes) / nepisodes ),
                    (1, np.exp(alpha)), (start_eps, end_eps))


class QLearningDiscrete(Alg):
    egreedy_prob_exp = egreedy_prob_exp
    def __init__(self,
                 action_space,
                 observation_space,
                 reward_range,
                 rng,
                 egreedy_prob          = partial(egreedy_prob_exp, nepisodes = 200),
                 action_value_momentum = 0.0, # Low momentum changes more frequently
                 discount              = 1.00, # step cost
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self.reward_range         = reward_range
        self.rng                  = rng
        self.egreedy_prob         = egreedy_prob
        self.action_value_momentum= action_value_momentum
        assert reward_range[0] >= 0, "Reward range"
        self.init_value           = discount * reward_range[0]
        self.discount             = discount
        self.reset()

    def episode_reset(self, episode_n):
        self.action_value[:]= self.init_value
        self.last_state_idx = None
        self.step = 0

    def reset(self):
        self.action_value    = self._default_action_value(0)
        self.hash_state     = dict()
        self.episode_reset(0)

    def _default_action_value(self, state_size):
        return self.init_value * np.ones((state_size, self.action_space.size))

    def _resize_action_value(self, new_size):
        new_action_value = self._default_action_value(new_size)
        if self.action_value.shape[0]:
            new_action_value[:self.action_value.shape[0], :] = self.action_value
        return new_action_value

    def egreedy(self, greedy_act):
        sample_greedy = (self.rng.rand() >= self.egreedy_prob(self.step))
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
        return q_policy(state_idx, self.action_value, self.rng)

    def _hit_goal(self, obs, act, rew, done, info):
        return info.get("hit_goal", False)

    def on_hit_goal(self, obs, act, rew):
        self.last_state_idx = None

    def is_terminal_step(self, obs, act, rew, done, info):
        return done or info.get("new_spawn", False)

    def update(self, obs, act, rew, done, info):
        self.step += 1
        # Protocol defined by: game.play:play_episode()
        # - act = alg.policy(obs)
        # - obs_plus_1, rew_plus_1 = the prob.step(act)
        # - the alg.update(obs, act, rew)
        # or
        # obs_m_1 --alg--> act --prob--> obs, rew # # # obs, rew = prob.step(action)
        if not self.observation_space.contains(obs):
            raise ValueError("Bad observation {obs}".format(obs=obs))

        # Encoding state_hash from observation
        st = self._state_idx_from_obs(obs, act, rew)
        stm1 = self.last_state_idx
        self.last_state_idx = st
        if stm1 is None:
            return self.egreedy(self.policy(obs))

        if self._hit_goal(obs, act, rew, done, info):
            self.on_hit_goal(obs, act, rew)

        # terminal step
        ts = self.is_terminal_step(obs, act, rew, done, info)

        # Abbreviate the variables
        qm = self.action_value_momentum
        Q = self.action_value
        d = self.discount

        # Update step from online observed reward
        Q[stm1, act] = (1-qm) * (rew + (1-ts) * d * np.max(Q[st, :])) + qm * Q[stm1, act]
        return self.egreedy(self.policy(obs))

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False

class QLearningVis(NoOPObserver):
    def __init__(self, log_file_dir,
                 update_interval = 1,
                 cellsize = 40,
                 rng = np.random.RandomState(seed = 0)):
        self.rng = rng
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

    def _action_value_to_mat(self, action_value, hash_state, grid_shape):
        mat = np.ones(grid_shape) * self.alg.init_value
        if action_value.size:
            for state_pose, state_idx in hash_state.items():
                mat[state_pose] = np.max(action_value[state_idx, :])
        return mat

    def _action_value_to_matrices(self, action_value, hash_state, grid_shape):
        big_mat = np.zeros(np.array(grid_shape) * 2)
        # if self.action_space.size != 4:
        #     raise NotImplementedError("Do not know how to visualize")

        for act in range(4):
            mat = np.zeros(grid_shape) * 0
            for state_pose, state_idx in hash_state.items():
                mat[state_pose] = action_value(state_idx, act) or mat[state_pose]
            big_r = act // 2
            big_c = act % 2
            big_mat[big_r::2, big_c::2] = mat
        return big_mat

    def visualize_action_value(self, ax, action_value, hash_state, grid_shape):
        cellsize = ax.get_xlim()[1] / grid_shape[1]
        action_value_mat = self._action_value_to_matrices(
            action_value, hash_state, grid_shape)
        action_value_mat += np.min(action_value_mat[:])
        draw.matshow(ax, self.normalize_by_std(action_value_mat))
        c50 = cellsize/2
        c25 = cellsize/4
        c75 = cellsize*3/4
        for i, j in np.ndindex(action_value_mat.shape):
            center = np.array((i*c50 + c25, j*c50 + c25))
            if i % 2 == 0 and j % 2 == 0:
                draw.rectangle(ax, center - c25, center + c75, (0, 0, 0))
            draw.putText(ax, "{:.3}".format(action_value_mat[i, j]),
                         center, fontScale=4)

    def _policy_to_mat(self, policy_func):
        return mat

    def visualize_policy(self, ax, policy_func, hash_state, grid_shape, goal_pose=None):
        cellsize = ax.get_xlim()[1] / grid_shape[0]
        VECTORS = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]])


        if goal_pose is not None and len(hash_state):
            vis_goal_pose = np.asarray(goal_pose
                            if tuple(goal_pose) in hash_state
                            else list(hash_state.keys())[-1])

            draw.rectangle(ax, vis_goal_pose *cellsize,
                           (vis_goal_pose + 1)*cellsize,
                           (0, 255, 0), thickness=-1)
        for state in hash_state.keys():
            act = policy_func(state)
            act_vec = np.array(VECTORS[act, :])
            center = (np.array(state) + 0.5) * cellsize
            draw.arrowedLine(ax,
                             center - act_vec * cellsize / 4,
                             center + act_vec * cellsize / 4,
                             (0, 0, 0),
                             thickness=5,
                             tipLength=10)


    def alg_policy(self, ax, action_value, hash_state):
        # policy = np.zeros(len(self.alg.hash_state.keys()), dtype='i8')
        # for obs, k in self.alg.hash_state.items():
        #     policy[k] = self.alg.policy(obs)
        # return policy
        return 


    def visualize(self, ax, action_value, hash_state, wait_time = None,
                  grid_shape = None, cellsize = None):
        ax.set_position([0, 0, 0.5, 1])
        ax.clear()
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, grid_shape[1]*cellsize])
        ax.set_ylim([0, grid_shape[0]*cellsize])
        self.visualize_action_value(
            ax,
            lambda s, a: (action_value[s, a] if action_value.size else None),
            hash_state,
            grid_shape)
        ax2 = ax.figure.add_axes([0.5, 0, 0.5, 1], frameon=False)
        ax2.set_position([0.5, 0, 0.5, 1])
        ax2.clear()
        ax2.axis('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim([0, grid_shape[1]*cellsize])
        ax2.set_ylim([0, grid_shape[0]*cellsize])
        self.visualize_policy(ax2,
                              lambda obs: (q_policy(hash_state[obs], action_value,
                                                    self.rng)
                                           if action_value.size else 0),
                              hash_state, grid_shape)

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
            (grid_shape[1]*cellsize, grid_shape[0]*2*cellsize),
            dpi=cellsize)
        self.visualize(ax, self.alg.action_value, self.alg.hash_state,
                       1, self.grid_shape, cellsize)
        self.goal_pose = goal_pose
        return ax

    def on_new_step(self, obs, rew, action, info):
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


def visualize_action_value(action_value, hash_state, grid_shape, cellsize):
    vis = QLearningVis(update_interval = 1, cellsize = cellsize, log_file_dir = None)
    ax = draw.white_img(
        (grid_shape[1] * vis.cellsize, grid_shape[0] * 2 * vis.cellsize),
        dpi = cellsize)
    vis.visualize(ax, action_value, hash_state,
                  grid_shape = grid_shape,
                  cellsize = cellsize)
    return ax


class Renderer:
    human = partial(show_ax_human, tag = "action_value")
    log = partial(show_ax_log, tag = "action_value")



def post_process_data_tag(data, tag, cellsize, renderer):
    ax = visualize_action_value(
        data["action_value"], data["hash_state"], data["grid_shape"], cellsize)
    return renderer(ax, data)


def post_process_generic(data_iter, process_data_tag=post_process_data_tag):
    return [process_data_tag(data=data, tag=tag)
            for data, tag in data_iter()] 

@extended_kwprop
def post_process(
        data_iter    = xargspartial(
            post_process_data_iter,
            "log_file_reader filter_criteria".split()),
        process_data_tag = xargspartial(
            post_process_data_tag,
            "cellsize renderer".split())):
    return post_process_generic(data_iter, process_data_tag)

post_process_from_log_conf = partial(
    post_process,
    cellsize         = 80,
    filter_criteria  = KWProp(
        lambda s : dict( tag = s.action_value_tag)),
    # Needs
    # renderer,
    # log_file_reader,
    # action_value_tag,
)

class QLearningLogger(NoOPObserver):
    @extended_kwprop
    def __init__(self,
                 logger,
                 log_interval     = 1,
                 human_tag        = "INFO",
                 action_value_tag = "QLearningLogger:action_value",
                 sep              = "\t",
                 renderer         = KWProp(lambda s: partial(Renderer.log,
                                                           image_file_fmt = s.image_file_fmt)),
                 post_process     = xargspartial(
                     post_process_from_log_conf,
                     "renderer log_file_reader action_value_tag".split()),
    ):
        self.logger           = logger
        self.log_interval     = log_interval
        self.human_tag        = human_tag
        self.action_value_tag = action_value_tag
        self.sep = sep
        self.episode_n        = None
        self.post_process     = post_process
        super().__init__()

    def info(self, tag, dct):
        self.logger.debug("", extra=dict(tag=tag, data=dct))

    def policy(self):
        policy = np.zeros(len(self.alg.hash_state.keys()), dtype=np.int8)
        for obs, k in self.alg.hash_state.items():
            policy[k] = self.alg.policy(obs)
        return policy

    def on_new_step_with_pose_steps(self, obs, rew, act, pose, steps,
                                    grid_shape, goal_pose):
        if steps % self.log_interval == 0:
            self.info(self.action_value_tag,
                      dict(episode_n    = int(self.episode_n),
                           steps        = int(steps),
                           obs          = obs.tolist(),
                           rew          = float(rew),
                           act          = int(act),
                           pose         = pose,
                           grid_shape   = grid_shape,
                           goal_pose    = goal_pose,
                           action_value = self.alg.action_value,
                           policy       = self.policy(),
                           hash_state   = self.alg.hash_state))

    def on_new_step(self, obs, rew, action, info):
        self.on_new_step_with_pose_steps(
            obs = obs, rew = rew,
            act = action, pose = self.prob.pose, steps = self.prob.steps,
            grid_shape = self.prob.grid_shape,
            goal_pose = self.prob.grid_shape)

    def on_new_episode(self, episode_n):
        self.episode_n = episode_n

    def on_play_end(self):
        logging.shutdown()
        self.post_process()

