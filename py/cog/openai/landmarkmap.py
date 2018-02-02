import time

import gym
from gym.envs.registration import register

#register(
#    id='LandmarkMap2D-v0',
#    entry_point='cog.openai.landmarkmap:LandmarkMap',
#)

import numpy as np
from ..slam.landmarkmap import (map_from_conf , LandmarksVisualizer
                               , T_from_pos_dir , pos_dir_from_T
                               , R2D_angle)
from ..slam.imap import imap_conf_gen

def timeseed():
    return np.uint64(time.time())

class ActionSpace(gym.Space):
    NO_ACTION, FORWARD , BACKWARD , LEFT , RIGHT = (0 , 1 , 2 , 3, 4)
    _space = (NO_ACTION, FORWARD , BACKWARD , LEFT , RIGHT)
    n = len(_space)
    def size(self):
        return len(self._space)

    def sample(self, seed=0):
        """
        Uniformly randomly sample a random elemnt of this space
        """
        return self._space[
            np.random.randint(len(self._space))]

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        return x in self._space

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n

class ObservationSpace(gym.Space):
    loc_scale = np.array([120, 120]).reshape(-1, 1)
    color_scale = np.array([255, 255, 255]).reshape(-1, 1)
    _nlandmarks = 5
    _loc_shape = (2, _nlandmarks)
    _color_shape = (3, _nlandmarks)
    _region_color_shape = (5, 1) # Dummy zeros added
    shape = (_color_shape[0] + _loc_shape[0]
             , _nlandmarks + _region_color_shape[1])
    def sample(self, seed=0):
        """
        Uniformly randomly sample a random elemnt of this space
        """
        locs = np.random.rand(*self._loc_shape) * 2 - 1
        colors = np.random.rand(*self._color_shape)
        region_color = np.random.rand(*self._region_color_shape)
        region_color[:2, :] = 0.
        return np.hstack((np.vstack((locs, colors)), region_color))

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        return (x.ndim == 2 and x.shape == self.shape
                and np.all(np.abs(x[:2, :]) <= 1.)
                and np.all(x[2:, :] <= 1.)
                and np.all(x[2:, :] >= 0))

    @classmethod
    def normalize(cls, unnorm_obs):
        unnorm_obs[:2, :] /= cls.loc_scale
        unnorm_obs[2:, :] /= cls.color_scale
        return unnorm_obs


    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n

def sign(a):
    if a > 0:
        return 1
    elif a == 0:
        return 0
    else:
        return -1

def clipped(a, max):
    return sign(a)*max if abs(a) > max else a

class MotionModel(object):
    def __init__(self, max_lin_vel, max_ang_vel=np.pi / 10.):
        self._action_space = ActionSpace()
        self._max_lin_vel = max_lin_vel
        self._inc_lin_vel = max_lin_vel / 2.
        self._max_ang_vel = max_ang_vel
        self._inc_ang_vel = max_ang_vel / 2.
        self._inc_time_step = 1.

    def increment_velocities(self, velocities, action):
        action = int(action)
        lin_vel, ang_vel = velocities
        if action == self._action_space.FORWARD:
            lin_vel = clipped(lin_vel + self._inc_lin_vel, self._max_lin_vel)
        elif action == self._action_space.BACKWARD:
            lin_vel = clipped(lin_vel - self._inc_lin_vel, self._max_lin_vel)
        elif action == self._action_space.LEFT:
            ang_vel = clipped(ang_vel + self._inc_ang_vel, self._max_ang_vel)
        elif action == self._action_space.RIGHT:
            ang_vel = clipped(ang_vel - self._inc_ang_vel, self._max_ang_vel)
        elif action == self._action_space.NO_ACTION:
            pass
        else:
            raise ValueError("Bad action %d" % action)
        return (lin_vel, ang_vel)

    def next(self, T, velocities, action):
        (lin_vel, ang_vel) = self.increment_velocities(velocities, action)
        del_theta = ang_vel * self._inc_time_step
        dir = T[:-1, :-1].dot([1, 0])
        del_pos = lin_vel * dir * self._inc_time_step
        T_new = T.copy()
        T_new[:-1, :-1] = R2D_angle(del_theta).dot(T[:-1, :-1])
        T_new[:-1, -1] = T[:-1, -1] + del_pos
        return T_new, (lin_vel, ang_vel)

class LandmarkMap(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, map_conf_gen=None, save_images=False):
        self.save_images = save_images
        if map_conf_gen is None:
            self._map_conf_gen = imap_conf_gen

        map_conf, self._robot_block, self._start_area =\
                self._map_conf_gen(timeseed())
        self._timesteps = 1500
        self._map = map_from_conf(map_conf, self._timesteps)
        self.viewer = LandmarksVisualizer([0,0], [100, 120], frame_period=1)
        self._action_space = ActionSpace()
        self._obs_space = ObservationSpace()
        self._motion_model = MotionModel(max_lin_vel=8.)
        self._reset()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def reward_range(self):
        return [-1, 1]

    def _take_action(self, action):
        old_T, old_vel = self._robot_block.current_T, \
                self._robot_block.current_vel
        T, vel = self._motion_model.next(self._robot_block.current_T
                                         , self._robot_block.current_vel
                                         , action)
        self._robot_block.current_T = T
        self._robot_block.current_vel = vel
        episode_over = False
        try:
            self._current_map = self._map_iter.next()
        except StopIteration:
            episode_over = True

        region_t = self._current_map.get_region_by_pos(
            self._robot_block.vertices.T)
        override_reward = None
        # Revert back if the robot tan into an obstacle 
        if region_t.is_terminal:
            #print("Terminal region with reward %f" % region_t.reward)
            episode_over = True
        elif region_t.is_obstacle:
            self._robot_block.current_T = old_T
            self._robot_block.current_vel = (
                - 0.5*np.random.rand()*old_vel[0]
                , - 0.5*np.random.rand()*old_vel[1])
            episode_over = False
            try:
                self._current_map = self._map_iter.next()
            except StopIteration:
                #print("Exceeded max iterations")
                override_reward = -1.0
                episode_over = True

        return episode_over, region_t, override_reward

    def _collect_observations(self, region_t):
        map_t = self._current_map
        ldmks = map_t.landmarks
        self._current_rob_view = self._robot_block.current_robot_view()
        in_view_ldmks = self._current_rob_view.in_view(ldmks.locations)
        in_view_ldmks_pos = ldmks.locations[:, in_view_ldmks]
        in_view_ldmks_colors = ldmks.colors[:, in_view_ldmks]
        pos, dir = pos_dir_from_T(self._robot_block.current_T)
        if in_view_ldmks_pos.shape[1] >= 1:
            available = in_view_ldmks_pos.shape[1]
            required = self._obs_space._nlandmarks
            choice  = np.random.choice(available, required)
            locs = in_view_ldmks_pos[:, choice]
            colors = in_view_ldmks_colors[:, choice]
            locs_colors = np.vstack((locs, colors))
            region_color = region_t.color
            region_color_col = np.hstack((np.zeros((2,)), region_color)).reshape(-1, 1)
            obs = np.hstack((locs_colors, region_color_col))
        else:
            obs = np.zeros(self._obs_space.shape)
        obs = self._obs_space.normalize(obs)
        assert self._obs_space.contains(obs), 'Obs: {0} {1}'.format(obs.shape, obs)
        return obs

    def _step(self, action):
        episode_over, region_t, override_reward = self._take_action(action)
        obs = self._collect_observations(region_t)
        if episode_over:
            self._reset()
        return obs, (region_t.reward if override_reward is None else
                     override_reward), episode_over, {}

    def _reset(self):
        map_conf, self._robot_block, self._start_area =\
                self._map_conf_gen(timeseed())
        self._map = map_from_conf(map_conf , self._timesteps)
        self._map_iter = self._map.get_map_traj()
        theta = np.random.rand()*2*np.pi
        self._robot_block.current_T = T_from_pos_dir(
            self._start_area.sample(1).flatten()
            , np.cos([theta, np.pi - theta]))
        self._current_rob_view = self._robot_block.current_robot_view()
        self._current_map = self._map_iter.next()
        region_t = self._current_map.get_region_by_pos(
            self._robot_block.vertices.T)
        return self._collect_observations(region_t)

    def _render(self, mode='human', close=False):
        if close:
            return

        frame = self.viewer.genframe(self._current_map.landmarks
                           , self._current_rob_view
                           , map_blocks = self._current_map.regions)
        self.viewer.visualizeframe(frame, write=self.save_images)

    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
