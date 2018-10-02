"""
Returns observation as a window around the agent
"""
import numpy as np
from gym.spaces import Box

from umcog.functools import compose
from .windy_grid_world import AgentInGridWorld


def maze_slice(maze, pose, window_rect):
    """
    >>> m = np.arange(25).reshape(5,5)
    >>> window_rect = ((-1, -1), (1, 1))            # in x, y
    >>> pose = np.hstack(np.where(m == 13))[::-1]   # row, col -> x,y
    >>> maze_slice(m, pose, window_rect)
    array([[ 7,  8,  9],
           [12, 13, 14],
           [17, 18, 19]])
    """
    pose_rect = np.asarray(window_rect) + pose
    (cmin, rmin), (cmax, rmax) = pose_rect
    return maze[rmin:rmax+1, cmin:cmax+1]


class ScrollingWindow:
    def __init__(self, agent_in_gw, window_rect = ((-1, -1), (1, 1))):
        self.agent_in_gw = agent_in_gw
        self.window_rect = window_rect

    @property
    def observation_space(self):
        (cmin, rmin), (cmax, rmax) = self.window_rect
        return Box(low   = 0,
                   high  = 10,
                   shape = (rmax-rmin+1, cmax-cmin+1))

    def _pose_to_obs(self, pose):
        return maze_slice(self.agent_in_gw.maze(), pose, self.window_rect)

    def step(self, act):
        pose, rew, done, info = self.agent_in_gw.step(act)
        return self._pose_to_obs(pose), rew, done, info

    def observation(self):
        return self._pose_to_obs(self.agent_in_gw.pose)

    def episode_reset(self, episode_n):
        episode_info = self.agent_in_gw.episode_reset(episode_n)
        if 'goal_obs' in episode_info:
            episode_info['goal_obs'] = self._pose_to_obs(episode_info['goal_obs'])
        return episode_info

    def __getattr__(self, attr):
        return getattr(self.agent_in_gw, attr)


class AgentInScrollingGW:
    """
    >>> agw = AgentInScrollingGW.from_maze_name(maze_name = "4-room-grid-world", seed=0)
    >>> agw.episode_reset(0)
    {'goal_obs': array([[0, 0, 0],
           [0, 6, 0],
           [0, 0, 0]], dtype=uint8)}
    >>> for _ in range(5):
    ...     obs, rew, done, info = agw.step(agw.action_space.sample())
    ...     print(obs)
    [[0 0 0]
     [0 0 0]
     [0 0 0]]
    [[0 0 0]
     [0 0 0]
     [1 1 1]]
    [[0 0 0]
     [1 0 0]
     [1 1 1]]
    [[1 0 0]
     [0 0 0]
     [1 0 0]]
    [[0 0 0]
     [1 0 0]
     [1 1 1]]
    """
    # equivalent to ScrollingWindow(AgentInScrollingGW.from_maze_name(**kw))
    from_maze_name = compose(ScrollingWindow, AgentInGridWorld.from_maze_name)
    from_random_maze = compose(ScrollingWindow, AgentInGridWorld.from_random_maze)
