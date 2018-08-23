"""
Returns observation as a window around the agent
"""
import numpy as np
from gym.spaces import Box

from umcog.misc import compose
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

    def __getattr__(self, attr):
        return getattr(self.agent_in_gw, attr)

class AgentInScrollingGW:
    from_maze_name = compose(ScrollingWindow, AgentInGridWorld.from_maze_name)
    from_random_maze = compose(ScrollingWindow, AgentInGridWorld.from_random_maze)
