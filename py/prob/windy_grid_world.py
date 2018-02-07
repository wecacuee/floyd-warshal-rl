#!/usr/bin/python3
from __future__ import absolute_import, division, print_function
from game.interface import Space, Problem
import numpy as np
from cog import draw


def maze_from_string(maze_string,
                     intable = " +^<>V",
                     outtable = range(6)):
    r"""
    >>> maze_from_string("  +  \n  ^  \n ^^^ \n ^^^ \n  +  ")
    array([[0, 0, 1, 0, 0],
           [0, 0, 2, 0, 0],
           [0, 2, 2, 2, 0],
           [0, 2, 2, 2, 0],
           [0, 0, 1, 0, 0]], dtype=uint8)

    >>> maze_from_string("  +  \n  ^  \n < > \n ^V^ \n  +  ")
    array([[0, 0, 1, 0, 0],
           [0, 0, 2, 0, 0],
           [0, 3, 0, 4, 0],
           [0, 2, 5, 2, 0],
           [0, 0, 1, 0, 0]], dtype=uint8)
    """
    maze_bytes = maze_string.encode("ascii")
    maze_lines = maze_string.split("\n")
    nrows = len(maze_lines)
    ncols = max(map(len, maze_lines))
    maze_arr = np.zeros((len(maze_lines), ncols), dtype='u1')
    for line in maze_lines:
        if len(line) < ncols:
            line += ' ' * (ncols - len(line))
    maze_ch = np.frombuffer(''.join(maze_lines).encode('ascii'), dtype='u1'
                ).reshape(nrows, ncols)
    for ch, replace in zip(intable, outtable):
        maze_arr[maze_ch == ord(ch)] = replace
    return maze_arr


class Act2DSpace(Space):
    NAMES = ["NORTH", "EAST", "WEST",  "SOUTH"]
    VECTORS = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]])
    def __init__(self, seed):
        self.rng = np.random.RandomState()
        self.rng.seed(seed)
        
    def sample(self):
        return self.rng.randint(0, len(self.NAMES))

    def contains(self, x):
        return x in self.values()

    def values(self):
        return range(self.size)

    @property
    def size(self):
        return len(self.NAMES)


class Loc2DSpace(Space):
    def __init__(self, lower_bound, upper_bound, seed):
        self.rng = np.random.RandomState()
        self.rng.seed(seed)
        self.lower_bound = np.asarray(lower_bound)
        self.upper_bound = np.asarray(upper_bound)
        
    def sample(self):
        return np.array([
            self.rng.randint(self.lower_bound[0], self.upper_bound[0]),
            self.rng.randint(self.lower_bound[1], self.upper_bound[1])])

    def contains(self, x):
        xint = np.int64(x)
        return np.all((self.lower_bound <= x) & (x < self.upper_bound))
        

class WindyGridWorld(object):
    CELL_FREE = 0
    CELL_WALL = 1
    CELL_WIND_NEWS = [2, 3, 4, 5]
    WIND_NEWS_NAMES = "NORTH EAST WEST SOUTH".split()
    WIND_NEWS_VECTORS = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]])
    def __init__(self, seed, maze=None, wind_strength=0.5):
        self.maze = maze if maze is not None else self.default_maze()
        self.rng = np.random.RandomState()
        self.rng.seed(seed)
        self.wind_strength = wind_strength

    def wind_dir(self, xy):
        row, col = xy[::-1]
        return self.wind_vectors(self.maze[row, col])

    def next_pose(self, pose, act_vec):
        wind_prob = 1 if (self.rng.uniform() < self.wind_strength) else 0
        potential_pose = (pose + act_vec
                          + wind_prob * self.wind_dir(pose))
        next_p = pose if self.iswall(potential_pose) else potential_pose
        return next_p

    def iswall(self, xy):
        row, col = xy[::-1]
        if np.any(np.asarray(xy) < 0):
            return True
        try:
            return self.maze[row, col] == self.CELL_WALL
        except IndexError as e:
            return True
        

    def wind_vectors(self, cell_code):
        if cell_code not in  self.CELL_WIND_NEWS:
            return np.array([0, 0])
        else:
            return self.WIND_NEWS_VECTORS[cell_code - self.CELL_WIND_NEWS[0]]

    def render(self, canvas, grid_size):
        if canvas is None:
            canvas = draw.white_img(self.maze.shape)

        nrows, ncols = self.maze.shape
        for r, c in np.ndindex(*self.maze.shape):
            bottom_left = np.array([c, r]) * grid_size
            if self.iswall((c,r)):
                draw.rectangle(canvas, bottom_left, bottom_left + grid_size,
                               color=draw.color_from_rgb((0,0,0)),
                               thickness = -1)
            else:
                draw.rectangle(canvas, bottom_left, bottom_left + grid_size,
                               color=draw.color_from_rgb((0,0,0)),
                               thickness = 1)

            if np.any(self.wind_dir((c,r))):
                center = bottom_left + 0.5 * grid_size
                wind_dir = self.wind_dir((c,r))
                pt1 = center - 0.5*wind_dir
                pt2 = center + 0.5*wind_dir
                draw.arrowedLine(canvas, pt1, pt2,
                                 draw.color_from_rgb((0,0,0)))
                
        return canvas

    @property
    def shape(self):
        return self.maze.shape

    @classmethod
    def default_maze(self):
        return maze_from_string("\n".join([
            "  +  ",
            "  ^  ",
            " ^^^ ",
            " ^^^ ",
            "  +  "]))
        

class AgentInGridWorld(Problem):
    def __init__(self, seed, grid_world, start_pose, goal_pose, goal_reward, max_steps):
        self.grid_world        = grid_world
        self.pose              = np.asarray(start_pose)
        self.action_space      = Act2DSpace(seed)
        self.goal_pose         = np.asarray(goal_pose)
        self.goal_reward       = goal_reward
        self.max_steps         = max_steps
        self.observation_space = Loc2DSpace(
            lower_bound = np.array([0, 0]),
            upper_bound = np.array(grid_world.shape),
            seed        = seed) 
        self.episode_reset()

    def step(self, act):
        self.pose = self.grid_world.next_pose(
            self.pose,
            self.action_space.VECTORS[act])
        self.steps += 1
        return self.pose

    def reward(self):
        return self.goal_reward if (np.all(self.goal_pose == self.pose)) else 0

    def observation(self):
        return self.pose

    def render(self, canvas, grid_size):
        if canvas is None:
            canvas = draw.white_img(self.grid_world.shape)
        self.grid_world.render(canvas, grid_size)
        # Render goal
        goal_top_left = self.goal_pose * grid_size
        draw.rectangle(canvas, goal_top_left, goal_top_left + grid_size,
                       color=draw.color_from_rgb((0, 255, 0)),
                       thickness=-1)

        # Render agent
        top_left = self.pose * grid_size
        draw.rectangle(canvas, top_left, top_left + grid_size,
                       color=draw.color_from_rgb((255, 0, 0)),
                       thickness = -1)
        return canvas

    def episode_reset(self):
        self.steps = 0

    def done(self):
        return self.steps >= self.max_steps

        
if __name__ == '__main__':
    agent = AgentInGridWorld(0,
        grid_world  = WindyGridWorld(0, WindyGridWorld.default_maze()),
        pose        = [1, 1],
        goal_pose   = [3, 4],
        goal_reward = 10)

    direc_ = dict(w=0, a=1, d=2, x=3)
    k = np.random.randint(4)
    for i in range(10):
        pose = agent.step(k)
        rew = agent.reward()
        print(f"rew = {rew}")
        cnvs = agent.render(canvas=draw.white_img(agent.grid_world.shape), grid_size=100)
        draw.imshow("c", cnvs)
        k = direc_[draw.waitKey(-1)]
