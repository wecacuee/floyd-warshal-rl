#!/usr/bin/python3
from __future__ import absolute_import, division, print_function
from pathlib import Path
import numpy as np
import functools
import os

from cog import draw
from cog.memoize import MEMOIZE_METHOD
from cog.confutils import (extended_kwprop, KWProp as prop, xargs, xargspartial, xargmem)

from game.play import (Space, Problem, NoOPObserver,
                       post_process_data_iter,
                       post_process_generic, LogFileReader)
from game.logging import NPJSONEncDec
import logging
from .generate_mazes import gen_maze

def logger():
    return logging.getLogger(__name__)

def maze_from_filepath(fpath):
    with open(fpath) as f:
        return maze_from_file(f)

def maze_from_file(f):
    return maze_from_string("".join(f.readlines()))

def maze_from_string(maze_string,
                     intable = ". +^<>V",
                     outtable = "0012345",
                     firstchar = "0",
                     defaultchar = "0",
                     breakline = "\n",
                     encoding = "ascii"):
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
    maze_string = maze_string.translate(str.maketrans(intable, outtable))
    maze_bytes = maze_string.encode(encoding)
    maze_lines = list(filter(len, maze_string.split(breakline))) # keep only non-empty lines
    nrows = len(maze_lines)
    ncols = max(map(len, maze_lines))
    for line in maze_lines:
        if len(line) < ncols:
            line += defaultchar * (ncols - len(line))
    return (np.frombuffer(''.join(maze_lines).encode(encoding), dtype='u1'
                ).reshape(nrows, ncols)
            - ord(firstchar))

def random_maze_from_shape(shape):
    maze = gen_maze(shape[0], shape[1])
    # make walls as 1 and free space as 0s
    return 1 - maze

class Act2DSpace(Space):
    def __init__(self,
                 rng,
                 NAMES = ["NORTH", "EAST", "WEST",  "SOUTH"],
                 VECTORS = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]])):
        self.rng = rng
        self.NAMES = NAMES
        self.VECTORS = VECTORS

    def sample(self):
        act = self.rng.randint(self.size)
        return act

    def contains(self, x):
        return x in self.values()

    def values(self):
        return range(self.size)

    @property
    def size(self):
        return len(self.NAMES)

    def tovector(self, x):
        return self.VECTORS[x]


class Loc2DSpace(Space):
    def __init__(self, lower_bound, upper_bound, rng):
        self.rng = rng
        self.lower_bound = np.asarray(lower_bound)
        self.upper_bound = np.asarray(upper_bound)

    def sample(self):
        return np.array([
            self.rng.randint(self.lower_bound[0], self.upper_bound[0]),
            self.rng.randint(self.lower_bound[1], self.upper_bound[1])])

    def contains(self, x):
        ndim = self.lower_bound.ndim
        if x.shape[-ndim:] != self.lower_bound.shape:
            return False
        xint = np.int64(x)
        return np.all((self.lower_bound <= x) & (x < self.upper_bound))

    def values(self):
        return  (tuple(np.array(index) + self.lower_bound)
                 for index in np.ndindex(
                         tuple(self.upper_bound - self.lower_bound)))


class WindyGridWorld(object):
    def __init__(self,
                 rng,
                 maze,
                 wind_strength=0.5,
                 CELL_FREE = 0,
                 CELL_WALL = 1,
                 CELL_WIND_NEWS = [2, 3, 4, 5],
                 WIND_NEWS_NAMES = "NORTH EAST WEST SOUTH".split(),
                 WIND_NEWS_VECTORS = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]]),
    ):
        self.rng = rng
        self.maze = maze
        self.wind_strength = wind_strength
        self.CELL_FREE = CELL_FREE
        self.CELL_WALL = CELL_WALL
        self.CELL_WIND_NEWS = CELL_WIND_NEWS
        self.WIND_NEWS_NAMES = WIND_NEWS_NAMES
        self.WIND_NEWS_VECTORS = WIND_NEWS_VECTORS

    @classmethod
    @extended_kwprop
    def from_maze_string(cls,
                         seed = 0,
                         maze_string = "\n".join([
                             "  +  ",
                             "  +  ",
                             " ^^^ ",
                             " ^^^ ",
                             "  +  "]),
                         maze = prop(lambda s: maze_from_string(s.maze_string)),
                         rng  = prop(lambda s: np.random.RandomState(s.seed)),
                         **kwargs
    ):

        return cls(rng = rng, maze = maze, **kwargs)

    @classmethod
    @extended_kwprop
    def from_maze_file_path(cls,
                            maze_file_path = None,
                            seed = 0,
                            maze = prop(lambda s: maze_from_filepath(s.maze_file_path)),
                            rng  = prop(lambda s: np.random.RandomState(s.seed)),
                            **kwargs
    ):
        return cls(rng = rng, maze = maze, **kwargs)


    @classmethod
    @extended_kwprop
    def from_random_maze(cls,
                         shape,
                         seed = 0,
                         maze = prop(lambda s: random_maze_from_shape(s.shape)),
                         rng  = prop(lambda s: np.random.RandomState(s.seed)),
                         **kwargs
    ):
        return cls(rng = rng, maze = maze, **kwargs)

    def wind_dir(self, xy):
        row, col = xy[::-1]
        return self.wind_vectors(self.maze[row, col])

    def next_pose(self, pose, act_vec):
        wind_prob = 1 if (self.rng.uniform() < self.wind_strength) else 0
        potential_pose = (pose + act_vec
                          + wind_prob * self.wind_dir(pose))
        next_p, hit_wall = ((pose, True)
                            if self.iswall(potential_pose)
                            else (potential_pose, False))
        return next_p, hit_wall

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
            canvas = draw.white_img(np.array(self.maze.shape) * grid_size)
        grid_size = canvas.get_xlim()[1] / self.shape[1]

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
                pt1 = center - wind_dir * grid_size / 8
                pt2 = center + wind_dir * grid_size / 8
                draw.arrowedLine(
                    canvas, pt1, pt2,
                    draw.color_from_rgb((0,0,0)), thickness=5, tipLength=10)
                
        return canvas

    def random_pos(self):
        x = self.rng.randint(0, self.shape[1])
        y = self.rng.randint(0, self.shape[0])
        return np.array((x,y))

    def valid_random_pos(self):
        pos = self.random_pos()
        while self.iswall(pos):
            pos = self.random_pos()

        return pos

    @property
    def shape(self):
        return self.maze.shape


def random_goal_pose_gen(prob):
    return prob.grid_world.valid_random_pos()


def random_start_pose_gen(prob, goal_pose):
    start_pose = goal_pose
    while np.all(start_pose == goal_pose):
        start_pose = prob.grid_world.valid_random_pos()
    return start_pose

def render_block(ax, pose, cellsize, color):
    pose_top_left = pose * cellsize
    draw.rectangle(ax, pose_top_left, pose_top_left + cellsize,
                   color=color, thickness = -1)
    return ax


def render_agent(ax, pose, cellsize, color=draw.color_from_rgb((0, 0, 255))):
    return render_block(ax, pose, cellsize, color)


def render_goal(ax, pose, cellsize, color=draw.color_from_rgb((0, 255, 0))):
    return render_block(ax, pose, cellsize, color)

def render_agent_grid_world(canvas, grid_world, agent_pose, goal_pose,
                            cellsize,
                            render_goal = render_goal,
                            render_agent = render_agent):
    grid_size = cellsize
    grid_world.render(canvas, grid_size)
    # Render goal
    render_goal(canvas, goal_pose, grid_size)

    # Render agent
    render_agent(canvas, agent_pose, grid_size)
    return canvas

class AgentInGridWorld(Problem):
    def __init__(self,
                 grid_world,
                 action_space,
                 observation_space,
                 start_pose_gen = random_start_pose_gen,
                 goal_pose_gen = random_goal_pose_gen,
                 max_steps = 300,
                 wall_penality = 1.0,
                 no_render = False,
                 log_file_dir = "/tmp/",
                 reward_range = (1, 10),
    ):
        self.grid_world        = grid_world
        self.goal_pose_gen     = goal_pose_gen
        self.start_pose_gen    = start_pose_gen
        self.action_space      = action_space #Act2DSpace(seed)
        self.reward_range      = reward_range
        self.max_steps         = max_steps
        self._hit_wall_penality = wall_penality
        self._last_reward      = 0
        self.no_render         = no_render
        self.log_file_dir      = log_file_dir
        self.observation_space = observation_space # Obs2DSpace
        #Loc2DSpace(
        #    lower_bound = np.array([0, 0]),
        #    upper_bound = np.array(grid_world.shape),
        #    seed        = seed)
        self.episode_reset(0)

    @property
    def goal_reward(self):
        return self.reward_range[1]

    @classmethod
    @extended_kwprop
    def from_maze_file_path(cls,
                            maze_file_path    = None,
                            seed              = 0,
                            rng               = prop(lambda s: np.random.RandomState(s.seed)),
                            action_space      = xargs(Act2DSpace, ["rng"]),
                            observation_space = xargs(Loc2DSpace,
                                                      "rng lower_bound upper_bound".split()),
                            lower_bound       = np.array([0, 0]),
                            upper_bound       = prop(lambda s : s.grid_world.shape),
                            grid_world        = xargs(WindyGridWorld.from_maze_file_path,
                                               "rng maze_file_path".split()),
                            **kwargs):
        return cls(grid_world = grid_world,
                   action_space = action_space,
                   observation_space = observation_space, **kwargs)

    @classmethod
    @extended_kwprop
    def from_random_maze(cls,
                         shape,
                         seed              = 0,
                         rng               = prop(lambda s: np.random.RandomState(s.seed)),
                         action_space      = xargs(Act2DSpace, ["rng"]),
                         observation_space = xargs(Loc2DSpace,
                                                   "rng lower_bound upper_bound".split()),
                         lower_bound       = np.array([0, 0]),
                         upper_bound       = prop(lambda s : s.grid_world.shape),
                         grid_world        = xargs(WindyGridWorld.from_random_maze,
                                                   "rng shape".split()),
                         **kwargs):
        return cls(grid_world = grid_world,
                   action_space = action_space,
                   observation_space = observation_space, **kwargs)

    @property
    def grid_shape(self):
        return self.grid_world.shape

    def imagine_step(self, pose, act):
        """ Imagine taking a step but do not modify any state variables """
        pose, hit_wall = self.grid_world.next_pose(
            pose,
            self.action_space.tovector(act))
        return pose, hit_wall

    def step(self, act):
        if self.hit_goal():
            self._respawn()
            self._last_reward = 0
        else:
            old_pose = self.pose
            self.pose, hit_wall = self.imagine_step(self.pose, act)
            self._last_reward = -self._hit_wall_penality if hit_wall else 0
        self.steps += 1
        return self.pose, self.reward()

    def _respawn(self):
        self.pose          = self.start_pose_gen(self, self.goal_pose)

    def hit_goal(self):
        return np.all(self.goal_pose == self.pose)

    def reward(self):
        return (self.goal_reward if self.hit_goal() else self._last_reward)

    def observation(self):
        return self.pose

    def render(self, canvas, grid_size, wait_time=0):
        if self.no_render:
            return canvas

        if canvas is None:
            canvas = draw.white_img(np.array(self.grid_world.shape) * 100)
        canvas = render_agent_grid_world(
            canvas, self.grid_world, self.pose, self.goal_pose,
            grid_world)
        if wait_time != 0:
            draw.imshow(self.__class__.__name__, canvas)
        if self.log_file_dir is not None:
            draw.imwrite(
                str(
                    Path(self.log_file_dir) / "{name}_{episode_n}_{step}.pdf".format(
                        name = self.__class__.__name__,
                        episode_n = self.episode_n,
                        step=self.steps)),
                canvas)
        return canvas

    def episode_reset(self, episode_n):
        self.steps         = 0
        self._last_reward  = 0
        self.goal_pose     = self.goal_pose_gen(self)
        self.pose          = self.start_pose_gen(self, self.goal_pose)
        self.episode_n     = episode_n

    def done(self):
        return self.steps >= self.max_steps

    @MEMOIZE_METHOD
    def valid_nbrs(self, pos, samples=10):
        nbrs = set()
        for _ in range(samples):
            for a in self.action_space.values():
                valid_pose, _ = self.imagine_step(np.asarray(pos), a)
                if np.any(valid_pose != np.asarray(pos)):
                    nbrs.add(tuple(valid_pose.tolist()))
        return nbrs

    def shortest_path_length(self, start, end, visited=None):
        assert start is not None and end is not None
        if visited is None:
            visited = set()
        visited.add(tuple(start))

        if np.all(start == end):
            return 0
        else:
            unvisited_nbrs = self.valid_nbrs(start) - visited
            length = min(
                [self.shortest_path_length(nbr, end, visited | unvisited_nbrs)
                 for nbr in unvisited_nbrs],
                default = np.inf)
            return length + 1#, [tuple(start)] + path


class DrawAgentGridWorldFromLogs:
    def __init__(self):
        self.new_episode_data = None

    def __call__(self, data = None, tag = None, windy_grid_world=None,
                 cellsize=None, image_file_fmt=None):
        if tag == "LoggingObserver:new_episode":
            self.new_episode_data = data
        elif tag == "LoggingObserver:new_step":
            if  self.new_episode_data is not None:
                assert self.new_episode_data["episode_n"] == data["episode_n"]
                ax = windy_grid_world.render(None, cellsize)
                # Render goal
                render_goal(ax, np.asarray(self.new_episode_data["goal_pose"]), cellsize)

                # Render agent
                render_agent(ax, np.asarray(data["pose"]), cellsize)
                img_filepath = image_file_fmt.format(
                    tag = "agent_grid_world",
                    episode=data["episode_n"], step=data["steps"])
                img_filedir = os.path.dirname(img_filepath)
                if not os.path.exists(img_filedir):
                    os.makedirs(img_filedir)

                draw.imwrite(img_filepath, ax)
        else:
            print("Skipping Unknown tag {}".format(tag))

class AgentVisObserver(NoOPObserver):
    @extended_kwprop
    def __init__(self,
                 windy_grid_world        = xargs(
                     WindyGridWorld.from_maze_file_path,
                     "rng maze_file_path".split()),
                 cellsize                = 80,
                 process_data_tag = xargspartial(
                     DrawAgentGridWorldFromLogs(),
                     "windy_grid_world cellsize image_file_fmt".split()),
                 image_file_fmt_template = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
                 image_file_fmt          = prop(lambda s: s.image_file_fmt_template.format(self=s)),
                 log_file_reader = xargs(LogFileReader,
                                         ["log_file_path"], enc = NPJSONEncDec()),
                 rng                     = xargmem(np.random.RandomState, ["seed"]),
                 seed                    = 0,
                 data_iter               = xargspartial(
                     post_process_data_iter,
                     "log_file_reader filter_criteria".split()),
                 filter_criteria         = dict(
                     tag = ['LoggingObserver:new_episode', 'LoggingObserver:new_step']),
                 post_process = xargspartial(
                     post_process_generic,
                     "data_iter process_data_tag".split()),
                 # Needs: log_file_path, log_file_dir, maze_file_path
    ):
        self.post_process = post_process

    def on_play_end(self):
        logging.shutdown()
        self.post_process()

if __name__ == '__main__':
    agent = AgentInGridWorld(
        np.random.RandomState(0),
        grid_world     = WindyGridWorld(0, WindyGridWorld.default_maze()),
        start_pose_gen = lambda s, g : [1, 1],
        goal_pose_gen  = lambda s : [3, 4],
        goal_reward    = 10,
        max_steps      = 100,
        wall_penality  = 1,
        no_render      = False,
        action_space   = Act2DSpace()
    )

    direc_ = dict(w=0, a=1, d=2, x=3)
    k = np.random.randint(4)
    for i in range(10):
        pose = agent.step(k)
        rew = agent.reward()
        print("rew = {rew}".format(rew=rew))
        grid_size=100
        cnvs = agent.render(
            canvas=draw.white_img(np.array(agent.grid_world.shape) * grid_size),
            grid_size=grid_size)
        draw.imshow("c", cnvs)
        k = direc_[draw.waitKey(-1)]
