#!/usr/bin/python3
from __future__ import absolute_import, division, print_function
from pathlib import Path
import numpy as np
import functools
from functools import partial
import os
import pkg_resources
import copy

from umcog import draw
from umcog.memoize import MEMOIZE_METHOD, MethodMemoizer
from umcog.confutils import (extended_kwprop, KWProp as prop, xargs, xargspartial, xargmem)

from ..game.play import (Space, Problem, NoOPObserver,
                       post_process_data_iter,
                       post_process_generic, LogFileReader)
from ..game.logging import NPJSONEncDec
import logging
from .generate_mazes import gen_maze

def logger():
    return logging.getLogger(__name__)

def maze_from_filepath(fpath):
    with open(fpath) as f:
        return maze_from_file(f)

def maze_from_file(f):
    return maze_from_string("".join(f.readlines()))

def maze_from_pkg_rsrc(frsrc):
    return maze_from_string(pkg_resources.resource_string(__name__, frsrc).decode('utf-8'))

def four_room_grid_world(filep="./data/4-room-grid-world.txt"):
    return maze_from_pkg_rsrc(filep)

def maze_from_string(maze_string,
                     intable = ". +^<>VGL",
                     outtable = "001234567",
                     firstchar = "0",
                     defaultchar = "0",
                     breakline = "\n"):
    r"""
    free space: .         -> 0
    wall      : +         -> 1
    Wind      : ^<>V      -> 2,3,4,5
    Goal      : G         -> 6
    Lava      : L         -> 7
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
    encoding = "ascii"
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

def maze_to_string(maze,
                   intable = "001234567",
                   outtable = ". +^<>VGL",
                   firstchar = "0",
                   defaultchar = "0",
                   breakline = "\n",
                   encoding = "utf-8"):
    r"""
    >>> maze = np.array(
    ...              [[0, 0, 1, 0, 0],
    ...               [0, 0, 2, 0, 0],
    ...               [0, 2, 2, 2, 0],
    ...               [0, 2, 2, 2, 0],
    ...               [0, 0, 1, 0, 0]], dtype='u1')
    >>> maze_to_string(maze)
    '  +  \n  ^  \n ^^^ \n ^^^ \n  +  \n'
    """
    # Add new lines
    maze_nl = np.hstack((maze + ord(firstchar),
                         ord(breakline) * np.ones((maze.shape[0], 1), dtype='u1')))
    # Convert to string
    maze_str = maze_nl.tostring().decode("utf-8")
    # Translate to visually meaningful characters
    maze_str_tl = maze_str.translate(str.maketrans(intable, outtable))
    return maze_str_tl


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

    @property
    def size(self):
        return (self.upper_bound - self.lower_bound)

    def sample(self):
        return np.array([
            self.rng.randint(self.lower_bound[0], self.upper_bound[0]),
            self.rng.randint(self.lower_bound[1], self.upper_bound[1])])

    def contains(self, x):
        if x.tolist() is None:
            return True
        ndim = self.lower_bound.ndim
        if x.shape[-ndim:] != self.lower_bound.shape:
            return False
        xint = np.int64(x)
        return np.all((self.lower_bound <= x) & (x < self.upper_bound))

    def values(self):
        return  (tuple(np.array(index) + self.lower_bound)
                 for index in np.ndindex(
                         tuple(self.upper_bound - self.lower_bound)))

class FreeSpaceHandler:
    def __init__(self, CELL_FREE):
        self.CELL_FREE = CELL_FREE

    def handles(self, cell_code, potential_cell_code):
        return cell_code == self.CELL_FREE

    def step(self, pose, potential_next_pose, potential_reward, potential_done,
             cell_code, potential_cell_code):
        return potential_next_pose, potential_reward, potential_done

    def render(self, canvas, pose, grid_size, cell_code,
               color = draw.color_from_rgb((255,255,255)), thickness = -1):
        render_block(canvas, pose, grid_size, color = color)


class WindHandler:
    def __init__(self,
                 wind_strength,
                 CELL_WIND_NEWS,
                 WIND_NEWS_NAMES = "NORTH EAST WEST SOUTH".split(),
                 WIND_NEWS_VECTORS = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]])):
        for k in "wind_strength CELL_WIND_NEWS WIND_NEWS_VECTORS WIND_NEWS_NAMES".split():
            setattr(self, k, locals()[k])

    def handles(self, cell_code, potential_cell_code):
        return cell_code in  self.CELL_WIND_NEWS

    def _wind_vectors(self, cell_code):
        return self.WIND_NEWS_VECTORS[cell_code - self.CELL_WIND_NEWS[0]]

    def _step(self, potential_next_pose, cell_code):
        wind_prob = 1 if (self.rng.uniform() < self.wind_strength) else 0
        potential_next_pose = (potential_next_pose
                               + wind_prob * self._wind_vectors(cell_code))
        return potential_next_pose

    def step(self, pose, potential_next_pose, potential_reward, potential_done,
             cell_code, potential_cell_code):
        return self._step(potential_next_pose, cell_code), potential_reward, potential_done

    def render(self, canvas, pose, grid_size, cell_code,
               wind_color = draw.color_from_rgb((0,0,0)),
               thickness = 5,
               tipLength = 10):
        render_block(canvas, pose, grid_size, color = draw.color_from_rgb((255, 255, 255)))
        pose_top_left = pose * grid_size
        center = pose_top_left + 0.5 * grid_size
        wind_dir = self._wind_vectors(cell_code)
        pt1 = center - wind_dir * grid_size / 8
        pt2 = center + wind_dir * grid_size / 8
        draw.arrowedLine(
            canvas, pt1, pt2,
            wind_color, thickness=thickness, tipLength=tipLength)


class GoalHandler:
    def __init__(self, goal_reward, GOAL_CELL_CODE, goal_next_pose = np.array(None)):
        self.goal_next_pose = goal_next_pose
        self.goal_reward = goal_reward
        self.GOAL_CELL_CODE = GOAL_CELL_CODE

    def handles(self, cell_code, potential_cell_code):
        return cell_code == self.GOAL_CELL_CODE

    def step(self, pose, potential_next_pose, potential_reward, potential_done,
             cell_code, potential_cell_code):
        potential_next_pose = self.goal_next_pose
        potential_reward    = self.goal_reward
        potential_done      = False
        return potential_next_pose, potential_reward, potential_done

    def render(self, canvas, pose, grid_size, cell_code):
        # Render goal
        render_goal(canvas, pose, grid_size)


class WallHandler:
    def __init__(self, wall_reward, WALL_CELL_CODE):
        self.wall_reward = wall_reward
        self.WALL_CELL_CODE = WALL_CELL_CODE

    def handles(self, cell_code, potential_cell_code):
        return potential_cell_code == self.WALL_CELL_CODE

    def step(self, pose, potential_next_pose, potential_reward, potential_done,
             cell_code, potential_cell_code):
        potential_next_pose = pose
        potential_reward    += self.wall_reward
        return potential_next_pose, potential_reward, potential_done

    def render(self, canvas, pose, grid_size, cell_code,
               color=draw.color_from_rgb((0,0,0)),
               thickness = -1):
        render_block(canvas, pose, grid_size, color)

class LavaHandler:
    def __init__(self, lava_reward, LAVA_CELL_CODE):
        self.lava_reward = lava_reward
        self.LAVA_CELL_CODE = LAVA_CELL_CODE

    def handles(self, cell_code, potential_cell_code):
        return potential_cell_code == self.LAVA_CELL_CODE

    def step(self, pose, potential_next_pose, potential_reward, potential_done,
             cell_code, potential_cell_code):
        potential_next_pose = np.array(None)
        potential_reward    += self.lava_reward
        potential_done      = True
        return potential_next_pose, potential_reward, potential_done

    def render(self, canvas, pose, grid_size, cell_code,
               color=draw.color_from_rgb((255,0,0)),
               thickness = -1):
        render_block(canvas, pose, grid_size, color)


def np_tuple(arr):
    return tuple(arr.tolist())


def shortest_path(grid_world, start, end, action_space, visited=None):
    assert start is not None and end is not None
    if visited is None:
        visited = set()
    visited.add(np_tuple(start))

    if np.all(start == end):
        return 0, [np_tuple(end)]
    else:
        unvisited_nbrs = grid_world.neighbors(start, action_space) - visited
        if not len(unvisited_nbrs):
            return np.inf, []
        else:
            length, path = min(
                [shortest_path(grid_world, np.asarray(nbr), end,
                            action_space, visited | unvisited_nbrs)
                    for nbr in unvisited_nbrs],
                key = lambda a:a[0],
                default = np.inf)
            if np.isinf(length):
                return length, []
            else:
                return length + 1, [np_tuple(start)] + path


class WindyGridWorld:
    @extended_kwprop
    def __init__(self,
                 rng,
                 maze,
                 CELL_FREE = 0,
                 OUTSIDE_GRID_CODE = 1,
                 wind_strength = 0.5,
                 wall_reward = 0,
                 goal_reward = 10,
                 lava_reward = -10,
                 CELL_WIND_NEWS = [2, 3, 4, 5],
                 WALL_CELL_CODE = 1,
                 GOAL_CELL_CODE = 6,
                 LAVA_CELL_CODE = 7,
                 freespace_handler = xargs(
                     FreeSpaceHandler, ["CELL_FREE"]),
                 wind_handler = xargs(
                     WindHandler, ["wind_strength", "CELL_WIND_NEWS"]),
                 wall_handler = xargs(
                     WallHandler, ["wall_reward", "WALL_CELL_CODE"]),
                 goal_handler = xargs(
                     GoalHandler, ["goal_reward", "GOAL_CELL_CODE"]),
                 lava_handler = xargs(
                     LavaHandler, ["lava_reward", "LAVA_CELL_CODE"]),
                 handler_keys = """freespace_handler wind_handler wall_handler
                                   goal_handler lava_handler""".split(),
                 handlers = prop(lambda s: [
                     getattr(s, h) for h in s.handler_keys
                 ])
    ):
        self.rng = rng
        self.maze = maze
        self.CELL_FREE = CELL_FREE
        self.OUTSIDE_GRID_CODE = OUTSIDE_GRID_CODE
        self.CELL_WIND_NEWS = CELL_WIND_NEWS
        self.WALL_CELL_CODE = WALL_CELL_CODE
        self.GOAL_CELL_CODE = GOAL_CELL_CODE
        self.LAVA_CELL_CODE = LAVA_CELL_CODE
        self.handlers = handlers

    # TODO: make these function as partials on the constructor
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
    def from_maze_name(cls,
                       maze_name = "4-room-grid-world.txt",
                       seed = 0,
                       maze = prop(lambda s: maze_from_pkg_rsrc("data/" + s.maze_name)),
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

    def cell_code(self, pose):
        row, col = pose[::-1]
        if np.any(np.asarray(pose) < 0):
            return self.OUTSIDE_GRID_CODE
        try:
            return self.maze[row, col]
        except IndexError as e:
            return self.OUTSIDE_GRID_CODE

    def set_maze_code(self, pose, code):
        self.maze[pose[1], pose[0]] = code

    def step(self, pose, potential_next_pose):
        potential_reward = 0
        potential_done = False
        cell_code = self.cell_code(pose)
        for h in self.handlers:
            potential_cell_code = self.cell_code(potential_next_pose)
            if h.handles(cell_code, potential_cell_code):
                potential_next_pose, potential_reward, potential_done = h.step(
                    pose, potential_next_pose, potential_reward, potential_done,
                    cell_code, potential_cell_code)
            if potential_next_pose.tolist() is None:
                break
        return potential_next_pose, potential_reward, potential_done

    def render(self, canvas, grid_size, mode=None):
        if canvas is None:
            canvas = draw.white_img(np.array(self.maze.shape) * grid_size)
        grid_size = canvas.get_xlim()[1] / self.shape[1]

        nrows, ncols = self.maze.shape
        for r, c in np.ndindex(*self.maze.shape):
            pose = np.array([c, r])
            bottom_left = pose * grid_size
            cell_code = self.cell_code(pose)
            for h in self.handlers:
                if h.handles(cell_code, cell_code):
                    h.render(canvas, pose, grid_size, cell_code)

        if mode == 'human':
            draw.imshow(self.__class__.__name__, canvas)
        return canvas

    def random_pos(self):
        x = self.rng.randint(0, self.shape[1])
        y = self.rng.randint(0, self.shape[0])
        return np.array((x,y))

    def valid_random_pos(self,
                         valid_codes = lambda gw: [gw.CELL_FREE] + gw.CELL_WIND_NEWS):
        pos = self.random_pos()
        while self.cell_code(pos) not in valid_codes(self):
            pos = self.random_pos()

        return pos

    @property
    def shape(self):
        return self.maze.shape

    def clone(self):
        c = copy.copy(self)
        c.maze = copy.copy(self.maze)
        return c

    @MethodMemoizer(a_key = lambda a: np_tuple(a[0]))
    def neighbors(self, pos, action_space, samples=10):
        """
        pos: np.ndarray
        action_space: should have methods
            values()    : -> iterator of actions
            tovector(a) : action -> np.ndarray (addable to pos)

        Returns:
        nbrs: set of pose tuples which are convertable to pos (ndarray) with np.asarray
        """
        nbrs = set()
        # take actions 10 times, for deterministic actions 1 time is enough to
        # cover all possibilties
        for _ in range(samples):
            for a in action_space.values():
                avect = action_space.tovector(a)
                valid_pose, _, _ = self.step(pos, pos + avect)
                if (np.any(valid_pose != np.asarray(pos))
                    and valid_pose.tolist() is not None):
                    nbrs.add(np_tuple(valid_pose))
        return nbrs


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
    draw.rectangle(ax, pose_top_left, pose_top_left + cellsize,
                   color=draw.color_from_rgb((0,0,0)), thickness = 1)
    return ax


def render_agent(ax, pose, cellsize, color=draw.color_from_rgb((0, 0, 255))):
    return render_block(ax, pose, cellsize, color)


def render_goal(ax, pose, cellsize, color=draw.color_from_rgb((0, 255, 0))):
    return render_block(ax, pose, cellsize, color)


def render_agent_grid_world(canvas, grid_world, agent_pose,
                            cellsize,
                            render_agent = render_agent):
    grid_size = cellsize
    grid_world.render(canvas, grid_size)
    # Render agent
    if agent_pose.tolist() is not None:
        render_agent(canvas, agent_pose, grid_size)
    return canvas


def canvas_gen(shape):
    return draw.white_img(shape)


def agent_renderer(self, canvas, canvas_gen = canvas_gen, grid_size = 50,
                   wait_time = 10, mode='log'):
    canvas = self._canvas = (
        self._canvas
        if self._canvas is not None else
        draw.white_img(np.array(self.grid_world_goal.shape) * grid_size))
    canvas = render_agent_grid_world(
        canvas, self.grid_world_goal, self.pose, grid_size)
    if wait_time != 0 and mode == 'human':
        draw.imshow(self.__class__.__name__, canvas)
    if self.log_file_dir is not None and mode == 'log':
        draw.imwrite(
            str(
                Path(self.log_file_dir) / "{name}_{episode_n}_{step}.pdf".format(
                    name = self.__class__.__name__,
                    episode_n = self.episode_n,
                    step=self.steps)),
            canvas)
    return canvas

class AgentRenderer:
    default = agent_renderer
    human = partial(agent_renderer, mode = 'human')

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
                 renderer = AgentRenderer.default
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
        self._canvas           = None
        self.renderer  = agent_renderer
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
    def from_maze(cls,
                  maze              = None,
                  seed              = 0,
                  rng               = prop(lambda s: np.random.RandomState(s.seed)),
                  action_space      = xargs(Act2DSpace, ["rng"]),
                  observation_space = xargs(Loc2DSpace,
                                            "rng lower_bound upper_bound".split()),
                  lower_bound       = np.array([0, 0]),
                  upper_bound       = prop(lambda s : s.grid_world.shape),
                  grid_world        = xargs(WindyGridWorld,
                                            "rng maze".split()),
                  **kwargs):
        return cls(grid_world = grid_world,
                   action_space = action_space,
                   observation_space = observation_space, **kwargs)

    @classmethod
    @extended_kwprop
    def from_maze_name(cls,
                       maze_name         = None,
                       seed              = 0,
                       rng               = prop(lambda s: np.random.RandomState(s.seed)),
                       action_space      = xargs(Act2DSpace, ["rng"]),
                       observation_space = xargs(Loc2DSpace,
                                                 "rng lower_bound upper_bound".split()),
                       lower_bound       = np.array([0, 0]),
                       upper_bound       = prop(lambda s : s.grid_world.shape),
                       grid_world        = xargs(WindyGridWorld.from_maze_name,
                                                 "rng maze_name".split()),
                       **kwargs):
        return cls(grid_world = grid_world,
                   action_space = action_space,
                   observation_space = observation_space, **kwargs)


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
        return self.grid_world_goal.shape

    def step(self, act):
        if self._done:
            raise StopIteration("Done here. Please call episode_reset")

        gw_pose, gw_rew, gw_done = self.grid_world_goal.step(
            self.pose, self.pose + self.action_space.tovector(act))
        if gw_pose.tolist() is None or gw_done:
            self._respawn()
        else:
            self.pose = gw_pose

        self._last_reward = gw_rew
        self.steps += 1
        self._done = gw_done or self.steps >= self.max_steps
        return self.pose, self._last_reward

    def _respawn(self):
        self.pose          = self.start_pose_gen(self, self.goal_pose)

    def reward(self):
        return self._last_reward

    def observation(self):
        return self.pose

    def render(self, canvas = None, **kw):
        return self.renderer(self, canvas = canvas, **kw)

    def episode_reset(self, episode_n):
        self.steps         = 0
        self._last_reward  = 0
        self.goal_pose     = self.goal_pose_gen(self)
        self.grid_world_goal = self.grid_world.clone()
        self.grid_world_goal.set_maze_code(
            self.goal_pose, self.grid_world.GOAL_CELL_CODE)
        self.pose          = self.start_pose_gen(self, self.goal_pose)
        self.episode_n     = episode_n
        self._done         = False

    def done(self):
        return self._done


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

    def shortest_path_to_goal(self):
        return shortest_path(self.grid_world, self.pose, self.goal_pose,
                             self.action_space)

def demo_agent():
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

if __name__ == '__main__':
    demo_agent()
