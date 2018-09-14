from functools import partial

import numpy as np

from umcog.confutils import xargspartial, xargsonce, xargs, KWProp

from ..alg.common import egreedy_prob_exp
from ..alg.qlearning import QLearningConcatenated, QLearningDiscrete
from ..alg.floyd_warshall_grid import FloydWarshallAlgDiscrete
from ..game.play import MultiObserver, train_episode, test_episode
from ..prob.windy_grid_world import (AgentInGridWorld, wrap_step_done)
from ..plots.vis_action_value_maze import VisMazeActionValueObsFromAlg

from .default import (PROJECT_NAME, grid_world_play)


def hyphenjoin(algname = "algname", maze_name = "prob"):
    return "-".join((algname, maze_name))


def fixed_goal_pose_gen(iter_, prob):
    ret = np.array(next(iter_))
    return ret


def fixed_start_pose_gen(iter_, prob, goal_pose):
    ret = np.array(next(iter_))
    return ret


LogVisMultiObserverXargs = xargs(
    partial(MultiObserver,
            observer_keys = """logging_observer visualizer_observer""".split()),
    """log_file_dir log_file_path prob
    logger_factory logging_encdec windy_grid_world visualizer_observer
    nepisodes""".split())


qlcat_grid_world_train = partial(
    grid_world_play,
    nepisodes         = 2,
    maze_name         = "h-maze",
    shape             = (15, 15),
    prob              = xargsonce(AgentInGridWorld.from_random_maze,
                                  ["shape", "goal_pose_gen",
                                   "start_pose_gen", "wrap_step"]),
    goal_pose_gen     = KWProp(
        lambda s: partial(fixed_goal_pose_gen,
                          iter([(2, 2), (2, 0)]))),
    start_pose_gen     = KWProp(
        lambda s: partial(fixed_start_pose_gen,
                          iter([(0, 0), (0, 2)]))),
    wrap_step         = wrap_step_done,
    project_name      = PROJECT_NAME,
    algname           = "qlcat",
    confname          = xargs(hyphenjoin, ["algname", "maze_name"]),
    egreedy_prob      = xargspartial(egreedy_prob_exp,
                                     dict(nepisodes="max_steps")),
    play_episode      = partial(train_episode, max_steps = 40),
    observer          = LogVisMultiObserverXargs,
    image_file_fmt_t  = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
    visualizer_observer = xargs(VisMazeActionValueObsFromAlg,
                                ["log_file_dir", "windy_grid_world",
                                 "image_file_fmt_t"]),
    alg               = xargsonce(QLearningDiscrete,
                                  """action_space observation_space
                                  reward_range rng egreedy_prob""".split()))

qlcat_grid_world_test = partial(
    qlcat_grid_world_train,
    alg = None,
    nepisodes         = 1,
    play_episode      = partial(test_episode, max_steps = 40),
    goal_pose_gen     = KWProp(
        lambda s: partial(fixed_goal_pose_gen,
                          iter([(2, 2)]))),
    start_pose_gen     = KWProp(
        lambda s: partial(fixed_start_pose_gen,
                          iter([(0, 2)]))),

)


def train_and_test(train_fun, test_fun, kw = dict()):
    obs = train_fun(**kw)
    test_fun(alg = obs.observers['logging_observer'].alg, **kw)

qlcat_grid_world_train_and_test = partial(
    train_and_test,
    train_fun = qlcat_grid_world_train,
    test_fun = qlcat_grid_world_test)

fw_grid_world_train = partial(
    qlcat_grid_world_train,
    qlearning = xargsonce(QLearningConcatenated,
                          """action_space observation_space
                          reward_range rng egreedy_prob""".split()),
    algname = "fw",
    alg  = xargsonce(FloydWarshallAlgDiscrete, ["qlearning"]))


fw_grid_world_test = partial(
    qlcat_grid_world_test,
    qlearning = xargsonce(QLearningConcatenated,
                          """action_space observation_space
                          reward_range rng egreedy_prob""".split()),
    algname = "fw",
    alg  = xargsonce(FloydWarshallAlgDiscrete, ["qlearning"]))

fw_grid_world_train_and_test = partial(
    train_and_test,
    train_fun = fw_grid_world_train,
    test_fun = fw_grid_world_test)


def runall(fs):
    return [f() for f in fs]


grid_world_train_and_test_all = partial(
    runall,
    fs = [qlcat_grid_world_train_and_test, fw_grid_world_train_and_test])

main = grid_world_train_and_test_all
