from functools import partial

import numpy as np

from umcog import draw
from umcog.confutils import extended_kwprop, xargsonce, KWProp, alias

from fwrl.game.play import LogFileReader
from fwrl.game.logging import NPJSONEncDec, LogFileConf, find_latest_file
from fwrl.conf.default import PROJECT_NAME
from fwrl.alg.qlearning import Renderer, QLearningVis
from fwrl.alg.floyd_warshall_grid import visualize_action_value
from fwrl.prob.windy_grid_world import (WindyGridWorld, render_goal,
                                        render_agent)


def numpy_to_callable(arr, *a):
    return arr[a] if arr.size else None


@extended_kwprop
def vis_action_value_maze(
        gitrev ="9be6f93",
        alg = "fw",
        run_month = "201809",
        confname = KWProp(lambda s: "-".join((s.alg, s.maze_name))),
        maze_name = "4-room-grid-world",
        windy_grid_world = xargsonce(WindyGridWorld.from_maze_name,
                                     ["maze_name"]),
        image_file_fmt_t=
        "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.pdf",
        cellsize = 60,
        project_name = PROJECT_NAME,
        log_file_conf = xargsonce(
            LogFileConf,
            """project_name gitrev run_month confname""".split()),
        log_file_dir = alias(["log_file_conf", "log_file_dir"]),
        logfile = xargsonce(find_latest_file,
                             ["log_file_dir"])):
    goal_obs = None
    pose_history = []

    renderer = partial(
        Renderer.log,
        image_file_fmt = image_file_fmt_t.format(self=log_file_conf))
    #wgw = windy_grid_world_gen(maze_name = maze_name)
    action_value_dct = None
    for dct, tag in LogFileReader(logfile, enc = NPJSONEncDec()).read_data():
        if tag == "LoggingObserver:new_episode":
            goal_obs = dct.get("goal_obs", dct.get("goal_pose"))
            pose_history = []
        elif tag == "LoggingObserver:new_step":
            pose_history.append(dct.get("obs", dct.get("pose")))
        elif tag == "LoggingObserver:goal_hit":
            pose = dct.get("obs", dct.get("pose"))
            if pose is not None:
                pose_history.append(pose)
            else:
                pose_history.append(goal_obs)
            if action_value_dct is not None and goal_obs is not None:
                grid_shape = action_value_dct["grid_shape"]
                ax = draw.white_img(
                    (grid_shape[1]*cellsize, grid_shape[0]*cellsize),
                    dpi=cellsize)
                action_value_mat = QLearningVis.action_value_to_value_mat(
                    partial(numpy_to_callable, action_value_dct["net_value"]),
                    action_value_dct["hash_state"],
                    action_value_dct["grid_shape"])
                draw.matshow(ax,
                             QLearningVis.normalize_by_std(action_value_mat))
                ax = windy_grid_world.render(ax, cellsize)

                # Render goal
                render_goal(ax, np.asarray(goal_obs), cellsize)

                # Render agent
                render_agent(ax, np.asarray(pose_history[1]), cellsize)
                prev_center = (np.array(pose_history[1]) + 0.5) * cellsize
                for pose in pose_history[2:]:
                    center = (np.array(pose) + 0.5) * cellsize
                    draw.arrowedLine(ax, prev_center, center, (0,0,0),
                                     thickness=2, tipLength=10)
                    prev_center = center
                draw.imshow("c", ax)

                renderer(ax, action_value_dct, tag = "action_value_on_maze")
            pose_history = []
        elif tag == "FloydWarshallLogger:action_value":
            action_value_dct = dct


vis_action_value_rnd_maze = partial(vis_action_value_maze,
        confname = KWProp(lambda s: s.alg + "_grid_world_play"),
        shape = (9, 9),
        windy_grid_world = xargsonce(WindyGridWorld.from_random_maze,
                                     ["shape"]),
)
