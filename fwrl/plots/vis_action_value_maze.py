from functools import partial

import numpy as np

from umcog import draw
from umcog.confutils import (extended_kwprop, xargsonce, KWProp, alias,
                             xargspartial)

from fwrl.game.play import LogFileReader, NoOPObserver
from fwrl.game.logging import NPJSONEncDec, LogFileConf, find_latest_file
from fwrl.conf.default import PROJECT_NAME
from fwrl.alg.qlearning import Renderer, QLearningVis
from fwrl.alg.floyd_warshall_grid import visualize_action_value
from fwrl.prob.windy_grid_world import (WindyGridWorld, render_goal,
                                        render_agent)


def numpy_to_callable(arr, *a):
    return arr[a] if arr.size else None


class VisMazeActionValueObs(NoOPObserver):
    @extended_kwprop
    def __init__(self,
                 windy_grid_world = None,
                 log_file_dir = None,
                 image_file_fmt_t=
                 "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.pdf",
                 tag = "action_value_on_maze",
                 cellsize = 60,
                 image_file_fmt = KWProp(
                     lambda s: s.image_file_fmt_t.format(self=s)),
                 renderer = xargspartial(Renderer.log, ["image_file_fmt"])):

        self.goal_obs = None
        self.pose_history = []
        self.action_value_dct = None
        self.windy_grid_world = windy_grid_world
        self.renderer = renderer
        self.cellsize = cellsize
        self.tag = tag

    def on_new_episode(self, **dct):
        self.goal_obs = dct.get("goal_obs", dct.get("goal_pose"))
        self.pose_history = [dct.get("obs", dct.get("goal_obs"))]
        print("episodes {}".format(dct))

    def on_new_step(self, **dct):
        self.pose_history.append(dct.get("obs", dct.get("pose")))

    def render(self):
        grid_shape = self.action_value_dct["grid_shape"]
        cellsize = self.cellsize
        pose_history = self.pose_history
        renderer = self.renderer
        action_value_dct = self.action_value_dct
        goal_obs = self.goal_obs
        ax = draw.white_img(
            (grid_shape[1]*cellsize, grid_shape[0]*cellsize),
            dpi=cellsize)

        hash_state = action_value_dct["hash_state"]
        assert action_value_dct["net_value"].shape[0] == len(hash_state)
        assert action_value_dct["net_value"].shape[0] == (max(hash_state.values())+1)
        net_value = action_value_dct["net_value"]
        net_value[np.isinf(net_value)] = -100
        action_value_mat = QLearningVis.action_value_to_value_mat(
            partial(numpy_to_callable, net_value),
            action_value_dct["hash_state"],
            action_value_dct["grid_shape"])
        draw.matshow(ax, QLearningVis.normalize_by_std(action_value_mat))
        ax = self.windy_grid_world.render(ax, cellsize)

        # Render goal
        render_goal(ax, np.asarray(goal_obs), cellsize)

        # Render agent
        render_agent(ax, np.asarray(pose_history[0]), cellsize)
        prev_center = (np.array(pose_history[0]) + 0.5) * cellsize
        for pose in pose_history[1:-1]:
            center = (np.array(pose) + 0.5) * cellsize
            draw.arrowedLine(ax, prev_center, center, (0, 0, 0),
                             thickness=2, tipLength=10)
            prev_center = center
        #draw.imshow("c", ax)

        renderer(ax, action_value_dct, tag = self.tag)

    def on_goal_hit(self, **dct):
        pose = dct.get("obs", dct.get("pose"))
        if pose is not None:
            self.pose_history.append(pose)
        else:
            self.pose_history.append(self.goal_obs)

        self.pose_history = []

    def on_action_value(self, **dct):
        self.action_value_dct = dct

    def on_episode_end(self, **dct):
        self.render()


class VisMazeActionValueObsFromAlg(VisMazeActionValueObs):
    def on_episode_end(self, **dct):
        self.on_action_value()
        super(VisMazeActionValueObsFromAlg, self).on_episode_end(**dct)

    def on_action_value(self, **dct):
        self.action_value_dct = self.alg.get_action_value_dct()
        hash_state = self.action_value_dct["hash_state"]
        assert self.action_value_dct["net_value"].shape[0] == len(hash_state)
        assert self.action_value_dct["net_value"].shape[0] == (max(hash_state.values())+1)
        self.action_value_dct["grid_shape"] = self.prob.grid_shape


@extended_kwprop
def vis_action_value_maze(
        gitrev ="9be6f93",
        alg = "fw",
        run_month = "201809",
        confname = KWProp(lambda s: "-".join((s.alg, s.maze_name))),
        maze_name = "4-room-grid-world",
        windy_grid_world = xargsonce(WindyGridWorld.from_maze_name,
                                     ["maze_name"]),
        project_name = PROJECT_NAME,
        log_file_conf = xargsonce(
            LogFileConf,
            """project_name gitrev run_month confname""".split()),
        log_file_dir = alias(["log_file_conf", "log_file_dir"]),
        logfile = xargsonce(find_latest_file,
                            ["log_file_dir"])):
    obs = VisMazeActionValueObs(
        log_file_conf = log_file_conf, windy_grid_world = windy_grid_world)
    for dct, tag in LogFileReader(logfile, enc = NPJSONEncDec()).read_data():
        if tag == "LoggingObserver:new_episode":
            obs.on_new_episode(**dct)
        elif tag == "LoggingObserver:new_step":
            obs.on_new_step(**dct)
        elif tag == "LoggingObserver:goal_hit":
            obs.on_goal_hit(**dct)
        elif tag == "FloydWarshallLogger:action_value":
            obs.action_value_dct = dct


vis_action_value_rnd_maze = partial(
    vis_action_value_maze,
    confname = KWProp(lambda s: s.alg + "_grid_world_play"),
    shape = (9, 9),
    windy_grid_world = xargsonce(WindyGridWorld.from_random_maze,
                                 ["shape"]), )

vis_action_value_maze_example = partial(
    vis_action_value_maze,
    logfile = "/z/home/dhiman/mid//floyd-warshall-rl/201808_7a324cf_fw-4-room-grid-world/08-021509.log",
    image_file_fmt_t= "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png")
