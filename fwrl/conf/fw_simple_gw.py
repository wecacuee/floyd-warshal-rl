import functools
from umcog.confutils import xargs, KWProp as prop, extended_kwprop, xargmem, xargsonce

from ..game.play import (MultiObserver, play, LogFileReader, NoOPObserver)
from ..game.logging import NPJSONEncDec, LogFileConf
from ..prob.windy_grid_world import AgentInGridWorld
from ..alg.fw_simple import FWTabularSimple
from .default import (grid_world_play, AgentVisMultiObserverXargs,
                      AgentVisMultiObserverNoAlgVisXargs,
                      NoVisMultiObserverXargs, PROJECT_NAME, random_state)


@extended_kwprop
def fw_simple_gw_play(
        confname       = "fw_simple_gw_play",
        seed           = 0,
        nepisodes      = 1,
        max_steps      = 500,
        log_file_conf  = xargmem(LogFileConf,
                               "project_name confname".split()),
        rng            = xargs(random_state, ["seed"]),
        log_file_path  = prop(lambda s: s.log_file_conf.log_file),
        logger_factory = prop(lambda s: s.log_file_conf.logger_factory),
        logging_encdec = prop(lambda s: s.log_file_conf.logging_encdec),
        prob           = xargsonce(AgentInGridWorld.from_maze_name,
                                  "rng maze_name max_steps".split()),
        maze_name      = "4-room-grid-world.txt",
        project_name   = PROJECT_NAME,
        image_file_fmt   = prop(
            lambda s: s.image_file_fmt_t.format(self=s)),
        log_file_dir     = prop(
            lambda s: s.log_file_conf.log_file_dir),
        image_file_fmt_t = "{self.log_file_dir}/{{tag}}_{{episode}}_{{step}}.png",
        action_value_img_fmt_t = prop(lambda s : s.image_file_fmt_t),
        alg                 = xargs(FWTabularSimple,
                                    """max_steps action_space observation_space
                                    reward_range rng""".split()),
        action_space        = prop(lambda s : s.prob.action_space),
        observation_space   = prop(lambda s : s.prob.observation_space),
        reward_range        = prop(lambda s : s.prob.reward_range),
        windy_grid_world    = prop(lambda s : s.prob.grid_world),
        observer            = AgentVisMultiObserverNoAlgVisXargs, # if visualize
        #observer            = NoVisMultiObserverXargs, # if no visualize
        visualizer_observer = NoOPObserver,
        logger              = prop(lambda s: s.logger_factory("FWTabularSimpleLogger")),
        log_file_reader     = xargs(LogFileReader, ["log_file_path"],
                                    enc = NPJSONEncDec())):
    return play(alg, prob, observer, nepisodes, logger_factory)

main = fw_simple_gw_play
