# coding: utf-8
if not __package__:
    __package__ = "fwrl.plot"

from fwrl.game.play import LogFileReader
from fwrl.game.logging import NPJSONEncDec, LogFileConf, find_latest_file
from fwrl.conf.default import PROJECT_NAME

import matplotlib.pyplot as plt
import numpy as np


def load_data_from_log_file(
        logfile,
        tags = "latencies_all_episodes distineff_all_episodes rewards_all_episodes".split(),
        tag2key = dict(rewards_all_episodes = "total_rewards_all_episodes")):
    file_data = dict()
    for dct, tag in LogFileReader(logfile, NPJSONEncDec()).read_data():
        if tag in tags:
            file_data[tag] = dct[tag2key.get(tag, tag)]
    assert list(file_data.keys()) == tags
    return file_data


def latency(times_to_goal):
    return times_to_goal[0]  / np.mean(times_to_goal[1:])

def latency_all_episodes(time_to_goal_episodes):
    return map(latency, time_to_goal_episodes)

def add_latencies(ql_grid_world, fw_grid_world):
    for gw_stats in (ql_grid_world, fw_grid_world):
        gw_stats["latency_all_episodes"] = list(latency_all_episodes(gw_stats["time_to_goal_episodes"]))

    for gw_stats in (ql_grid_world, fw_grid_world):
        for k, v in gw_stats.items():
            gw_stats[k] = np.asarray(v)
    return ql_grid_world, fw_grid_world


def plot_metrics(algs_grid_world, figname = "ql-fw-grid-world.pdf", labels = dict(),
                 figdir = "/tmp"):
    labels = [labels.get(k, k.upper()) for k in algs_grid_world.keys()]
    fig = plt.figure(figsize=(4*1.618, 3))
    fig.subplots_adjust(wspace = 0.45)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.boxplot([a["distineff_all_episodes"] for a in algs_grid_world.values()],
                labels = labels)
    ax1.set_ylabel("Distance Ineff. per episode")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.boxplot([a["latencies_all_episodes"] for a in algs_grid_world.values()],
                labels = labels)
    ax2.set_ylabel("Latency 1: >1")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.boxplot([a["rewards_all_episodes"] for a in algs_grid_world.values()],
                labels = labels)
    ax3.set_ylabel("Rewards per episode")
    #ax2.yaxis.tick_right()
    #ax2.yaxis.set_label_position("right")
    fig.savefig("/{figdir}/{figname}".format(figdir = figdir, figname = figname))
    return fig


def log_file_from_template(gitrev, confname):
    lfc = LogFileConf(project_name = PROJECT_NAME, confname = confname,
                      gitrev = gitrev)
    log_file_dir = lfc.log_file_dir
    latest_log_file = find_latest_file(log_file_dir)
    if not latest_log_file:
        raise RuntimeError("Unable to find file in {}".format(log_file_dir))
    return latest_log_file


def main(algos = "ql mb fw".split(),
         gitrev = "91a0c46",
         probs = ["4-room-grid-world", "4-room-windy-world"],
         figname = {"4-room-grid-world" : "metrics-grid-world.pdf",
                    "4-room-windy-world": "metrics-windy-world.pdf"},
         labels = dict(mb = "MBRL", fw = "FWRL", ql = "QL"),
         figdir = "/tmp"):

    for prob in probs:
        plot_metrics({k : load_data_from_log_file(
            str(log_file_from_template(gitrev, "-".join((k, prob)))))
                      for k in algos},
                     figname = figname[prob],
                     figdir = figdir
                     labels = labels)
    plt.show()

if __name__ == '__main__':
    main()

