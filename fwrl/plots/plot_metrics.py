# coding: utf-8
from fwrl.game.play import LogFileReader
from fwrl.game.logging import NPJSONEncDec, log_file_from_template
from fwrl.conf.default import PROJECT_NAME

import matplotlib.pyplot as plt
import numpy as np


def load_data_from_log_file(
        logfile,
        tags = "distineff_all_episodes rewards_all_episodes".split(),
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
                 figdir = "/tmp",
                 data_ylabels = dict(
                     distineff_all_episodes="Distance Ineff. per episode",
                     latency_all_episodes="Latency 1: >1",
                     rewards_all_episodes="Rewards per episode")):
    algos = algs_grid_world.keys()
    label_names = [labels.get(k, k.upper()) for k in algos]
    fig = plt.figure(figsize=(4*1.618, 3))
    fig.subplots_adjust(wspace = 0.45)
    data_keys = list(list(algs_grid_world.values())[0].keys())
    for di, dk in enumerate(data_keys):
        ax1 = fig.add_subplot(1, len(data_keys), di+1)
        ax1.boxplot([algs_grid_world[k][dk] for k in algos],
                    labels = label_names)
        ax1.set_ylabel(data_ylabels[dk])
    fig.savefig("/{figdir}/{figname}".format(figdir = figdir, figname = figname))
    return fig


def plot_rewards(algs_grid_world, figdir, figname, labels = dict()):
    fig = plt.figure(figsize=(4*1.618, 3))
    ax = fig.add_subplot(1,1,1)
    plts = list()
    for k, a in algs_grid_world.items():
        hndl, = ax.plot(a["rewards_all_episodes"], label=labels[k])
        plts.append(hndl)
    fig.legend(plts, [labels[k] for k in algs_grid_world.keys()])
    fig.savefig("/{figdir}/{figname}".format(figdir = figdir, figname = figname))
    return fig


def main(algos = "ql mb fw".split(),
         gitrev = "91a0c46",
         probs = ["4-room-grid-world", "4-room-windy-world"],
         figname = {"4-room-grid-world" : "metrics-grid-world.pdf",
                    "4-room-windy-world": "metrics-windy-world.pdf"},
         labels = dict(mb = "MBRL", fw = "FWRL", ql = "QL"),
         figdir = "/tmp"):

    for prob in probs:
        log_files = {k: log_file_from_template(PROJECT_NAME, gitrev, "-".join((k, prob)))
                    for k in algos}
        log_data = {k: load_data_from_log_file(str(log_file))
                    for k, log_file in log_files.items()}
        log_dir = log_files[algos[0]].parent
        print("Saving plots in {}".format(log_dir))
        plot_metrics(log_data,
                     figname = figname[prob],
                     figdir = log_dir,
                     labels = labels)
        plot_rewards(log_data,
                     figname = "rewards-" + figname[prob],
                     figdir = log_dir,
                     labels = labels)

    plt.show()

if __name__ == '__main__':
    main()

