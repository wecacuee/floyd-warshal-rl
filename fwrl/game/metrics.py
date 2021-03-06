import json
import numpy as np
from functools import reduce
import logging

from umcog.confutils import extended_kwprop, KWProp as prop, xargs

def LOG():
    return logging.getLogger(__name__)

def info(msg, extra=dict(tag="human", data={})):
    LOG().info(msg, extra=extra)

def mean(l):
    return sum(l) / max(len(l), 1)


def latency_from_time_list(time_list, n=1):
    return mean(time_list[:n]) / mean(time_list[n:])

def text_quartile(quart):
    return """ |{min:<03.3}  ====  {quart_1:<03.3}  IIII  {median:<03.3}  IIII  {quart_3:<03.3}  ====  {max:>03.3}| """.format(**quart)

def quartiles(data):
    data = np.asarray(data)
    return dict(
        min = min(data, default=0),
        quart_1 = np.quantile(data, 0.25),
        median = np.quantile(data, 0.50),
        quart_3 = np.quantile(data, 0.75),
        max = max(data, default=0))

def compute_latency(times_to_goal_hit_all_episodes):
    # Need at least two hits to the goal for validity
    valid_times = filter(lambda t: len(t) >= 2, times_to_goal_hit_all_episodes)
    latencies_all_episodes = list(map(latency_from_time_list, valid_times))
    info("", extra = dict(tag = "latencies_all_episodes",
                          data = dict(latencies_all_episodes = latencies_all_episodes)))
    return quartiles(latencies_all_episodes)

class RewardObserver:
    def __init__(self, prob):
        self.rewards_all_episodes = []
        self.rewards = []

    def on_new_episode(self, episode_n=None, obs=None, **kwargs):
        if len(self.rewards):
            self.rewards_all_episodes.append(self.rewards)
        self.rewards = []

    def on_new_step_with_pose_steps(self, obs=None, act=None,
                                    rew=None, pose=None, steps=None,
                                    episode_n=None, **kwargs):
        self.rewards.append(rew)

    def on_play_end(self):
        self.on_new_episode(episode_n=len(self.rewards_all_episodes) + 1)
        total_rewards_all_episodes = list(map(sum, self.rewards_all_episodes))
        info(str(total_rewards_all_episodes),
             extra = dict(tag = "rewards_all_episodes",
                          data = dict(total_rewards_all_episodes = total_rewards_all_episodes)))
        info("Reward quartiles: "
             + text_quartile(quartiles(total_rewards_all_episodes)))


class LatencyObserver:
    def __init__(self, prob):
        self.prob = prob
        self.times_to_goal_hit_all_episodes = []
        self.times_to_goal_hit              = None
        self.obs_history                   = []
        super().__init__()

    def on_new_episode(self, episode_n=None, obs=None, **kwargs):
        if self.times_to_goal_hit:
            self.times_to_goal_hit_all_episodes.append(self.times_to_goal_hit)
        self.times_to_goal_hit = []
        self.obs_history = [obs]

    def on_goal_hit(self, steps=None, **kwargs):
        time_to_hit = len(self.obs_history)
        if time_to_hit:
            self.times_to_goal_hit.append(time_to_hit)
        self.obs_history = []

    def on_new_step_with_pose_steps(self, obs=None, act=None,
                                    rew=None, pose=None, steps=None,
                                    episode_n=None, **kwargs):
        self.obs_history.append(obs)

    def on_new_spawn(self, pose, episode_n, steps):
        # reset the start step but append nothing to time_to_hit
        #self.obs_history = []
        pass

    def on_play_end(self):
        self.on_new_episode(episode_n=len(self.times_to_goal_hit)+1)
        info(str(self.times_to_goal_hit_all_episodes))
        latency_quartiles = compute_latency(
            self.times_to_goal_hit_all_episodes)
        info("Latency quartiles: " + text_quartile(latency_quartiles))


class DistineffObs:
    def __init__(self, prob):
        self.prob                      = prob
        self.distineff_all_episodes    = []
        self.distineff_per_episode     = []
        self.pose_history              = []
        self.goal_was_hit_on_last_step = False
        self.goal_obs                 = None

    def on_new_episode(self, episode_n=None, goal_obs=None, obs=None, **kwargs):
        if len(self.distineff_per_episode):
            self.distineff_all_episodes.append(
                self.distineff_per_episode)
        self.distineff_per_episode     = []
        self.pose_history              = [obs]
        self.goal_was_hit_on_last_step = True
        self.goal_obs                 = goal_obs
        self.episode_n                 = episode_n

    def on_respawn(self):
        self.goal_was_hit_on_last_step = False
        diffs = np.diff(np.array(self.pose_history), axis=0)
        distance_traveled = np.sum(np.abs(diffs))
        shortest_distance, _ = self.prob.shortest_path(
            start = self.pose_history[0],
            end = self.goal_obs)
        assert shortest_distance != 0, "shortest distance cannot be zero"
        distineff = distance_traveled / shortest_distance
        assert distineff >= 1.0, """[Error]: Distineff should not be less than one. Find out why? Entering debug mode"""
        self.distineff_per_episode.append(distineff)

        self.pose_history = []

    def on_goal_hit(self, **kwargs):
        self.on_respawn()

    def on_new_spawn(self, pose, episode_n, steps):
        # reset the start step but append nothing to time_to_hit
        # Do not reset the pose history as a penalty to not being able to hit
        # goal while getting respawned.
        #self.pose_history = []
        pass

    def on_new_step_with_pose_steps(self, obs=None, act=None,
                                    rew=None, pose=None, steps=None,
                                    episode_n=None, **kwargs):
        self.pose_history.append(np.array(pose))

    def on_new_step(self, obs, act, rew, info):
        print("Deprecated")
        self.on_new_step_with_pose_steps(self, obs, act, rew, self.prob.pose)

    def on_play_end(self):
        self.on_new_episode(len(self.distineff_all_episodes) + 1)
        print(self.distineff_all_episodes)
        alldistineff = sum(map(lambda l: l[1:],
                               self.distineff_all_episodes), [])
        info("", extra=dict(
            tag = "distineff_all_episodes",
            data = dict(distineff_all_episodes = alldistineff)))
        info("Distance inefficiency quartiles: " +
             text_quartile(quartiles(alldistineff)))


class ComputeMetricsFromLogReplay:
    @extended_kwprop
    def __init__(self,
                 logging_observer = None,
                 log_file_reader = None,
                 metric_observer_keys = """latency_observer distineff_observer
                                           reward_observer """.split(),
                 metrics_observers = prop(lambda s: [
                     getattr(s, k) for k in s.metric_observer_keys]),
                 latency_observer = xargs(LatencyObserver, ["prob"]),
                 distineff_observer = xargs(DistineffObs, ["prob"]),
                 reward_observer    = xargs(RewardObserver, ["prob"])):
        self.logging_observer = logging_observer
        self.metrics_observers = metrics_observers
        self.log_file_reader = log_file_reader
        super().__init__()

    def __getattr__(self, attr):
        return lambda *args, **kwargs: 0

    def on_play_end(self):
        self.logging_observer.replay_observers_from_logs(
            self.metrics_observers, self.log_file_reader)


