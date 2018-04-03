import json
from game.play import NoOPObserver
import numpy as np
from functools import reduce

def mean(l):
    return sum(l) / len(l)


def latency_from_time_list(time_list, n=1):
    return mean(time_list[:n]) / mean(time_list[n:])


def compute_latency(times_to_goal_hit_all_episodes):
    # Need at least two hits to the goal for validity
    valid_times = filter(lambda t: len(t) >= 2, times_to_goal_hit_all_episodes)
    latencies_all_episode = list(map(latency_from_time_list, valid_times))
    return (mean(latencies_all_episode),
            min(latencies_all_episode), max(latencies_all_episode))

class ComputeMetricsFromLogReplay(NoOPObserver):
    def __init__(self, loggingobserver, metrics_observers, logfilereader):
        self.loggingobserver = loggingobserver
        self.metrics_observers = metrics_observers
        self.logfilereader = logfilereader
        super().__init__()

    def __getattr__(self, attr):
        return lambda *args, **kwargs : 0

    def on_play_end(self):
        self.loggingobserver.replay_observers_from_logs(
            self.metrics_observers, self.logfilereader)

class LatencyObserver(NoOPObserver):
    def __init__(self, prob):
        self.prob = prob
        self.times_to_goal_hit_all_episodes = []
        self.times_to_goal_hit              = None
        self.start_step                     = 0
        super().__init__()

    def on_new_episode(self, episode_n=None, **kwargs):
        if self.times_to_goal_hit:
            self.times_to_goal_hit_all_episodes.append(self.times_to_goal_hit)
        self.times_to_goal_hit = []
        self.start_step = 0

    def on_goal_hit(self, steps=None, **kwargs):
        time_to_hit = steps - self.start_step
        if time_to_hit:
            self.times_to_goal_hit.append(time_to_hit)
        self.start_step = steps + 1

    def on_new_step_with_pose_steps(self, obs=None, act=None, rew=None,
                                    pose=None, steps=None, episode_n=None,**kw):
        if self.prob and self.prob.hit_goal(): self.on_goal_hit(steps)

    def on_play_end(self):
        self.on_new_episode(episode_n=len(self.times_to_goal_hit)+1)
        print(self.times_to_goal_hit_all_episodes)
        mean_latency, min_l, max_l = compute_latency(
            self.times_to_goal_hit_all_episodes)
        print(f"latency : {mean_latency}; min latency {min_l}; max latency {max_l}")

class DistineffObs(NoOPObserver):
    def __init__(self, prob):
        self.prob                      = prob
        self.distineff_all_episodes    = []
        self.distineff_per_episode     = []
        self.pose_history              = []
        self.goal_was_hit_on_last_step = False
        self.goal_pose                 = None

    def on_new_episode(self, episode_n=None, goal_pose=None, **kwargs):
        if len(self.distineff_per_episode):
            self.distineff_all_episodes.append(
                self.distineff_per_episode)
        self.distineff_per_episode     = []
        self.pose_history              = []
        self.goal_was_hit_on_last_step = True
        self.goal_pose                 = goal_pose
        self.episode_n                 = episode_n

    def on_respawn(self, pose=None, steps=None):
        self.goal_was_hit_on_last_step = False
        if len(self.pose_history):
            diffs = np.diff(np.array(self.pose_history), axis=0)
            distance_traveled = np.sum(np.abs(diffs))
            shortest_distance = self.prob.shortest_path_length(
                tuple(self.pose_history[0].tolist()),
                tuple(self.goal_pose))
            distineff = distance_traveled / shortest_distance
            if distineff < 1.0:
                import pdb; pdb.set_trace()
            self.distineff_per_episode.append(distineff)

        self.pose_history = []

    def on_goal_hit(self, **kwargs):
        self.goal_was_hit_on_last_step = True

    def on_new_step_with_pose_steps(self, obs=None, act=None,
                                    rew=None, pose=None, steps=None,
                                    episode_n=None, **kwargs):
        if self.goal_was_hit_on_last_step and np.any(pose != self.goal_pose):
            self.on_respawn(pose, steps)

        self.pose_history.append(np.array(pose))

    def on_new_step(self, obs, act, rew):
        self.on_new_step_with_pose_steps(self, obs, act, rew, self.prob.pose)

    def on_play_end(self):
        self.on_new_episode(len(self.distineff_all_episodes)+1)
        print(self.distineff_all_episodes)
        alldistineff = sum(map(lambda l: l[1:],
                               self.distineff_all_episodes), [])
        mean_distineff = mean(alldistineff)
        min_distineff = min(alldistineff)
        max_distineff = max(alldistineff)
        print(f"""mean distineff = {mean_distineff} +-
                  ({mean_distineff - min_distineff},
                   {max_distineff - mean_distineff})""")
