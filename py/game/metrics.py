import json
from .play import NoOPObserver

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

class LatencyLogger(NoOPObserver):
    def __init__(self, logfilepath):
        self.logfilepath = logfilepath
        self.goal_hit_tag = f"{self.__class__.__name__}:goal_hit"
        self.new_episode_tag = f"{self.__class__.__name__}:new_episode"
        self.play_end_tag = f"{self.__class__.__name__}:play_end"
        self.last_episode_n = None
        self.sep = "\t"

    def info(self, tag, message):
        with open(self.logfilepath, "a") as log:
            log.write(self.sep.join((tag, str(len(message)) , message)))

    def on_new_episode(self, episode_n):
        self.last_episode_n = episode_n
        json_string = json.dumps(dict(episode_n=episode_n))
        self.info(self.new_episode_tag, json_string)

    def on_goal_hit(self, current_step):
        json_string = json.dumps(
            dict(episode_n=self.last_episode_n, steps=current_step))
        self.info(self.goal_hit_tag, json_string)

    def on_new_step(self, obs, act, rew):
        if self.prob.hit_goal(): self.on_goal_hit(current_step)

    def on_play_end(self):
        self.info(self.play_end_tag, json.dumps({}))

    def run_observer_from_logfile(obs):
        with open(self.logfilepath, "r") as log:
            for logline in log:
                tag, length, json_string = logline.strip().split("\t")
                if tag == self.goal_hit_tag:
                    latency_obs.on_goal_hit(
                        json.loads(json_string)["steps"])
                elif tag == self.new_episode_tag:
                    latency_obs.on_new_episode(
                        json.loads(json_string)["episode_n"])
                elif tag == self.play_end_tag:
                    break


class LatencyObserver(NoOPObserver):
    def __init__(self):
        self.times_to_goal_hit_all_episodes = []
        self.times_to_goal_hit = None
        self.start_step = 0
        super().__init__()

    def on_new_episode(self, episode_n):
        if self.times_to_goal_hit:
            self.times_to_goal_hit_all_episodes.append(self.times_to_goal_hit)
        self.times_to_goal_hit = []
        self.start_step = 0

    def on_goal_hit(self, current_step):
        time_to_hit = current_step - self.start_step
        if time_to_hit:
            self.times_to_goal_hit.append(time_to_hit)
        self.start_step = current_step + 1

    def on_new_step(self, obs, act, rew):
        if self.prob.hit_goal(): self.on_goal_hit(self.prob.steps)

    def on_play_end(self):
        self.on_new_episode(len(self.times_to_goal_hit)+1)
        mean_latency, min_l, max_l = compute_latency(
            self.times_to_goal_hit_all_episodes)
        print(f"latency : {mean_latency}; min latency {min_l}; max latency {max_l}")

