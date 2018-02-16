from __future__ import absolute_import, division, print_function
import logging
import json
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Space(object):
    def values():
        raise NotImplementedError()

    @property
    def size():
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def contains(self, x):
        raise NotImplementedError()

    def to_jsonable(self, sample_n):
        return sample_n

    def from_jsonable(self, sample_n):
        return sample_n
        
class Problem(object):
    action_space = Space()
    observation_space = Space()
    reward_range = Space()
    def step(self, action):
        raise NotImplementedError()

    def observation(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def episode_reset(self):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        raise NotImplementedError()

    def unwrapped(self):
        raise NotImplementedError()

    def done(self):
        raise NotImplementedError()
    

class Alg(object):
    action_space = Space()
    observation_space = Space()
    reward_range = Space()
    def update(self, obs, act, reward):
        raise NotImplementedError()

    def policy(self, obs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def episode_reset(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        raise NotImplementedError()

    def unwrapped(self):
        raise NotImplementedError()

    def done(self):
        raise NotImplementedError()


class NoOPObserver(object):
    def __init__(self):
        self.prob = None
        self.alg = None
        self.parent = None

    def set_prob(self, prob):
        self.prob = prob

    def set_alg(self, alg):
        self.alg = alg

    def __getattr__(self, attr):
        return lambda *args, **kwargs : 0


class MultiObserver(object):
    """
    Class to independently observe the experiments
    """
    def __init__(self, observers):
        self.observers = observers
        for o in self.observers.values():
            o.parent = self

    def set_prob(self, prob):
        for o in self.observers.values():
            o.set_prob(prob)

    def set_alg(self, alg):
        for o in self.observers.values():
            o.set_alg(alg)

    def on_new_episode(self, episode_n):
        for o in self.observers.values():
            o.on_new_episode(episode_n)

    def on_new_step(self, obs, rew, action):
        for o in self.observers.values():
            o.on_new_step(obs, rew, action)

    def on_play_end(self):
        for o in self.observers.values():
            o.on_play_end()
        

class LoggingObserver(NoOPObserver):
    def __init__(self, logfilepath, log_interval):
        self.logfilepath = logfilepath
        self.log_interval = log_interval
        self.human_tag = "INFO"
        self.goal_hit_tag = f"{self.__class__.__name__}:goal_hit"
        self.new_step_tag = f"{self.__class__.__name__}:new_step"
        self.new_episode_tag = f"{self.__class__.__name__}:new_episode"
        self.play_end_tag = f"{self.__class__.__name__}:play_end"
        self.sep = "\t"
        super().__init__()
        print(f"LatencyLogger to file path: {self.logfilepath}")

    def info(self, tag, message):
        with open(self.logfilepath, "a") as log:
            log.write(self.sep.join((tag, str(len(message)) , message)))
            log.write("\n")

    def on_new_episode(self, episode_n):
        self.last_episode_n = episode_n
        json_string = json.dumps(
            dict(episode_n=episode_n, goal_pose=self.prob.goal_pose.tolist()))
        self.info(self.human_tag, 
            f" +++++++++++++++++ New episode: {episode_n} +++++++++++++")
        self.info(self.new_episode_tag, json_string)

    def on_goal_hit(self, current_step):
        json_string = json.dumps(
            dict(episode_n=self.last_episode_n, steps=current_step))
        self.info(self.goal_hit_tag, json_string)

    def on_new_step_with_pose_steps(self, obs,rew, act, pose, steps):
        if self.prob.hit_goal(): self.on_goal_hit(self.prob.steps)
        if steps % self.log_interval == 0:
            self.info(self.new_step_tag,
                      json.dumps(dict(episode_n = int(self.last_episode_n),
                                      steps     = int(steps),
                                      obs       = obs.tolist(),
                                      rew       = float(rew),
                                      act       = int(act),
                                      pose      = pose.tolist())))

    def on_new_step(self, obs, rew, act):
        self.on_new_step_with_pose_steps(
            obs, rew, act, self.prob.pose, self.prob.steps)

    def on_play_end(self):
        self.info(self.play_end_tag, json.dumps({}))
        self.info(self.human_tag, "End of play")

    def tag_event_map(self):
        return dict([(self.new_step_tag, "on_new_step_with_pose_steps"),
                     (self.goal_hit_tag, "on_goal_hit"),
                     (self.new_episode_tag, "on_new_episode"),
                     (self.play_end_tag, "on_play_end")])

    def replay_observers_from_logs(self, observers, logfilepath):
        tag_event_map = self.tag_event_map()
        with open(logfilepath, "r") as log:
            for logline in log:
                tag, length, json_string = logline.strip().split("\t")
                assert int(length) == len(json_string), \
                    f"Corrupted log line {length} <=> {len(json_string)}"
                if tag in tag_event_map:
                    for obs in observers:
                        if hasattr(obs, tag_event_map[tag]):
                            getattr(obs, tag_event_map[tag])(
                                **json.loads(json_string))

# Sample refrees
def play(alg, prob, observer, nepisodes):
    observer.set_prob(prob)
    observer.set_alg(alg)
    for n in range(nepisodes):
        observer.on_new_episode(n)
        play_episode(alg, prob, observer, n)

    observer.on_play_end()
    return observer

def play_episode(alg, prob, observer, episode_n):
    obs = prob.observation()
    rew = prob.reward()
    action = prob.action_space.sample()
    while not (prob.done() or alg.done()):
        observer.on_new_step(obs=obs, rew=rew, action=action)
        alg.update(obs, action, rew)
        action = alg.egreedy(alg.policy(obs))
        obs, rew = prob.step(action)
        #prob.render(None, 100, wait_time=1)

    prob.episode_reset()
    alg.episode_reset()
        
def multiplay(trials):
    for conf in trials:
        print(f"Running trial {conf.__class__.__name__}")
        play(conf.alg, conf.prob, conf.observer, conf.nepisodes)
