from __future__ import absolute_import, division, print_function
import logging
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
        

# Sample refrees
def play(alg, prob, nepisodes):
    all_rewards = []
    for n in range(nepisodes):
        logger.info(f" +++++++++++++++++ New episodes {n} +++++++++++++")
        total_rew = play_episode(alg, prob, n)
        all_rewards.append(total_rew)
    logger.info(f"total rewards : {all_rewards}")

def play_episode(alg, prob, episode_n):
    obs = prob.observation()
    rew = prob.reward()
    action = prob.action_space.sample()
    total_rew = rew
    while not (prob.done() or alg.done()):
        if prob.steps % 5 == 0:
            print(f"episode = {episode_n}, step = {prob.steps}, obs = {obs}; action = {action}; rew = {rew}; total_episode_reward = {total_rew}")
        alg.update(obs, action, rew)
        action = alg.egreedy(alg.policy(obs))
        obs, rew = prob.step(action)
        prob.render(None, 100, wait_time=1)
        total_rew += rew
    prob.episode_reset()
    alg.episode_reset()
    return total_rew
        
def play_from_conf(conf):
    play(conf.alg, conf.prob, conf.nepisodes)

if __name__ == '__main__':
    import sys
    from conf.default import Conf
    conf = Conf.parse_all_args(sys.argv[1:])
    play_from_conf(conf)
