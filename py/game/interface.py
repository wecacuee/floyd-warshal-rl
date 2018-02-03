from __future__ import absolute_import, division, print_function

class Space(object):
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
    def step(self, obs, reward):
        raise NotImplementedError()

    def action(self):
        raise NotImplementedError()

    def reset(self):
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
    for _ in range(nepisodes):
        play_episode(alg, prob)

def play_episode(alg, prob):
    while not (prob.done() or alg.done()):
        obs, rew = prob.observation()
        action = alg.step(obs, rew)
        prob.step(action)
    prob.reset()
    alg.reset()
        
