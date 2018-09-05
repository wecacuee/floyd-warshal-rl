from functools import partial
from .play import Renderer


def call_alg_update(alg, *a):
    return alg.update(*a)


def call_alg_policy(alg, obs, act, rew, done, info):
    return alg.policy(obs)


def play_episode(alg, prob, observer, episode_n,
                 renderer = Renderer.noop,
                 alg_per_step_cb = call_alg_update,
                 # 5 million steps in one episode is a lot
                 max_steps = 5000000):
    prob.episode_reset(episode_n)
    alg.episode_reset(episode_n)
    # New goal location
    alg.set_goal_obs(prob.goal_obs)
    obs = prob.observation()
    observer.on_new_episode(episode_n=episode_n, obs=obs,
                            goal_obs=prob.goal_obs)
    act = alg.update(obs, None, None, None, dict())
    reward = 0
    for step_n in range(max_steps):
        obs, rew, done, info = prob.step(act)
        reward += rew
        observer.on_new_step(obs=obs, rew=rew, action=act, info=info)

        act = alg_per_step_cb(alg, obs, act, rew, done, info)

        renderer(prob)
        if done:
            break

    # Record end of episode
    observer.on_episode_end(episode_n=episode_n)
    return reward

train_episode = play_episode
test_episode = partial(play_episode, alg_per_step_cb = call_alg_policy)

