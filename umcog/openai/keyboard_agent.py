#!/usr/bin/env python
from __future__ import print_function

import sys, gym
import numpy as np

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#
import cog.openai.landmarkmap
import deepmind_lab_gym

deepmind_lab_gym.register_gym_env('DL_small_star_map_random_goal_01')
env = gym.make('DL-small-star-map-random-goal-01-v1' if len(sys.argv)<2 else sys.argv[1])

NACTIONS = env.action_space.size()
ROLLOUT_TIME = 1000
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(event):
    print('Press: ', event.key)
    key = event.key
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= NACTIONS: return
    human_agent_action = a

def key_release(event):
    key = event.key
    print('Release: ', event.key)
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= NACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.viewer.add_keypress_handle(key_press)
env.viewer.add_keyrelease_handle(key_release)

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    for t in range(ROLLOUT_TIME):
        # if not skip:
        #     #print("taking action {}".format(human_agent_action))
        #     a = human_agent_action
        #     skip = SKIP_CONTROL
        # else:
        #     skip -= 1
        a = np.random.randint(4)
        print("Taking action %d" % a)

        obser, r, done, info = env.step(a)
        print("Got reward %f" % r)
        env.render()
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            import time
            time.sleep(0.1)

print("NACTIONS={}".format(NACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    rollout(env)
