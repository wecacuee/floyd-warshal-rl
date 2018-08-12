-*- mode: org; -*-
* Floyd-Warshall algorithm

** Getting started
*** For first time project setup

    ./onetime_setup.py
    source setup.sh

    To edit configurations, edit `onetime_setup.py`

*** Everytime you return the project run

    source setup.sh

** List of issues
*** Floyd-Warshall had unexplored area. How?
*** Log numpy arrays and then visualize.

** Log
*** DONE Mould greedy policy to explore unexplored areas.
  py/alg/floyd_warshall_grid.py
  CLOCK: [2018-02-08 Thu 17:22]--[2018-02-08 Thu 18:37] =>  1:15
  * Not done. Looks like we need to take a relook into the algorithm.
  The policy should be optimal to the extent that it visits unvisited
  states/edges at every point in the algorithm.

  * The exploration is happening right but the visualization is not.
    Fix the visualization.
    py/alg/floyd_warshall_grid.py
*** DONE Fix the visualization.
*** DONE Fix the respawning on hitting the goal.
*** DONE Randomize the goal on new episode
  py/prob/windy_grid_world.py
*** TODO Evalute in comparison to Q-learning with fresh explorer and finder, Q-learning with goal state.
*** Implement latency 1: > 1
  in terms of latency and distance inefficiency.
  Floyd rewards: 773.0 = sum([35.0, 33.0, 145.0, 11.0, 91.0, 123.0, 54.0, 237.0, 44.0])
  Latency: 10.53
  min latency : 0.23
  max latency : 58.5

  Q learning rewards: 750 = sum([159.0, 11.0, 80.0, 175.0, 151.0, 9.0, 85.0, 31.0, 49.0])
  Latency : 6.001897273610246;
  min latency 1.135483870967742;
  max latency 18.433734939759034





** Experimental setup

*** Problem setup
  py/game/play.py
*** Wind problem from the cse498
[hw1](./hw1.jpg)
*** Change random start point with random goal location
*** Interface as OpenAI gym interface
    + step
    + reset
*** Add a test mode to the gym interfaces so that the information is logged but not transmitted.

** Agent setup
*** Interface


** Updated plan
   1. Algorithms:
      - Simplified floyd warshall (11 Aug)
        + Make simplified FW work on 4-room grid world.
      - Make DQN work on your own terms (12 Aug)
      - Model-based: predict next state (13 Aug)
      - HER : Hindsight Experience Replay with non-end as goals as well as keyframes as goals.
      - (15 Aug)
   2. Environments:
      - Lava world : 10 Aug
      - Mujoco goal based continuous environments : Test DQN on Mujoco (13 Aug)
   3. Writeup:
      Write about simplified FW on 4-room grid world and lava world. (12 Aug)
 
** List of multi-goal testing environments
   1. HER: Mujoco pushing, sliding and pick and place, Fetch
   2. TDM: 7-DOF reacher, pusher, half-cheetah, Ant, Sawyer robot
   3. UVFA: Lava world (two rooms 7x7), Ms Pacman
   4. HAC: Mujoco, Pendulum, UR5 reacher, cartpole swingup
 
** Problem
   1. HER is a bummer. What is the point of UVFA if you are going to remember all
      the past transitions.
   2. Advantages of HER: remembers the past
      Cons: uses too much of memory.


** Latest results logs
/z/home/dhiman/mid/floyd_warshall_rl/qsub_86207.blindspot.out
/z/home/dhiman/mid/floyd_warshall_rl/201808_4493581_ql-4-room-grid-world/06-154235.log
QLearning
latency : 9.247249953793283; min latency 0.0037313432835820895; max latency 294.0
mean distineff = 8.484575787460404; +- (7.484575787460404, 40.0154242125396;)

Floyd-Warshall
/z/home/dhiman/mid/floyd_warshall_rl/201808_4493581_fw-4-room-grid-world/06-155055.log
latency : 17.604684042326305; min latency 0.13358778625954199; max latency 48.2
mean distineff = 1.1067439386686548; +- (0.10674393866865484, 7.607541775617059;)

Latest results:
/z/home/dhiman/mid/floyd_warshall_rl/201807_1e2ac0f_fw_grid_world_play
23 01:33:35 game.metrics    latency : 5.730895496953011; min latency 0.16086956521739132; max latency 10.686493184634449     human    2       {}
23 01:33:35 game.metrics    mean distineff = 1.0513843243829977; +- (0.05138432438299767, 2.9486156756170026;)  human   2    {}

/z/home/dhiman/mid/floyd_warshall_rl/201807_50ce457_ql_grid_world_play
30 10:09:24 game.metrics    latency : 1.8926652129883779; min latency 0.1939655172413793; max latency 3.803854875283447 human2
        {}
30 10:09:24 game.metrics    mean distineff = 1.6819494929376055; +- (0.6819494929376055, 11.068050507062395;)   human   2    {}

Aug 9:
dhiman@lens:.../mid/floyd_warshall_rl$ tail -2 201808_7a324cf_ql-4-room-grid-world/07-131330.log 
07 13:22:53 fwrl.game.metrics latency : 9.247249953793283; min latency 0.0037313432835820895; max latency 294.0 human   2    {}
07 13:22:53 fwrl.game.metrics mean distineff = 8.484575787460404; +- (7.484575787460404, 40.0154242125396;)     human   2    {}
dhiman@lens:.../mid/floyd_warshall_rl$ tail -2 201808_7a324cf_fw-4-room-grid-world/08-021509.log 
08 03:33:50 fwrl.game.metrics latency : 17.604684042326305; min latency 0.13358778625954199; max latency 48.2   human   2    {}
08 03:33:50 fwrl.game.metrics mean distineff = 1.1067439386686548; +- (0.10674393866865484, 7.607541775617059;) human   2    {}
dhiman@lens:.

Aug 12:
dhiman@lens:~/wrk/floyd-warshall-rl$ tail -2 /z/home/dhiman/mid/floyd_warshall_rl/201808_b77f1bb_ql-4-room-grid-world/12-140554.log
12 14:21:32 fwrl.game.metrics latency : 5.422566651204752; min latency 0.05363984674329502; max latency 49.0    human   2    {}
12 14:21:32 fwrl.game.metrics mean distineff = 9.641941793146612; +- (8.641941793146612, 67.35805820685339;)    human   2    {}
dhiman@lens:~/wrk/floyd-warshall-rl$ tail -2 /z/home/dhiman/mid/floyd_warshall_rl/201808_b77f1bb_fw-4-room-grid-world/12-142132.log
12 14:35:41 fwrl.game.metrics latency : 39.247887970615245; min latency 0.5; max latency 139.0  human   2       {}
12 14:35:41 fwrl.game.metrics mean distineff = 1.1000429830217062; +- (0.10004298302170622, 2.3444014614227386;)        human2
        {}



# Observations
1. Q(s,a) = F(s, a, \infty) is an exploration specific build up. Encourages
   exploration of unexplored areas, might not be useful for large state spaces.
   But still gives an estimate of the distance from the nearest exploration
   border.
   This is like distance transform from the exploration border.
2. Discount factor makes reward shift specific.
3. F(i, l, j) = max_a Q(s, a) - Q(s, a)
