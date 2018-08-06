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
*** DONE Implement latency 1: > 1
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


* Updated plan
  1. Write a LaTeX version of FWRL
     Propose an algorithm that competes with TDM, UVFA with HER. Probably need
     implementation and comparison with both. Run it on simplest to implement
     games.
  2. List and then chose a multi-goal platform that is easy to implement especially in grid world.
  3. Make an updated plan
  4. Need a working version of DQN or UVFA before FWRL can be tried
 
* List of multi-goal testing environments
  1. HER: Mujoco pushing, sliding and pick and place, Fetch
  2. TDM: 7-DOF reacher, pusher, half-cheetah, Ant, Sawyer robot
  3. UVFA: Lava world (two rooms 7x7), Ms Pacman
  4. HAC: Mujoco, Pendulum, UR5 reacher, cartpole swingup
 
* Problem
  1. HER is a bummer. What is the point of UVFA if you are going to remember all
     the past transitions.
  2. Advantages of HER: remembers the past
     Cons: uses too much of memory.

