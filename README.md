-*- mode: org; -*-
# Floyd-Warshall algorithm

## Log
* DONE Mould greedy policy to explore unexplored areas.
  py/alg/floyd_warshall_grid.py
  CLOCK: [2018-02-08 Thu 17:22]--[2018-02-08 Thu 18:37] =>  1:15
  * Not done. Looks like we need to take a relook into the algorithm.
  The policy should be optimal to the extent that it visits unvisited
  states/edges at every point in the algorithm.

  * The exploration is happening right but the visualization is not.
    Fix the visualization.
    py/alg/floyd_warshall_grid.py
* DONE Fix the visualization.
* DONE Fix the respawning on hitting the goal.
* DONE Randomize the goal on new episode
  py/prob/windy_grid_world.py
* TODO Evalute in comparison to Q-learning with fresh explorer and finder, Q-learning with goal state.
  in terms of latency and distance inefficiency.

## Experimental setup

### Problem setup
* Wind problem from the cse498
[hw1](./hw1.jpg)
* Change random start point with random goal location
* Interface as OpenAI gym interface
    + step
    + reset
* Add a test mode to the gym interfaces so that the information is logged but not transmitted.

### Agent setup
* Interface

