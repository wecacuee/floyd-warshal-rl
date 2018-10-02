# Problems package

1. Clone repository

``` bash
git clone ssh://git@opticnerve.eecs.umich.edu:2222/dhiman/floyd-warshall-rl.git
cd floyd-warshall-rl
```

1. Run installation script

``` bash
. setup.sh
```

### Windy grid world


1. Launch ipython

``` bash
ipython
```

1. The following code creates a grid world from a text file `fwrl/prob/data/<maze_name>`. You can also create a random grid world using  `AgentInGridWorld.from_random_maze(shape = (9, 9))` or from a file name `AgentInGridWorld.from_maze(maze = "data/4-room-grid-world")`


``` python
>>> from fwrl.prob.windy_grid_world import AgentInGridWorld, AgentRenderer
>>> agw = AgentInGridWorld.from_maze_name(maze_name = "4-room-grid-world", renderer = AgentRenderer.human)
>>> agw.episode_reset(0)
{'goal_obs': array([3, 3])}
>>> for _ in range(5):
    obs, rew, done, info = agw.step(agw.action_space.sample())
    agw.render()
```

### Scrolling Grid World

``` python
>>> from fwrl.prob.windy_grid_world import AgentRenderer
>>> from fwrl.prob.scrolling_grid_world import AgentInScrollingGW
>>> agw = AgentInScrollingGW.from_maze_name(maze_name = "4-room-grid-world", renderer=AgentRenderer.human)
>>> agw.episode_reset(0)
{'goal_obs': array([[0, 0, 0],
       [0, 6, 0],
       [0, 0, 0]], dtype=uint8)}
>>> for _ in range(5):
...     obs, rew, done, info = agw.step(agw.action_space.sample()); print(obs)
...     agw.render()
[[0 0 0]
 [0 0 0]
 [0 0 0]]
[[0 0 0]
 [0 0 0]
 [0 0 0]]
[[0 0 0]
 [0 0 0]
 [1 1 1]]
[[0 0 0]
 [1 0 0]
 [1 1 1]]
[[0 0 0]
 [1 0 0]
 [1 1 1]]
>>> import umcog.draw as draw
>>> draw.destroyAllWindows()
```


