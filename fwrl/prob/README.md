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


``` python3
from fwrl.prob.windy_grid_world import AgentInGridWorld, AgentRenderer
agw = AgentInGridWorld.from_maze_name(maze_name = "4-room-grid-world", renderer = AgentRenderer.human)
agw.episode_reset(0)
{'goal_obs': array([3, 3])}
for _ in range(5):
    obs, rew, done, info = agw.step(agw.action_space.sample())
    agw.render()
```


