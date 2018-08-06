import numpy as np
import landmarkmap as lmap
from landmarkmap import (Polygon, MultiPolygonMapBlock, StaticTraj
                         , RectangleMapBlock, RobotViewMapBlock
                         , PieceWiseLinearTraj, InteractiveTrajectory
                         , Trajectory, Rectangle, MapBlock, Landmarks)

def imap_conf_gen(choice_seed):
    choice = np.random.RandomState(choice_seed).choice(2, 1)[0]
    return imap_conf(choice, start_pos_seed=choice_seed, landmarks_seed=127)

def imap_conf(hint_choice, start_pos_seed=None, landmarks_seed=None):
    sample_per_block = 72
    # Imap
    #
    # (0, 120)                  (100, 120)
    # +-----------------------+
    # |          O5           |
    # +  .  +-----------+  .  + (100, 100)
    # |     | R  B1   G |     |
    # +     +---+   +---+     +
    # |         |   |         |
    # + O1   O2 |B2 | O3  O4  +
    # |         |   |         |
    # +     +---+   +---+     +
    # |     |   B3    Y |     |
    # +  .  +-----------+  .  + (100, 20)
    # |          O6           |
    # +-----------------------+
    # (0,0)                     (100, 0)
    #
    #
    FREE_REWARD = +0.0
    OBSTACLE_REWARD = -0.00
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    HINT_COLOR_CHOICE = [YELLOW, BLUE]
    hint_color = HINT_COLOR_CHOICE[hint_choice]
    red_reward = (+0.8 if hint_color == YELLOW else -0.8)
    green_reward = (-0.8 if hint_color == YELLOW else +0.8)
    DARK_GREEN = (0, 120, 0)
    BLACK = (0, 0, 0)
    defaultB = MapBlock(name='default'
                        , reward=OBSTACLE_REWARD
                        , color=(220, 220, 220))
    defaultFree = MapBlock(name='defaultFree'
                        , reward=FREE_REWARD
                        , is_obstacle = False
                        , color=(255, 255, 255))
    o1 = Rectangle([0, 0], [20, 120])
    o2 = Rectangle([20, 40], [40, 80])
    o3 = Rectangle([60, 40], [80, 80])
    o4 = Rectangle([80, 0], [100, 120])
    o5 = Rectangle([20, 100], [80, 120])
    o6 = Rectangle([20, 0], [80, 20])
    obstacleRegionBlock = MultiPolygonMapBlock(nsamples = 0
                                               , color = (220, 220, 220)
                                               , name = 'obstacle'
                                               # Check for collisions
                                               , check_contains = True
                                               # Considered in the region only 
                                               # if all points in the region
                                               , all_collision = False
                                               , is_obstacle = True
                                               , reward = OBSTACLE_REWARD
                                               , polygons = [o1, o2, o3, o4,
                                                             o5, o6]
                                               , trajectory = StaticTraj())
    b1 = Rectangle([20, 80], [80, 100])
    b2 = Rectangle([40, 40], [60, 80])
    b3 = Rectangle([20, 20], [80, 40])
    freeRegionBlock = MultiPolygonMapBlock(nsamples = 0
                                           , color=(255, 255, 255)
                                           , name = 'free'
                                           # Check for collisions
                                           , check_contains = True
                                           # Considered in the region only 
                                           # if all points in the region
                                           , all_collision = True
                                           , reward = FREE_REWARD
                                           , polygons = [b1, b2, b3]
                                           , trajectory = StaticTraj())
    landmarksBlock = MultiPolygonMapBlock(nsamples = sample_per_block
                                          , color_choice = [DARK_GREEN, BLACK]
                                          , seed = landmarks_seed
                                          , name = 'landmarks'
                                          , check_contains = False
                                          , polygons = [
                                              Polygon([
                                                  [20, 20]
                                                  , [80, 20]
                                                  , [80, 40]
                                                  , [60, 40]
                                                  , [60, 80]
                                                  , [80, 80]
                                                  , [80, 100]
                                                  , [20, 100]
                                                  , [20, 80]
                                                  , [40, 80]
                                                  , [40, 40]
                                                  , [20, 40]
                                              ]) ]
                                          , trajectory = StaticTraj())
    blockR = RectangleMapBlock(nsamples = 0
                               , name = 'BlockR'
                               , check_contains = True
                               , is_terminal = True
                               , all_collision = False
                               , reward = red_reward
                               , shape = (10, 20)
                               , color = (0, 0, 255)
                               , trajectory=Trajectory(initpos=(20, 80)))
    blockG = RectangleMapBlock(nsamples = 0
                               , name = 'BlockG'
                               , check_contains = True
                               , is_terminal = True
                               , all_collision = False
                               , reward = green_reward
                               , shape = (10, 20)
                               , color = (0, 255, 0)
                               , trajectory=Trajectory(initpos=(70,80)))
    blockHint = RectangleMapBlock(nsamples = 0
                               , reward = None
                               , name = 'BlockY'
                               , check_contains = False
                               , shape = (10, 20)
                               , color = hint_color
                               , trajectory=Trajectory(initpos=(70, 20)))
    robotsize = 5
    startArea = Polygon(vertices=[ [20 + robotsize, 20 + robotsize]
                                 , [70 - robotsize, 20 + robotsize]
                                 , [70 - robotsize, 40 - robotsize]
                                 , [20 + robotsize, 40 - robotsize]])
    random_state = np.random if start_pos_seed is None else \
            np.random.RandomState(start_pos_seed)
    robotBlock = RobotViewMapBlock(
        viewangle = 45*np.pi/180.
        , name = 'Robot'
        , check_contains = False
        , viewdist = 40
        , robotsize = robotsize
        , color = (0, 0, 255)
        , edgewidth = 4
        , _landmarks = Landmarks(np.array([[0., 0.]]).T)
        , trajectory = InteractiveTrajectory(
            initpos = startArea.sample(1, seed=start_pos_seed).flatten()
            , inittheta= random_state.rand()*2*np.pi))

    map_conf = [ robotBlock, landmarksBlock, blockR, blockG, blockHint,
                obstacleRegionBlock , defaultFree ]
    return map_conf, robotBlock, startArea

def imap():
    nframes = 1500
    map_conf, robblock, startArea = imap_conf()
    lmmap = lmap.map_from_conf(map_conf, nframes)
    lmv = lmap.LandmarksVisualizer([0, 0] , [100 , 120]
                                   , frame_period=80)
    return nframes, lmmap, lmv, robblock

if __name__ == '__main__':
    nframes, lmmap, lmv, robblock = imap()
    piecewise_trajectory = PieceWiseLinearTraj(
        np.array([[10, 110], [90, 10]])
            , 5 # frame break points
            , np.pi/25  # angular velocity
        )
    traj_iter = piecewise_trajectory.transforms(1000)

    # to get the landmarks with ids that are being seen by robot
    for r, theta, id, rs, ldmk, ldmk_obs, rew, color in lmap.get_robot_observations(
        lmmap, robblock, # Do not pass visualizer to
                         # disable visualization
                         lmv):
        T = traj_iter.next()
        robblock.current_T = T
        robblock.current_vel = piecewise_trajectory.current_vel
        print(rs[0], rs[1], rew, color)
