"""
Landmark map
"""
import copy
import itertools
import operator as op
from collections import namedtuple

import numpy as np
import numpy.random as nprnd
from numpy.linalg import norm as vnorm

from cog.geom.polygons import convex_polygon_contains

from cog.mpl2cv2 import MPLAsCV, is_cv2_matplotlib
import matplotlib.cm as cmap

cv2 = MPLAsCV()
white_img = cv2.white_img
#white_img = lambda x : np.ones(x[0], x[1], 3)*255

DEFAULT_REWARD = 0.2 # high reward for free areas
REWARD_RANGE = (-1., 1.) # for coloring

class Landmarks(object):
    nD = 2
    def __init__(self, locations=np.zeros((nD, 0)),
                 colors=None):
        """
        locations : D x N array
        colors : 3[RGB] x N array with RGB in (0, 255)
        """
        if colors is None:
            colors = np.zeros((3, locations.shape[1]))
        assert isinstance(locations, np.ndarray)
        assert isinstance(colors, np.ndarray)
        assert locations.shape[1] == colors.shape[1], '{0} <=> {1}'.format(
            locations.shape, color.shape)
        self.locations = locations
        self.colors = colors

    def transformed(self, T):
        clone = copy.copy(self)
        clone.locations = applyT(T, self.locations)
        return clone

    @classmethod
    def merge(cls, ldmk_iter):
        return cls(locations=np.hstack((ldmk.locations for ldmk in ldmk_iter))
                   , colors = np.hstack((ldmk.colors for ldmk in ldmk_iter)))

class Shape(object):
    def contains(self, pos):
        raise NotImplementedError()

    def transformed(self, T):
        raise NotImplementedError()

    def draw(self, frame, **kwargs):
        return frame

# Each map is made up of blocks of customizable shapes and sizes
# The blocks can be dynamic as defined by the trajectory.
class MapBlock(object):
    def __init__(self, name=None, reward=None, color=None, edgewidth=None
                , is_obstacle = True
                , check_contains = True
                , is_terminal = False):
        self.name = name
        self.reward = reward
        self.color = color
        self.edgewidth = edgewidth
        self.shape = None
        self.check_contains = check_contains
        self.is_obstacle = is_obstacle
        self.is_terminal = is_terminal

    def landmarks(self):
        return Landmarks()

    def trajectory_transforms(self, nframes):
        return (np.eye(3) for i in xrange(nframes))

    def transformed(self, T):
        return self

    def contains(self, pos):
        return True

    def draw(self, frame, **kwargs):
        if is_cv2_matplotlib(cv2):
            frame.set_axis_bgcolor((0.5, .5, .5))
        else:
            frame = frame * 128
        return frame

class LandmarkMapBlock(MapBlock):
    def __init__(self, ldmks):
        self.ldmks = ldmks
        self.check_contains = False
        self.is_obstacle = True

    def landmarks(self):
        return self.ldmks

    def transformed(self):
        return LandmarkMapBlock(applyT(T, self.ldmks.copy().T).T)

    def contains(self, pos):
        return False

class Polygon(Shape):
    def __init__(self, vertices, show_edges=None, scale=1):
        """
        An ndarray of n x 2 [(x1, y1), ..., (xn, yn)]
        """
        self.vertices = scale * np.asarray(vertices)
        assert self.vertices.shape[1] == 2, \
                'Expected 2, found %d' % self.vertices.shape[1]
        if show_edges is None:
            show_edges = [True]* self.vertices.shape[0]
        self.show_edges = show_edges

    def edges(self):
        return zip(list(self.vertices),
                   list(self.vertices[1:]) + list(self.vertices[0:1]))

    def visible_edges(self):
        return [e for i, e in enumerate(self.edges())
                if self.show_edges[i]]

    def transformed(self, T):
        return Polygon(applyT(T, np.asarray(self.vertices).T).T)

    def sample(self, n, seed=None):
        random_state = np.random if seed is None else \
                np.random.RandomState(seed)
        # Sample inside the bounding box
        bbox_left_bottom = np.min(self.vertices, axis=0)
        bbox_spread = np.max(self.vertices, axis=0) - bbox_left_bottom
        # We have to loop till we find enough points inside the polygon
        filtered = np.zeros((2, 0))
        while filtered.shape[1] < n:
            # Sample 10 times the points required
            pts = (random_state.rand(10*n, 2) * bbox_spread + bbox_left_bottom).T
            # Filter the points inside the polygon
            inside = self.contains(pts)
            filtered = pts[:, inside]
        return filtered[:, :n]

    def contains(self, pos):
        # assume convex polygons
        return convex_polygon_contains(self.vertices, pos)

    def draw(self, img, facecolor=(0,0,0), scale=1, edgewidth=None):
        if edgewidth is not None and edgewidth > 0:
            cv2.polylines(img, self.vertices*scale, True
                          , color=(0, 0, 0), thickness=edgewidth)

        cv2.fillConvexPoly(img , self.vertices*scale ,
                           color=facecolor)
        return img

class Rectangle(Polygon):
    def __init__(self, left_bottom, right_top, show_edges=None, scale=1):
        """ show_edges : clockwise from bottom """
        vertices = [[left_bottom[0], left_bottom[1]],
                    [right_top[0],   left_bottom[1]],
                    [right_top[0],   right_top[1]],
                    [left_bottom[0], right_top[1]]]
        Polygon.__init__(self, vertices, scale=scale, show_edges=show_edges)

def vec_linspace(start, end, num):
    """
    Returns evenly spaced vectors over a specified interval

    >>> s = np.array([1, 2])
    >>> e = np.array([11, 12])
    >>> vec_linspace(s, e, 11)
    array([[ 1,  2],
           [ 2,  3],
           [ 3,  4],
           [ 4,  5],
           [ 5,  6],
           [ 6,  7],
           [ 7,  8],
           [ 8,  9],
           [ 9, 10],
           [10, 11],
           [11, 12]])

    # Corner cases
    >>> vec_linspace(s, e, 1)
    array([[11, 12]])
    >>> vec_linspace(np.array([1, 2.]), np.array([11, 12]), 0)
    array([], shape=(0, 2), dtype=float64)
    """
    assert num >= 0, "Need non-negative integer"
    if num == 0:
        return np.zeros(([0]+list(start.shape)), dtype=start.dtype)
    start = np.asarray(start)
    end = np.asarray(end)
    step = (end - start) / (num-1) if num > 1 else (end - start)
    y = np.arange(num).reshape(-1, 1)*step + start
    y[-1, :] = end
    return y

def landmarks_on_edges(edges, n, color_choice, seed=None):
    random_state = np.random if seed is None else \
            np.random.RandomState(seed)
    length_per_side = [np.linalg.norm(np.asarray(e[0]) - np.asarray(e[1]))
                       for e in edges ]
    frac_per_side = length_per_side / sum(length_per_side)
    ldmk_per_side = np.maximum(np.floor(n * np.asarray(frac_per_side)), 1)
    remaining = np.int(n - sum(ldmk_per_side))
    ldmk_per_side[:remaining] += 1 # allocate one more landmark to first sides
    ldmks_NxD = np.vstack(
        map(lambda e_n: vec_linspace(e_n[0][0] , e_n[0][1] , e_n[1])
            , zip(edges, ldmk_per_side))
    )
    ldmks = ldmks_NxD.T
    ldmk_colors_Nx3 = np.asarray(color_choice)[
        random_state.choice(len(color_choice), n), :]
    ldmk_colors = ldmk_colors_Nx3.T
    return Landmarks(ldmks, ldmk_colors)

def landmarks_on_polygon(polygon, n):
    """
    Let polygon be represented 
    """
    if n == 0:
        return Landmarks()

    visible_edges = polygon.visible_edges()
    return landmarks_on_edges(visible_edges, n)

def landmarks_on_multipolygon(polygons, n, color_choice, seed=None):
    if n == 0:
        return Landmarks()
    return landmarks_on_edges(sum((p.visible_edges() for p in polygons), [])
                              , n , color_choice, seed = seed)

class MultiShape(Shape):
    def __init__(self, shapes):
        self.shapes = shapes

    def contains(self, pos):
        """ pos is DxN array
        """
        assert pos.ndim == 2, "Need 2D array"
        return reduce(lambda acc, bool: acc | bool
                      , (s.contains(pos) for s in self.shapes)
                      , np.zeros(pos.shape[1], dtype=np.bool))

    def sample(self, n):
        n_per_shape = np.ones(len(self.shapes)) * np.floor(n/len(self.shapes))
        remaining = np.int(n - sum(n_per_shape))
        n_per_shape[:remaining] += 1
        return np.hstack((s.sample(nps) for s, nps in zip(self.shapes,
                                                          n_per_shape)))

    def transformed(self, T):
        clone = copy.copy(self)
        clone.shapes = [s.transformed(T) for s in self.shapes]

    def draw(self, frame, **kwargs):
        for s in self.shapes:
            frame = s.draw(frame, **kwargs)
        return frame

class MultiPolygonMapBlock(MapBlock):
    _attrs = """name nsamples polygons trajectory reward color
                edgewidth _landmarks check_contains all_collision
                is_obstacle is_terminal color_choice seed""".split()
    def __init__(self, **kwargs):
        color_choice = kwargs.get("color_choice", [(0, 0, 0)])
        seed = kwargs.get("seed", None)
        defaults = dict(_landmarks=
                        lambda :
                        landmarks_on_multipolygon(kwargs["polygons"]
                                                  , kwargs["nsamples"]
                                                  , color_choice
                                                  , seed = seed)
                        , check_contains = lambda : True
                        , all_collision = lambda : False
                        , is_obstacle = lambda : False
                        , is_terminal = lambda : False)
        lazynone = lambda : None
        for a in self._attrs:
            setattr(self, a
                    , kwargs.get(a, defaults.get(a, lazynone)()))

    def landmarks(self):
        return self._landmarks

    def sample(self, n):
        return self.shape.sample(n)

    @property
    def shape(self):
        return MultiShape(self.polygons)

    def trajectory_transforms(self, nframes):
        return self.trajectory.transforms(nframes)

    def transformed(self, T):
        clone = copy.copy(self)
        for a in self._attrs:
            setattr(clone, a, getattr(self, a))
        clone.polygons = [p.transformed(T) for p in self.polygons]
        clone._landmarks = self._landmarks.transformed(T)
        return clone

    def contains(self, pos):
        assert self.check_contains, 'Not meant to call contains for %s' % self.name
        if self.all_collision:
            return np.all(self.shape.contains(pos))
        else:
            return np.any(self.shape.contains(pos))

    def draw(self, img, scale=1, radius=1, maybecolor=lambda c,r: c, **kwargs):
        if self.reward is not None or self.color is not None:
            facecolor = maybecolor(self.color, self.reward)
            img = self.shape.draw(img, scale=scale, facecolor=facecolor
                                  , edgewidth=self.edgewidth, **kwargs)
        return img

class RectangleMapBlock(MultiPolygonMapBlock):
    def __init__(self, nsamples, shape, trajectory, **kwargs):
        MultiPolygonMapBlock.__init__(self, nsamples=nsamples
                                 , polygons = [Polygon(
                                     vertices_rect(shape).T)]
                                 , trajectory=trajectory
                                 , **kwargs)

def vertices_rect(shape):
    """Generate n landmarks within rectangle"""
    nD = 2
    vertices = [[0,0],
                [shape[0], 0],
                [shape[0], shape[1]],
                [0, shape[1]],
               ]
    return np.array(vertices).T

def applyT(T, points):
    nD = points.shape[0]
    nD_T = T.shape[1] - 1
    assert nD == nD_T, "Dimension mismatch %d <> %d " % (nD, nD_T)
    return T[:nD, :nD].dot(points) + T[:nD, nD:nD+1]

class RigidBody2D(object):
    """ A set of landmarks """
    _id_counter = 0;
    def __init__(self, landmarks, map_block):
        assert landmarks.locations.shape[0] == 2, \
                "Only 2D supported, found %d" % landmarks.locations.shape[0]
        self._landmarks = landmarks
        self._map_block = map_block

    def get_landmarks(self, T):
        return self._landmarks.transformed(T)

    def get_map_block(self, T):
        return self._map_block.transformed(T)

class RigidMotions(object):
    """ A pair of rigid body and it's transform trajectory"""
    def __init__(self, rb, trajectory):
        self.rb = rb
        self.trajectory = trajectory

    def get_landmarks_traj(self):
        for T in self.trajectory:
            yield self.rb.get_landmarks(T)

    def get_map_block_traj(self):
        for T in self.trajectory:
            yield self.rb.get_map_block(T)

def static_trajectory(Tinit, n):
    """ Static trajectory for n frames """
    for i in xrange(n):
        yield Tinit

def prismatic_trajectory(*args):
    """ Prismatic trajectory for n frames """
    return dyn_trajectory(*args)

def dyn_trajectory(Tinit, delT, n):
    """ Revolute trajectory for n frames """
    Titer = Tinit
    # note that copy is necessary, otherwise delPos is just a view on delT
    # which causes it to change when delT changes
    delPos = delT[:-1, -1:].copy() 
    for i in xrange(n):
        tmp = Titer
        positer = Titer[:-1, -1:]
        # Center of rotation is around around positer (TODO: accept it as
        # paramter)
        delT[:-1, -1:] = delPos -delT[:-1, :-1].dot(positer) + positer 
        Titer = delT.dot(Titer)
        yield tmp

class InstantMap(object):
    """
    A static map at any instant t
    """
    default_region = MapBlock(reward=DEFAULT_REWARD,
                              color=(255, 255, 255))
    def __init__(self, regions, landmarks):
        self.regions = regions
        self.landmarks = landmarks

    def get_region_by_pos(self, pos):
        """
        pos is DxN numpy array
        """
        possible_rewards = []
        for i, rr in enumerate(
            (rr for rr in self.regions if rr.check_contains)):
            if rr.contains(pos):
                # Use the first region's reward that matches
                return rr

        # Robot is in the default area
        return self.default_region

class LandmarkMap(object):
    """ A dynamic map of landmarks """
    def __init__(self, rigid_motions):
        self._rigid_motions = rigid_motions

    def get_map_traj(self):
        return (InstantMap(rr_t, ldmk_t)
                for rr_t, ldmk_t in itertools.izip(self.get_map_blocks()
                                                   , self.get_landmarks()))

    def get_map_blocks(self):
        map_block_trajs = map(lambda rm: rm.get_map_block_traj()
                                  , self._rigid_motions)
        return itertools.izip(*map_block_trajs)

    def get_landmarks(self):
        ldmk_gen = map(lambda rm: rm.get_landmarks_traj(), self._rigid_motions)
        return itertools.imap(lambda ldmk: Landmarks.merge(ldmk)
                              , itertools.izip(*ldmk_gen))

class RobotView(object):
    """ Describes the cone in view by robot """
    def __init__(self, pos, dir, maxangle, maxdist):
        self._pos = pos.reshape((-1,1))
        self._dir = dir.reshape((-1,1)) / vnorm(dir)
        self._maxangle = maxangle
        self._maxdist = maxdist

    def in_view(self, points):
        # Need to change this to view in 3D ??
        """ Returns true for points that are within view """
        pos = self._pos
        dir = self._dir
        cpoints = points - pos
        dists = np.sqrt(np.sum(cpoints**2, axis=0))

        # Taking dot product for cosine angle
        cosangles = dir.T.dot(cpoints) / np.where(dists == 0, 1, dists)
        cosangles = cosangles[0, :]

        # The cos angle is negative only when landmark lies behind the robot's heading direction. 
        # Max distance landmarks can be retained. There is no argument or counter-argument yet for in/ex-clusion
        in_view_pts = (cosangles > np.cos(self._maxangle)
                      ) & (dists <= self._maxdist)

        if len(in_view_pts.shape) > 1:
            import pdb;pdb.set_trace()
        return in_view_pts

class RobotViewMapBlock(MultiPolygonMapBlock):
    _parent_class = MultiPolygonMapBlock
    _attrs_this = """viewangle viewdist robotsize""".split()
    _attrs = _parent_class._attrs + _attrs_this
    def __init__(self, **kwargs):
        nD = 2
        viewangle = kwargs["viewangle"]
        robotsize = kwargs["robotsize"]
        hva = viewangle / 2.
        kwargs.setdefault("nsamples", 0)
        kwargs.setdefault("polygons",
                          [Polygon(robotsize *
                                  np.array([(0, 0)
                                            , (np.cos(hva), np.sin(hva))
                                            , (np.cos(hva), np.sin(-hva)) ]))])
        kwargs.setdefault("reward", None)
        kwargs.setdefault("check_contains", False)
        kwargs.setdefault("is_obstacle", False)
        self._parent_class.__init__(
            self, **dict([(k, kwargs[k])
                      for k in self._parent_class._attrs
                         if k in kwargs]))
        for a in self._attrs_this:
            setattr(self, a, kwargs.get(a, None))

    def current_robot_view(self):
        T = self.current_T
        pos = T[:-1, -1]
        dir = T[:-1, :-1].dot([1, 0])
        return RobotView(pos, dir, self.viewangle, self.viewdist)

    @property
    def vertices(self):
        vertices = np.vstack(p.vertices for p in self.polygons)
        return applyT(self.current_T, vertices.T).T

    def points_in_view(self, points):
        return self.current_robot_view().in_view(points)

    def trajectory_transforms(self, nframes):
        return self._parent_class.trajectory_transforms(self, nframes)

    @property
    def current_T(self):
        return self.trajectory.current_T

    @current_T.setter
    def current_T(self, T):
        self.trajectory.set_next_transform(T)

    @property
    def current_vel(self):
        return self.trajectory.current_vel

    @current_vel.setter
    def current_vel(self, vel):
        self.trajectory.current_vel = vel

def R2D_angle(theta):
    '''
    Return rotation matrix for theta

    >>> theta = 2.602584724164764  
    >>> R = R2D_angle(theta)
    >>> np.allclose(R, np.array([[-0.85821832, -0.51328483],
       [ 0.51328483, -0.85821832]]))
    True
    '''
    return np.array([[ np.cos(theta),  -np.sin(theta)],
                     [ np.sin(theta),   np.cos(theta)]])

def start_dir_like(dir):
    orig_dir = np.zeros_like(dir)
    orig_dir[0] = 1
    return orig_dir

def pos_dir_from_T(T):
    """
    >>> T = np.array([[-0.85821832, -0.51328483,  1.07977331],
    >>>    [ 0.51328483, -0.85821832,  0.14016698],
    >>>    [ 0.        ,  0.        ,  1.        ]])
    >>> pos_gt = np.array([ 1.07977331,  0.14016698])
    >>> dir_gt = np.array([-0.85821832,  0.51328483])
    >>> pos_dir_comp = pos_dir_from_T(T)
    >>> np.allclose(pos_gt, pos_dir_comp[0])
    True
    >>> np.allclose(dir_gt, pos_dir_comp[1])
    True
    """
    nD = T.shape[0]
    pos = T[:-1, -1]
    dir = T[:-1, :-1].dot(start_dir_like(pos))
    return (pos, dir)

def cross_prod_matrix(v):
    assert v.shape[0] == 3
    return [  [    0 , -v[2] ,  v[0]]
            , [ v[2] ,     0 , -v[1]]
            , [-v[0] ,  v[1] ,    0]]

def rodrigues_rotation(axis, angle):
    assert axis.shape[0] == 3
    K = np.array(cross_prod_matrix(axis))
    return np.eye(3) + np.sin(angle) * K + (1-np.cos(angle)) * (K.dot(K))

def normalized(v):
    norm_v = np.linalg.norm(v)
    return v / np.where(norm_v == 0, 1, norm_v)

def T_from_pos_dir(pos, dir):
    """
    >>> pos = np.random.rand(2) * np.random.randint(20)
    >>> theta = np.random.rand() * 2 * np.pi
    >>> R = R2D_angle(theta)
    >>> dir = R.dot([1, 0])
    >>> np.allclose(T_from_pos_dir(pos, dir)[:-1, :-1], R)
    True
    >>> np.allclose(T_from_pos_dir(pos, dir)[:-1, -1], pos)
    True
    """
    nD = dir.shape[0]
    assert nD <= 3, 'Cannot compute a rotation matrix for 4-D or more'
    orig_dir = start_dir_like(dir)
    abstheta = np.arccos(orig_dir.dot(dir)) # ambiguous to sign
    if nD == 2:
        dir = np.hstack((dir, 0))
        orig_dir = np.hstack((orig_dir, 0))
    axis = normalized(np.cross(orig_dir, dir)) # orig_dir -> dir
    R = rodrigues_rotation(axis, abstheta)
    T = np.eye(nD+1)
    T[:-1, :-1] = R[:-1, :-1]
    T[:-1, -1] = pos
    return T

class TrajInterface(object):
    def transforms(self, nframes):
        raise NotImplementedError()

class StaticTraj(object):
    def __init__(self, initpos=(0, 0), inittheta=0):
        self.current_T = T_from_pos_dir(
            initpos, np.array([np.cos(inittheta), np.sin(inittheta)]))

    def transforms(self, nframes):
        return itertools.repeat(self.current_T, nframes)

class PieceWiseLinearTraj(TrajInterface):
    def __init__(self, turnpoints, lin_vel, angular_vel, scale=1.0):
        self.turnpoints = turnpoints
        self.lin_vel = lin_vel
        self.angular_vel = angular_vel
        self.scale = scale
        self.current_T = np.eye(turnpoints.shape[1])
        self.current_vel = (lin_vel, 0)

    def transforms(self, nframes):
        '''
        Returns (position_t, direction_t, linear_velocity_{t+1},
        angular_velocity_{t+1})
        '''
        positions = self.turnpoints
        lin_vel = self.lin_vel
        angular_vel = self.angular_vel
        scale = self.scale
        prev_dir = None
        from_pos = scale * np.asarray(positions[:-1])
        to_pos = scale * np.asarray(positions[1:])
        for fp, tp in zip(from_pos, to_pos):
            dir = (tp - fp) / vnorm(tp-fp)

            if prev_dir is not None:
                to_dir = dir
                dir = prev_dir
                # Try rotation, if it increases the projection then we are good.
                after_rot = R2D_angle(angular_vel).dot(prev_dir)
                after_rot_proj = to_dir.dot(after_rot)
                before_rot_proj = to_dir.dot(prev_dir)
                angular_vel = np.sign(after_rot_proj - before_rot_proj) * angular_vel
                # Checks if dir is still on the same side of to_dir as prev_dir
                # Uses the fact that cross product is a measure of sine of
                # differences in orientation. As long as sine of the two
                # differences is same, the product is +ve and the robot must
                # keep rotating otherwise we have rotated too far and we must
                # stop.
                while np.cross(dir, to_dir) * np.cross(prev_dir, to_dir) > 0:
                    T = T_from_pos_dir(pos, dir)
                    self.current_T = T
                    self.current_vel = (0, angular_vel)
                    yield T
                    dir = R2D_angle(angular_vel).dot(dir)
                dir = to_dir

            #for i in range(nf+1):
            pos = fp
            vel = (tp - fp) * lin_vel / vnorm(tp - fp)
            # continue till pos is on the same side of tp as fp
            while np.dot((pos - tp), (fp - tp)) > 0:
                T = T_from_pos_dir(pos, dir)
                self.current_T = T
                self.current_vel = (lin_vel, 0)
                yield T
                pos = pos + vel
            prev_dir = dir

def norm_reward(r):
    return (r - REWARD_RANGE[0])/(REWARD_RANGE[1] - REWARD_RANGE[0])

class LandmarksVisualizer(object):
    def __init__(self, min, max, frame_period=10, scale=1):
        self._scale = scale
        self._name = "c"
        dims = np.asarray(max) - np.asarray(min)
        nrows = dims[1] * scale
        ncols = dims[0] * scale
        self._imgdims = (nrows, ncols, 3)
        self.frame_period = frame_period
        self._frame_count = 0
        self._initialized_window = False
        #cv2.waitKey(-1)

    def _init_window(self):
        if not self._initialized_window:
            cv2.namedWindow(self._name, flags=cv2.WINDOW_NORMAL)
        self._initialized_window = True

    def add_keypress_handle(self, keypress_event_handle):
        if not cv2.am_i_matplotlib:
            raise NotImplementedError("Not implemented with opencv")
        cv2.namedWindow(self._name).canvas.mpl_connect('key_press_event',
                                                       keypress_event_handle)

    def add_keyrelease_handle(self, keyrelease_event_handle):
        if not cv2.am_i_matplotlib:
            raise NotImplementedError("Not implemented with opencv")
        cv2.namedWindow(self._name).canvas.mpl_connect('key_release_event',
                                                       keyrelease_event_handle)

    def genframe(self, landmarks, robview=None, map_blocks=None
                 , highlight_ldmks_in_robot_view=True):
        colorfunc = lambda x: tuple(64*c+96 for c in cmap.get_cmap('gray')(x)[:3])
        img = white_img(self._imgdims)
        ldmks_loc = landmarks.locations
        colors = list(landmarks.colors.T)
        if ldmks_loc.shape[1] > 10:
            radius = 0.5 * self._scale
        else:
            radius = 4 * self._scale
        red = (0, 0, 255)
        blue = (255, 0, 0)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        black = (0., 0., 0.)

        if highlight_ldmks_in_robot_view:
            assert robview is not None
            in_view_ldmks = robview.in_view(ldmks_loc)
            colors = [(blue if in_view_ldmks[i] else colors[i]) for i in
                      range(ldmks_loc.shape[1])]

        if map_blocks is not None:
            f_scale = lambda x: self._scale * x
            maybereward = lambda r: DEFAULT_REWARD if r is None else r
            maybecolor = lambda c, r: (
                colorfunc( norm_reward( maybereward(r)))
                if c is None
                else c)
            for rr in reversed(map_blocks):
                img = rr.draw(img, maybecolor=maybecolor, scale=self._scale)

        for i in range(ldmks_loc.shape[1]):
            pt1 = ldmks_loc[:, i] * self._scale
            if ldmks_loc.shape[1] > 10:
                pt2 = pt1 + radius
                cv2.circle(img, tuple(pt1), radius, colors[i],
                              thickness=-1)
            else:
                cv2.circle(img, tuple(pt1), radius, colors[i], thickness=-1)
        return img

    def visualizeframe(self, frame, write=False):
        self._init_window()
        cv2.imshow(self._name, frame)
        cv2.waitKey(self.frame_period)
        if write:
            self.writeframe(frame)

    def writeframe(self, frame):
        fname = "/tmp/landmarkmap_%04d.png" % self._frame_count
        cv2.imwrite(fname, frame)
        self._frame_count += 1

    def visualizemap(self, map):
        for lmks in map.get_landmarks():
            self.visualizeframe(lmks)

def T_from_angle_pos(theta, pos):
    return np.array([[np.cos(theta),  np.sin(theta), pos[0]],
                     [-np.sin(theta), np.cos(theta), pos[1]],
                     [0,            0,               1]])

def get_robot_observations(lmmap, robblock, lmvis=None):
    """ Return a tuple of r, theta and ids for each frame"""
    """ v2.0 Return a tuple of lndmks in robot frame and ids for each frame"""
    map_t_iter = lmmap.get_map_traj()
    for map_t in map_t_iter:
        ldmks = map_t.landmarks.locations
        rew_r = map_t.regions
        robot_T = robblock.current_T
        robot_inputs = robblock.current_vel

        posdir = pos_dir_from_T(robot_T)
        robview = robblock.current_robot_view()
        #robview = RobotView(posdir[0], posdir[1], maxangle, maxdist)
        if lmvis is not None:
            img = lmvis.genframe(map_t.landmarks, robview, map_blocks=rew_r)
            #img = lmvis.drawrobot(robview, img)
            cv2.imshow(lmvis._name, img)
            lmvis.writeframe(img)
            cv2.waitKey(lmvis.frame_period)
        in_view_ldmks = robview.in_view(ldmks)
        selected_ldmks = ldmks[:, in_view_ldmks]
        pos = posdir[0].reshape(2,1)

        # v1.0 Need to update after new model has been implemented
        dists = np.sqrt(np.sum((selected_ldmks - pos)**2, 0))
        dir = posdir[1]
        #angles = np.arccos(dir.dot((selected_ldmks - pos))/dists)
        obsvecs = selected_ldmks - pos
        rob_theta = np.arctan2(dir[1], dir[0])
        angles = np.arctan2(obsvecs[1, :], obsvecs[0, :]) - rob_theta
        ldmks_idx = np.where(in_view_ldmks)

        # Get reward

        # Changed selected_ldmks to robot coordinate frame -> looks like we need to directly send           obsvecs with rotation according to heading
        # v2.0 Rename gen_obs 
        # NOTE: Modify R2D_Angle function based on dimensions of feature space
        ldmk_robot_obs = R2D_angle(rob_theta).dot(obsvecs)
        region_t = map_t.get_region_by_pos(robblock.vertices.T)
        yield (dists, angles, ldmks_idx[0],
               [float(pos[0]), float(pos[1]), rob_theta,
                                             float(robot_inputs[0]),
                                             float(robot_inputs[1])],
               ldmks, ldmk_robot_obs, region_t.reward, region_t.color)

class Trajectory(TrajInterface):
    def __init__(self, inittheta=0, initpos=(0, 0), deltheta=0, delpos=(0, 0)):
        self.inittheta = inittheta
        self.initpos = initpos
        self.deltheta = deltheta
        self.delpos = delpos

    def transforms(self, nframes):
        traj = dyn_trajectory(T_from_angle_pos(self.inittheta,
                                               self.initpos),
                              T_from_angle_pos(self.deltheta,
                                               self.delpos),
                              nframes)
        return traj

class InteractiveTrajectory(TrajInterface):
    def __init__(self, inittheta=0, initpos=(0, 0)):
        self.inittheta = inittheta
        self.initpos = initpos
        self._next_T = T_from_angle_pos(self.inittheta,
                                        self.initpos)
        self.current_vel = (0,0)

    def transforms(self, nframes):
        for n in xrange(nframes):
            yield self._next_T

    def set_next_transform(self, T):
        self._next_T = T

    @property
    def current_T(self):
        return self._next_T

def map_from_conf(map_conf, nframes):
    """ Generate LandmarkMap from configuration """
    rmlist = []
    for block_conf in map_conf:
        rm = RigidMotions(RigidBody2D(block_conf.landmarks(),
                                      block_conf)
                          , block_conf.trajectory_transforms(nframes))
        rmlist.append(rm)

    return LandmarkMap(rmlist)

def hundred_ldmk_map(sample_per_block=20):
    nframes = 150
    # The map consists of 5 rectangular rigid bodies with given shape and
    # initial position. Static bodies have deltheta=0, delpos=[0,0]
    gray = (100, 100, 100)
    map_conf = [
        # robot block
        RobotViewMapBlock(
            viewangle = 45*np.pi/180.
            , viewdist = 80
            , robotsize = 10
            , color = (0, 0, 255)
            , edgewidth = 1
            , trajectory = PieceWiseLinearTraj(
                np.array([[20, 130], [50,100], [35,50]])
                , 5, np.pi/25.)
            , name = 'Robot'
            )
        , RectangleMapBlock(
            nsamples = sample_per_block,
            color = gray,
            shape = [50,50],
            name = 'static1',
            trajectory=Trajectory(inittheta=0,
                                  initpos=[60, 90],
                                  deltheta=0,
                                  delpos=[0,0])),
        # prismatic
        RectangleMapBlock(
            nsamples=sample_per_block,
            color = gray,
            shape=[50, 10],
            name = 'prismatic',
            trajectory=Trajectory(inittheta=0,
                                  initpos=[10, 80],
                                  deltheta=0,
                                  delpos=[50./(nframes/2), 0])),
        RectangleMapBlock(
            nsamples=sample_per_block,
            color = gray,
            shape=[50, 50],
            name = 'static2',
            trajectory=Trajectory(inittheta=0,
                                  initpos=[60, 30],
                                  deltheta=0,
                                  delpos=[0, 0])),
        # revolute
        RectangleMapBlock(
            nsamples=sample_per_block,
            color = gray,
            shape=[25, 5],
            name = 'revolute',
            trajectory=Trajectory(inittheta=0,
                                  initpos=[35, 30],
                                  deltheta=2*np.pi/(nframes/2),
                                  delpos=[0, 0])),
        RectangleMapBlock(
            nsamples=sample_per_block,
            color = gray,
            shape=[10, 140],
            name = 'static3',
            trajectory=Trajectory(inittheta=0,
                                  initpos=[0, 0],
                                  deltheta=0,
                                  delpos=[0, 0]))
    ]

    lmmap = map_from_conf(map_conf, nframes)
    lmv = LandmarksVisualizer([0,0], [110, 140], frame_period=80)
    robblock = map_conf[0]
    return nframes, lmmap, lmv, robblock

if __name__ == '__main__':
    """ Run to see visualization of a dynamic map"""
    nframes, lmmap, lmv, robblock = hundred_ldmk_map()
    # to get the landmarks with ids that are being seen by robot
    for r, theta, id, rs, ldmk, ldmk_obs, rew, color in get_robot_observations(
        lmmap, robblock, # Do not pass visualizer to
                         # disable visualization
                         lmv):
        print(r, theta, id, rs)
