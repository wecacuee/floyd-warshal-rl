""" Dealing with polygons """
import numpy as np

def vertices2edges(vertices):
    """
    >>> vertices = [[3, 3], [2, 2], [4, 2.5]]
    >>> vertices2edges(vertices)
    [(array([ 3.,  3.]), array([ 2.,  2.])), (array([ 2.,  2.]), array([ 4. ,  2.5])), (array([ 4. ,  2.5]), array([ 3.,  3.]))]
    """
    vertices = np.asarray(vertices)
    return zip(vertices, np.roll(vertices, -1, axis=0))

def nullspace(M, n):
    """
    Returns the last n columns from V where U*S*V.H = M is SVD of M
    """
    U, S, V_H = np.linalg.svd(M)
    return V_H.T[:, -n:]

def hyplane(points):
    """ points: N x D
    Returns : h such that h^T x = 0 for points x on the plane

    >>> vertices = np.array([[3, 3], [2, 2], [4, 2.5]])
    >>> hyplane(vertices[:2, :])
    array([  7.07106781e-01,  -7.07106781e-01,   1.73472348e-15])

    >>> hyplane(vertices[1:3, :])
    array([ 0.13736056, -0.54944226,  0.82416338])

    >>> hyplane(vertices[(2,0), :])
    array([-0.10783277, -0.21566555,  0.97049496])
    """
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    plane_vec =  nullspace(points_h, 1).flatten()
    # Normalize the last dimension of the plane representation to be +ve
    sign = lambda x : (1 if x >= 0 else -1)
    return sign(plane_vec[-1]) * plane_vec

def whichside(hyplane, x):
    """ Returns 1 of point is away from origin else -1 as compared to hyplane

    x is either (D,) array indicating single D-dimensional point.
         OR (N, D) array indicating N D-dimensional points.

    Returns: a scalar 1 or (N,) array of 1s and 0s

    >>> vertices = np.array([[3, 3], [2, 2], [4, 2.5]])
    >>> whichside(np.array([0.7, -0.7, 0]), np.zeros(2))
    array(1)

    >>> whichside(np.array([0.13, -0.54, 0.82]), np.zeros(2))
    array(1)

    >>> whichside(np.array([-0.10, -0.21, 0.97]), np.zeros(2))
    array(1)

    >>> whichside(np.array([-0.10, -0.21, 0.97]), np.zeros((1, 2)).T)
    array([1])

    >>> whichside(np.array([0.13, -0.54, 0.82])
    ...     , np.array([[0, 0], [1, 2], [2,3], [-4, -5]]).T)
    array([1, 0, 0, 1])
    """
    X_h = (np.hstack((x, 1))
           if x.ndim == 1 else
           np.vstack((x, np.ones((1, x.shape[1])))))
    return np.where(np.dot(hyplane, X_h) >= 0,
                    np.array(1, dtype=np.int),
                    np.array(0, dtype=np.int))

def convex_polygon_contains(vertices, x):
    """
    >>> vertices = np.array([[3, 3], [2, 2], [4, 2.5]])
    >>> x = np.mean(vertices, axis=0)
    >>> convex_polygon_contains(vertices, x)
    True

    >>> convex_polygon_contains(vertices, np.array([2.52, 2.5]))
    True

    >>> convex_polygon_contains(vertices, np.array([2.48, 2.5]))
    False

    >>> convex_polygon_contains(vertices, np.array([3.5, 2.77]))
    False

    >>> convex_polygon_contains(vertices, np.array([3.5, 2.73]))
    True

    >>> convex_polygon_contains(vertices, np.array([3.1, 2.25]))
    False

    >>> convex_polygon_contains(vertices, np.array([2.9, 2.25]))
    True

    >>> x = np.array([ [2.52, 2.5],
    ... [2.48, 2.5],
    ... [3.5, 2.77],
    ... [3.5, 2.73],
    ... [3.1, 2.25],
    ... [2.9, 2.25],
    ... ]).T
    >>> np.allclose(convex_polygon_contains(vertices, x),
    ...     np.array([ True, False, False,  True, False,  True],
    ...     dtype=np.bool))
    True
    """
    lines = [hyplane(np.vstack((v1, v2)))
             for v1, v2 in vertices2edges(vertices)]
    initial = (np.ones(x.shape[1], dtype=np.bool)
               if x.ndim > 1
               else np.array(True, dtype=np.bool))
    same_sides = [whichside(line, v) == whichside(line, x)
                     for line, v in zip(lines, np.roll(vertices, -2, axis=0))]
    return reduce(lambda a, side: a & side , same_sides , initial)
