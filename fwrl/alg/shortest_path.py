"""Computes shortest path abstractly
"""
from functools import partial
import math

from umcog.memoize import MEMOIZE_METHOD
from umcog.functools import groupby, getitem, head, revargs


class Graph:
    def __init__(self, edges):
        self.edges = edges

    @MEMOIZE_METHOD
    def out_adj_list(self):
        return groupby(self.edges.keys(), valfun = partial(getitem, n = 1))

    @MEMOIZE_METHOD
    def in_adj_list(self):
        return groupby(self.edges.keys(), keyfun = partial(getitem, n = 1),
                       valfun = head)

    def in_neighbors(self, x):
        return self.in_adj_list()[x]

    def out_neighbors(self, x):
        return self.out_adj_list()[x]

    def distances(self, i, j):
        return self.edges[i, j]


def shortest_path(start, goal, neighbors, distances, visited=None,
                  memory=None, visted_gen=set, memory_gen=dict):
    """Returns shortest path length and the shortest path

          (2)
    0.2 /     \ -10
    (0) ------ (1)
          0.1
    >>> g = Graph( {(0, 1): 0.1, (0, 2): 0.2, (2, 1): -10})
    >>> shortest_path(0, 1, g.out_neighbors, g.distances)
    (-9.8, [0, 2, 1])
    >>> shortest_path(1, 0, g.in_neighbors,
    ...               revargs(g.distances))
    (-9.8, [1, 2, 0])


          (2)
        /  |  \
    (0)    |   (3)
        \  |   /
          (1)
    >>> g2 = Graph( {(0, 1): 0.1, (0, 2): 0.5, (1, 2): 0.3, (2, 3): 0.1,  (2, 3): 0.5} )
    >>> shortest_path(0, 3, g2.out_neighbors, g2.distances)
    (0.9, [0, 1, 2, 3])
    >>> shortest_path(3, 0, g2.in_neighbors, revargs(g2.distances))
    (0.9, [3, 2, 1, 0])


          (2)
        /  |  \
    (0)    |   (3)
        \  |   /
          (1)
    >>> g3 = Graph( {(0, 1): 0.1, (2, 0): 0.5, (1, 2): 0.3, (2, 3): 0.1,  (2, 3): 0.5} )
    >>> shortest_path(0, 3, g3.out_neighbors, g3.distances)
    (0.9, [0, 1, 2, 3])
    >>> shortest_path(3, 0, g3.in_neighbors, revargs(g3.distances))
    (0.9, [3, 2, 1, 0])

    """
    assert start is not None and goal is not None, ""

    if memory is None:
        memory = memory_gen()

    if visited is None:
        visited = set()

    if (start, goal) in memory:
        return memory[start, goal]

    if (start == goal):
        memory[start, goal] = 0, [goal]
    else:
        visited.add(start)
        nbrs = neighbors(start)
        unvisited_nbrs = set(nbrs).difference(visited)
        if not len(unvisited_nbrs):
            memory[start, goal] = float('inf'), []
        else:
            #visited = visited.union(unvisited_nbrs)
            nbr_lengths, nbr_paths = zip(*[
                shortest_path(nbr, goal, neighbors, distances,
                                  visited=visited, memory=memory)
                for nbr in unvisited_nbrs])
            lengths = [(nbr_l + distances(start, nbr))
                       for nbr, nbr_l in zip(unvisited_nbrs, nbr_lengths)]
            paths = [[start] + p for p in nbr_paths]
            length, path = min(zip(lengths, paths),
                               key = lambda a: a[0], default = float('inf'))
            if math.isinf(length):
                memory[start, goal] = length, []
            else:
                memory[start, goal] = length, path

    return memory[start, goal]
