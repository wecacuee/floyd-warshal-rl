"""
Generates maze by DFS and closes loops some of the times.
"""

import os
import datetime
import subprocess
from shutil import copyfile
import numpy as np
import random
import math
import time
import sys

def getRandomEvenCoordinate(rows, cols):
    # Shape must be bigger than 2 and odd
    assert (rows > 2 and rows % 2 == 1)
    assert (cols > 2 and cols % 2 == 1)

    rowIdx = 2*int(random.randint(1, math.floor(rows/2))) - 1
    colIdx = 2*int(random.randint(1, math.floor(cols/2))) - 1

    return rowIdx, colIdx

def findVisitableCells(r, c, mazeT):
    """
    Find visitable cells in the neighborhood of r,c
    """
    shape = mazeT.shape

    visitableCells = []
    if (r - 2) > 0 and mazeT[r - 2, c] == 0:
        visitableCells.append(((r - 2, c), (r - 1, c)))
    if (c - 2) > 0 and mazeT[r, c-2] == 0:
        visitableCells.append(((r, c-2), (r, c-1)))
    if (r + 2) < (shape[0]-1) and mazeT[r + 2, c] == 0:
        visitableCells.append(((r+2, c), (r+1, c)))
    if (c + 2) < (shape[1]-1) and mazeT[r, c+2] == 0:
        visitableCells.append(((r, c+2), (r, c+1)))

    return visitableCells

def gen_maze_imp(rows=11, cols=11, visited=1):
    """
    Recursive version of Maze generator
    """
    unvisited = 1 - visited
    mazeT = np.ones((rows, cols), dtype=np.int32) * unvisited
    start = getRandomEvenCoordinate(rows, cols)
    stack = [start]

    while len(stack) != 0:
        r, c = stack[-1]
        mazeT[r, c] = visited
        visitableCells = findVisitableCells(r, c, mazeT)
        if len(visitableCells) > 0:
            choice = random.choice(visitableCells)
            mazeT[choice[1]] = visited
            stack.append(choice[0])
        else:
            stack.pop()
    return mazeT


def four_nbr(step):
    return np.asarray([[0, 1], [0, -1], [1, 0], [-1, 0]]) * step


def findVisitableCells_(idx, mazeT, unvisited=0, nbrs = four_nbr(2)):
    poss_idx = idx + nbrs
    in_arr_idx = poss_idx[np.all((poss_idx < shp) & (poss_idx > 0), axis=1), :]
    return  in_arr_idx[mazeT[in_arr_idx[:, 0], in_arr_idx[:, 1]] == unvisited, :]


def fill_maze_dfs(mazeT, idx,
                 visited   = 1,
                 unvisited = 0,
                 nbrs      = four_nbr(2),
                 prob_loop = 0.25,
                 rng       = np.random.RandomState()):
    mazeT[tuple(idx)] = visited
    visitableCells = findVisitableCells_(
        idx, mazeT, unvisited = unvisited, nbrs = nbrs)
    if visitableCells.shape[0]:
        rng.shuffle(visitableCells)
        for idx0 in visitableCells:
            if mazeT[tuple(idx0)] == unvisited:
                mazeT[tuple((idx + idx0) // 2)] = visited
                #print(mazeT) # uncomment to see the evolution of mazeT
                fill_maze_dfs(mazeT, idx0)
            elif rng.uniform() < prob_loop:
                mazeT[tuple((idx + idx0) // 2)] = visited


    return mazeT


def gen_maze_rec(shape, unvisited = 0, visited = 1,
                 fill_maze_dfs = fill_maze_dfs,
                 nbrs      = four_nbr(2),
                 rng = np.random.RandomState()):
    """
    Recursive version of Maze generator
    """
    shape = np.asarray(shape)
    mazeT = np.empty(np.array(shape)*2 + 1)
    mazeT.fill(unvisited)
    start_visited = (rng.randint(0, shape.max(), size=shape.shape[0]) % shape)*2 + 1
    return fill_maze_dfs(
        mazeT, start_visited,
        unvisited = unvisited,
        visited = visited, rng = rng)


# In [90]: %timeit gen_maze_rec((9,9))
# 5.83 ms ± 1.01 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
#
# In [91]: %timeit gen_maze_imp(9,9)
# 317 µs ± 812 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
gen_maze = gen_maze_imp

if __name__ == '__main__':
    print(gen_maze(9,9))
