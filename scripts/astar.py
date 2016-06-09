#!/usr/bin/env python

import heapq as hq
from read_config import read_config

def astar():
    # read the config
    config = read_config()
    move_list = config['move_list']
    start = config['start']
    goal = config['goal']
    walls = config['walls']
    pits = config['pits']
    map_size = config['map_size']

    # add start tuple
    h = []
    hq.heappush(h, (heuristics(start, goal), start, 0))
    prev = [[[-1 for i in range(2)] for j in range(map_size[1])] for k in range(map_size[0])]
    visited = [[False for i in range(map_size[1])] for j in range(map_size[0])]

    # loop
    while len(h) != 0:
        cur = hq.heappop(h)
        visited[cur[1][0]][cur[1][1]] = True
        # print cur
        # if cur is goal, break
        if cur[1][0] == goal[0] and cur[1][1] == goal[1]:
            break

        # otherwise, add its neighbours
        for i in range(0, 4):
            next = [sum(x) for x in zip(cur[1], move_list[i])]
            # out of bound check, pits and walls check
            if next[0] < 0 or next[0] >= map_size[0] or next[1] < 0 or next[1] >= map_size[1]\
            or next in pits or next in walls or visited[next[0]][next[1]]:
                continue
            hq.heappush(h, (heuristics(next, goal) + cur[2] + 1, next, cur[2] + 1))
            prev[next[0]][next[1]] = cur[1]

    res = []
    cur = goal
    while cur != start:
        # print cur
        res.append(cur)
        cur = prev[cur[0]][cur[1]]
    res.append(cur)
    res.reverse()
    return res

def heuristics(loc, goal):
    return abs(loc[0] - goal[0]) + abs(loc[1] - goal[1])
