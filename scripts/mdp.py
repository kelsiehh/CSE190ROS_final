#!/usr/bin/env python

import heapq as hq
from read_config import read_config

def mdp():
    # read the config
    config = read_config()
    move_list = config['move_list']
    start = config['start']
    goal = config['goal']
    walls = config['walls']
    pits = config['pits']
    map_size = config['map_size']

    reward_for_reaching_goal = config['reward_for_reaching_goal']
    reward_for_falling_in_pit = config['reward_for_falling_in_pit']
    reward_for_hitting_wall = config['reward_for_hitting_wall']
    reward_for_each_step = config['reward_for_each_step']

    max_iterations = config['max_iterations']
    threshold_difference = config['threshold_difference']
    discount_factor = config['discount_factor']

    prob_move_forward = config['prob_move_forward']
    prob_move_backward = config['prob_move_backward']
    prob_move_left = config['prob_move_left']
    prob_move_right = config['prob_move_right']
    prob_move = [prob_move_forward, prob_move_backward, prob_move_left, prob_move_right]

    # initialize value map
    value = [[0 for i in range(map_size[1])] for j in range(map_size[0])]
    policy = [['' for i in range(map_size[1])] for j in range(map_size[0])]
    tmp_value = [[0 for i in range(map_size[1])] for j in range(map_size[0])]
    # add goal and pits value to value, add goal, wall and pits to policy
    value[goal[0]][goal[1]] = reward_for_reaching_goal
    policy[goal[0]][goal[1]] = 'GOAL'
    tmp_value[goal[0]][goal[1]] = reward_for_reaching_goal
    for pit in pits:
        value[pit[0]][pit[1]] = reward_for_falling_in_pit
        policy[pit[0]][pit[1]] = 'PIT'
        tmp_value[pit[0]][pit[1]] = reward_for_falling_in_pit
    for wall in walls:
        policy[wall[0]][wall[1]] = 'WALL'

    diff = threshold_difference + 1
    i = 0
    # max_iterations = 10
    while i < max_iterations and diff >= threshold_difference:
        i += 1
        diff = 0
        for row in range(0, map_size[0]):
            for col in range(0, map_size[1]):
                if [row, col] == goal or [row, col] in pits or [row, col] in walls:
                    continue
                max_value = float('-inf')
                max_action = ''
                for action in ['E', 'W', 'S', 'N']:
                    new_value = 0
                    for possible_action in ['E', 'W', 'S', 'N']:
                        prob = get_prob(action, prob_move, possible_action)
                        land_pos = [sum(x) for x in zip([row, col], get_move(possible_action))]
                        # check if it hits the wall
                        reward_and_value = 0
                        if land_pos[0] < 0 or land_pos[0] >= map_size[0]\
                        or land_pos[1] < 0 or land_pos[1] >= map_size[1]\
                        or land_pos in walls:
                            reward = reward_for_hitting_wall
                            future_value = discount_factor * value[row][col]
                            reward_and_value = reward + future_value
                        # or it doesn't hit the wall, move
                        else:
                            reward = reward_for_each_step
                            future_value = discount_factor * value[land_pos[0]][land_pos[1]]
                            reward_and_value = reward + future_value
                        new_value += prob * reward_and_value
                    # update max value and action
                    # if new_value == max_value:
                    #     max_action += action
                    if new_value >= max_value:
                        max_value = new_value
                        max_action = action
                diff += abs(value[row][col] - max_value)
                tmp_value[row][col] = max_value
                policy[row][col] = max_action
        value = tmp_value
    return policy

def get_move(x):
    return {
        'E': [0, 1],
        'W': [0, -1],
        'S': [1, 0],
        'N': [-1, 0],
    }[x]

def get_prob(action, prob_move, possible_action):
    if action == 'E':
        return {
            'E': prob_move[0],
            'W': prob_move[1],
            'N': prob_move[2],
            'S': prob_move[3],
        }[possible_action]
    elif action == 'W':
        return {
            'W': prob_move[0],
            'E': prob_move[1],
            'S': prob_move[2],
            'N': prob_move[3],
        }[possible_action]
    elif action == 'S':
        return {
            'S': prob_move[0],
            'N': prob_move[1],
            'E': prob_move[2],
            'W': prob_move[3],
        }[possible_action]
    elif action == 'N':
        return {
            'N': prob_move[0],
            'S': prob_move[1],
            'W': prob_move[2],
            'E': prob_move[3],
        }[possible_action]














