#!/usr/bin/env python

from read_config import read_config
import random
from cse_190_final.msg import AStarPath, PolicyList
import image_util2

class Grid():
    def __init__(self):
        self.N = 0
        self.S = 0
        self.W = 0
        self.E = 0
        self.countN = 0
        self.countS = 0
        self.countW = 0
        self.countE = 0

    def __str__(self):
        return "N: %f, S: %f, W: %f, E: %f (cN: %d, cS: %d, cW: %d, cE: %d)" \
        % (self.N, self.S, self.W, self.E, self.countN, self.countS, self.countW, self.countE)

    def __repr__(self):
        return "N: %f, S: %f, W: %f, E: %f (cN: %d, cS: %d, cW: %d, cE: %d)" \
        % (self.N, self.S, self.W, self.E, self.countN, self.countS, self.countW, self.countE)

class qlearning():
    def __init__(self):
        # read the config
        random.seed(0)
        config = read_config()
        self.goal = config['goal']
        self.walls = config['walls']
        self.pits = config['pits']
        self.map_size = config['map_size']
        self.alpha = config['alpha']
        self.constant = config['constant']
        self.maxL = config['maxL']

        self.reward_for_reaching_goal = config['reward_for_reaching_goal']
        self.reward_for_falling_in_pit = config['reward_for_falling_in_pit']
        self.reward_for_hitting_wall = config['reward_for_hitting_wall']
        self.reward_for_each_step = config['reward_for_each_step']

        self.max_iterations = config['max_iterations']
        self.threshold_difference = config['threshold_difference']
        self.discount_factor = config['discount_factor']

        self.uncertainty = config['uncertainty']
        prob_move_forward = config['prob_move_forward']
        prob_move_backward = config['prob_move_backward']
        prob_move_left = config['prob_move_left']
        prob_move_right = config['prob_move_right']
        self.prob_move = [prob_move_forward, prob_move_backward, prob_move_left, prob_move_right]

        self.qvalues = [[Grid() for i in range(self.map_size[1])] for j in range(self.map_size[0])]
        self.generate_intermedia_video = config['generate_intermedia_video']
        self.video_iteration = 0

    def exeqlearning(self):
        # self.max_iterations = 2
        i = 0
        diff = self.threshold_difference + 1
        while i < self.max_iterations and diff > self.threshold_difference:
            i += 1
            # print "current iteration: %d" % i
            startState = self.resetStartState()
            # print "start state: [%d, %d]" % (startState[0], startState[1])
            diff = self.runqlearning(startState)

        print "iteration: ",
        print i
        print "======== qvalues ========"
        for i in range(0, self.map_size[0]):
            print "Row: %d" % i
            print self.qvalues[i]
        print "-------------------------"

        image_util2.save_final_image(self.qvalues)

        if self.generate_intermedia_video:
            image_util2.generate_video(self.video_iteration)

        res = [['' for i in range(self.map_size[1])] for j in range(self.map_size[0])]
        for row in range(0, self.map_size[0]):
            for col in range(0, self.map_size[1]):
                if [row, col] == self.goal:
                    res[row][col] = "GOAL"
                elif [row, col] in self.pits:
                    res[row][col] = "PIT"
                elif [row, col] in self.walls:
                    res[row][col] = "WALL"
                else:
                    res[row][col] = self.optimalPolicy(row, col)
        return res
    
    def resetStartState(self):
        randState = [random.randint(0, self.map_size[0]-1), random.randint(0, self.map_size[1]-1)]
        while randState == self.goal or randState in self.pits or randState in self.walls:
            randState = [random.randint(0, self.map_size[0]-1), random.randint(0, self.map_size[1]-1)]
        return randState
        
    def runqlearning(self, startState):
        curState = startState
        diff = 0
        stop = False
        while stop == False:
            # choose the action based on cur state's fvalues (qvalue + l)
            action = self.chooseAction(curState)
            # print "action: " + action

            # execute the action
            nextState, reward, stop = self.exeAction(curState, action)
            # print "nextState: [%d, %d]" % (nextState[0], nextState[1])
            # print "reward: %f" % reward
            # print "stop: %d" % stop

            # publish qvalues
            if self.generate_intermedia_video:
                image_util2.save_image_for_iteration(self.qvalues, curState, action, self.video_iteration)
                self.video_iteration += 1

            # update q value
            if stop:
                u = reward + self.discount_factor * self.getTerminalReward(nextState)
                old_qvalue = getattr(self.qvalues[curState[0]][curState[1]], action)
                new_qvalue = (1 - self.alpha) * old_qvalue + self.alpha * u
                setattr(self.qvalues[curState[0]][curState[1]], action, new_qvalue)
                diff += abs(new_qvalue - old_qvalue)
            else:
                u = reward + self.discount_factor * self.maxFvalue(nextState)
                old_qvalue = getattr(self.qvalues[curState[0]][curState[1]], action)
                new_qvalue = (1 - self.alpha) * old_qvalue + self.alpha * u
                setattr(self.qvalues[curState[0]][curState[1]], action, new_qvalue)
                diff += abs(new_qvalue - old_qvalue)

            self.updateCount(curState, action)

            # move on to next state
            curState = nextState

            # print "======== qvalues ========"
            # print self.qvalues
            # print "-------------------------"


        return diff
    
    def chooseAction(self, curState):
        curQvalue = self.qvalues[curState[0]][curState[1]]
        max_value = curQvalue.E + (self.maxL if curQvalue.countE == 0 else self.constant / curQvalue.countE)
        action = "E"
        for a in ['W', 'S', 'N']:
            fvalue = getattr(curQvalue, a) + \
            (self.maxL if getattr(curQvalue, "count" + a) == 0 else self.constant / getattr(curQvalue, "count" + a))
            if fvalue > max_value:
                max_value = fvalue
                action = a
        return action

    def exeAction(self, curState, action):
        # decide the actual direction
        if self.uncertainty == 1:
            direction = self.decideDirection(action, self.prob_move)
        else:
            direction = action
        nextState = [sum(x) for x in zip(curState, self.get_move(direction))]
        # check if it hits the wall
        if nextState[0] < 0 or nextState[0] >= self.map_size[0]\
        or nextState[1] < 0 or nextState[1] >= self.map_size[1]\
        or nextState in self.walls:
            nextState = curState
            return nextState, self.reward_for_hitting_wall, False
        # or it doesn't hit the wall
        else:
            if nextState in self.pits or nextState == self.goal:
                return nextState, self.reward_for_each_step, True
            else:
                return nextState, self.reward_for_each_step, False

    def getTerminalReward(self, state):
        if state == self.goal:
            return self.reward_for_reaching_goal
        elif state in self.pits:
            return self.reward_for_falling_in_pit
        else:
            return 0

    def maxFvalue(self, nextState):
        nextQvalue = self.qvalues[nextState[0]][nextState[1]]
        return max(nextQvalue.N + (self.maxL if nextQvalue.countN == 0 else self.constant / nextQvalue.countN),\
        nextQvalue.S + (self.maxL if nextQvalue.countS == 0 else self.constant / nextQvalue.countS),\
        nextQvalue.W + (self.maxL if nextQvalue.countW == 0 else self.constant / nextQvalue.countW),\
        nextQvalue.E + (self.maxL if nextQvalue.countE == 0 else self.constant / nextQvalue.countE))

    def updateCount(self, curState, action):
        curQvalue = self.qvalues[curState[0]][curState[1]]
        setattr(curQvalue, "count" + action, getattr(curQvalue, "count" + action) + 1)

    def optimalPolicy(self, row, col):
        qvalue = self.qvalues[row][col]
        maxQvalue = max(qvalue.N, qvalue.S, qvalue.W, qvalue.E)
        for direction in ("N", "S", "W", "E"):
            if maxQvalue == getattr(qvalue, direction):
                return direction

    def get_move(self, x):
        return {
            'E': [0, 1],
            'W': [0, -1],
            'S': [1, 0],
            'N': [-1, 0],
        }[x]

    def decideDirection(self, action, prob_move):
        r = random.random()

        if action == 'E':
            if r < prob_move[0]:
                return 'E'
            elif r < prob_move[0] + prob_move[1]:
                return 'W'
            elif r < prob_move[0] + prob_move[1] + prob_move[2]:
                return 'N'
            else:
                return 'S'

        elif action == 'W':
            if r < prob_move[0]:
                return 'W'
            elif r < prob_move[0] + prob_move[1]:
                return 'E'
            elif r < prob_move[0] + prob_move[1] + prob_move[2]:
                return 'S'
            else:
                return 'N'

        elif action == 'S':
            if r < prob_move[0]:
                return 'S'
            elif r < prob_move[0] + prob_move[1]:
                return 'N'
            elif r < prob_move[0] + prob_move[1] + prob_move[2]:
                return 'E'
            else:
                return 'W'

        elif action == 'N':
            if r < prob_move[0]:
                return 'N'
            elif r < prob_move[0] + prob_move[1]:
                return 'S'
            elif r < prob_move[0] + prob_move[1] + prob_move[2]:
                return 'W'
            else:
                return 'E'
