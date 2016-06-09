#!/usr/bin/env python

import rospy
from read_config import read_config
from cse_190_final.msg import AStarPath, PolicyList
from std_msgs.msg import String, Float32, Bool
from astar import astar
from mdp import mdp
from qlearning import qlearning

class Robot():
    def __init__(self):
        """Read config file and setup ROS things"""
        self.config = read_config()
        rospy.init_node("robot")

        # result publisher
        astar_publisher = rospy.Publisher(
                "/results/path_list",
                AStarPath,
                queue_size = 10
        )
        mdp_publisher = rospy.Publisher(
                "/results/policy_list",
                PolicyList,
                queue_size = 10
        )
        qlearning_publisher = rospy.Publisher(
                "/results/qlearning_list",
                PolicyList,
                queue_size = 10
        )
        # qvalues_publisher = rospy.Publisher(
        #         "/results/qvalue_list",
        #         QvalueLlist,
        #         queue_size = 10
        # )
        complete_publisher = rospy.Publisher(
                "/map_node/sim_complete",
                Bool,
                queue_size = 10
        )


        rospy.sleep(1)
        policyList = PolicyList()
        mdpRes = mdp()
        # print mdpRes
        policyList.data = reduce(lambda x,y: x+y , mdpRes)
        mdp_publisher.publish(policyList)
        rospy.sleep(1)

        qlearningList = PolicyList()
        qlearningObj = qlearning()
        qlearningRes = qlearningObj.exeqlearning()
        # print qlearningRes
        qlearningList.data = reduce(lambda x,y: x+y , qlearningRes)
        qlearning_publisher.publish(qlearningList)
        rospy.sleep(1)

        for i in range(0, len(mdpRes)):
            print "Row: %d" % i
            print "      MDP",
            print mdpRes[i]
            print "qlearning",
            print qlearningRes[i]
        rospy.sleep(1)

        complete_publisher.publish(True)
        rospy.sleep(2)
        rospy.signal_shutdown("Finish")


if __name__ == '__main__':
    rs = Robot()