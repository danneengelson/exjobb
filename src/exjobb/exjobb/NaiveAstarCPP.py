from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx

from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE

RRT_STEP_SIZE = 3*STEP_SIZE
COVEREAGE_EFFICIENCY_GOAL = 0.8
MAX_ITERATIONS = 10000
GOAL_CHECK_FREQUENCY = 50

ASTAR_FLAG = -1

TRAPPED = 0
ADVANCED = 1
REACHED = 2

class NaiveRAstarCPP(CPPSolver):

    def __init__(self, logger, ground_pcd):
        super().__init__(logger, ground_pcd)
        self.name = "Naive Astar CPP"

    def get_cpp_path(self, start_point):
        self.start_tracking()
        path = self.get_astar_path(start_point)
        self.print_stats(path)
        return path

    