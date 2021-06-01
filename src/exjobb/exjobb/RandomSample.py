from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx
import timeit
from numba import njit
import open3d as o3d 

from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE
from exjobb.MotionPlanner import MotionPlanner
CELL_STEP_SIZE = 1*ROBOT_RADIUS
VISITED_TRESHOLD = 0.99*ROBOT_RADIUS
COVEREAGE_EFFICIENCY_GOAL = 0.95
MAX_ITERATIONS = 10000

TRAPPED = 0
ADVANCED = 1
REACHED = 2

class RandomSample(CPPSolver):

    def __init__(self, logger, motion_planner):
        
        self.logger = logger
        super().__init__(logger, motion_planner)
        self.name = "Random Sample"

    def get_cpp_path(self, start_point):
        self.start_tracking()
        coverage = 0
        self.path = np.array([start_point])
        self.points_to_mark = np.array([start_point])
        self.pcd.visit_point(start_point, ROBOT_RADIUS)
        corners = self.detect_biggest_square_plane(start_point)
        self.points_to_mark = corners

        k_coverage_complete = False
        waypoints = np.empty((0,3))
        k = 3
        
        while not k_coverage_complete:
            k_covered_points = self.get_k_covered_points(waypoints, k)
            mask = np.ones(len(self.pcd.points))
            mask[k_covered_points] = 0

            p = 


        self.print_stats(self.path)
        return self.path


    def get_k_covered_points(self, waypoints, k):
        covered_points = np.array([])
        for point in waypoints:
            covered_points = np.append(covered_points, self.pcd.points_idx_in_radius(point, ROBOT_RADIUS))
        
        points_idx, counts = np.unique(covered_points, return_counts=True)
        k_covered_points = points_idx[counts > k]
        return k_covered_points

    def has_been_visited(self, point, path=None):
        if path is None:
            path = self.path

        distances = np.linalg.norm(path - point, axis=1)
        return np.any(distances <= VISITED_TRESHOLD) 

    def is_blocked(self, from_point, to_point, path = None):
        if path is None:
            path = self.path

        if self.has_been_visited(to_point, path):
            return True

        

        if not self.motion_planner.is_valid_step(from_point, to_point):
            return True
        
        return False

    def get_points_to_mark(self):
        #return self.backtrack_list
        return self.points_to_mark