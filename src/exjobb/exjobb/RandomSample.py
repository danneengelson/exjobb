from os import replace
from exjobb.Parameters import ROBOT_SIZE
from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx
import timeit
from numba import njit
import open3d as o3d 
import pickle
from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE
from exjobb.MotionPlanner import MotionPlanner

DO_GET_WAYPOINTS = False

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
        self.move_to(start_point)
        #corners = self.detect_biggest_square_plane(start_point)
        #self.points_to_mark = corners
        if DO_GET_WAYPOINTS:
            waypoints = self.get_waypoints_for_k_coverage()
            with open('cached_random_waypoints.dictionary', 'wb') as cached_pcd_file:
                cache_data = {"waypoints": waypoints}
                pickle.dump(cache_data, cached_pcd_file)
        else:
            with open('cached_random_waypoints.dictionary', 'rb') as cached_pcd_file:
                cache_data = pickle.load(cached_pcd_file)
                waypoints = cache_data["waypoints"]
        
        self.print(len(waypoints))
        waypoints = self.remove_redundant_with_greedy(waypoints)
        self.print(len(waypoints))
        self.print("golow")
        self.follow_path(waypoints)
        self.print("golowasdf")
        coverage = self.pcd.get_coverage_efficiency()
        self.print("coverage" + str(coverage))
        self.print_stats(self.path)

        self.points_to_mark = np.array(waypoints)
        return self.path


    def remove_redundant_with_greedy(self, waypoints):
        sub = np.empty((1,3))
        uncovered = np.arange(len(self.pcd.points))
        covered = np.array([])
        left = waypoints

        while len(uncovered):
            new_covered = np.array([])
            for waypoint in left:
                covers = self.pcd.points_idx_in_radius(waypoint, ROBOT_SIZE/2)
                new = self.delete_values(covers, covered)
                new_covered = np.append(new_covered, len(new))

            max_idx = np.argmax(new_covered)
            max_point = left[max_idx]
            sub = np.append(sub, [max_point], axis=0)
            covers = self.pcd.points_idx_in_radius(max_point, ROBOT_SIZE/2)
            uncovered = self.delete_values(uncovered, covers)
            left = np.delete(left, max_idx, 0)
            self.print("new_covered" + str(new_covered))
            self.print("max_idx" + str(max_idx))
            self.print("nof points" + str(new_covered[max_idx]))
            self.print("max_point" + str(max_point))
            self.print("sub" + str(sub))
            self.print("covers" + str(covers))
            self.print("uncovered" + str(uncovered))
            self.print("left" + str(left))
            self.print("uncovered" + str(len(uncovered)))

        return  sub


    def get_waypoints_for_k_coverage(self):
        waypoints = np.empty((0,3))
        k = 2
        
        uncovered_points_idx = np.arange(len(self.pcd.points))
        covered_points_idx = np.array([], dtype=np.int)
        k_covered_points_idx = np.array([], dtype=np.int)
        
        while len(k_covered_points_idx) < len(self.pcd.points):
            if len(uncovered_points_idx):
                random_idx = np.random.choice(len(uncovered_points_idx), 1, replace=False)[0]
                new_waypoint = self.pcd.points[uncovered_points_idx[random_idx]]
            else:
                #self.print("ALL POINTS COVERED")
                random_idx = np.random.choice(len(covered_points_idx), 1, replace=False)[0]
                new_waypoint = self.pcd.points[covered_points_idx[random_idx]]

            waypoints = np.append(waypoints, [new_waypoint], axis=0)
            covered_points_idx = np.append(covered_points_idx, self.pcd.points_idx_in_radius(new_waypoint, ROBOT_RADIUS))
            #self.print("k_covered_points" + str(k_covered_points))
            #self.print("uncovered_points_idx" + str(uncovered_points_idx))
            #self.print("new_waypoint" + str(new_waypoint))
            #self.print("waypoints" + str(waypoints))
            #self.print("all_covered_points_idx" + str(all_covered_points_idx))
            uncovered_points_idx = self.delete_values(uncovered_points_idx, covered_points_idx)
            k_covered_points_idx = np.append(k_covered_points_idx, self.get_k_covered_points_idx(covered_points_idx, k))
            covered_points_idx = self.delete_values_not_unique(covered_points_idx, k_covered_points_idx)
        
        return waypoints
            
            #self.print(len(k_covered_points_idx))

    def get_k_covered_points_idx(self, covered_points_idx, k):
        points_idx, counts = np.unique(covered_points_idx, return_counts=True)
        k_covered_points = points_idx[counts >= k]
        return k_covered_points

    def delete_values(self, array, values):
        return array[ np.isin(array, values, assume_unique=True, invert=True) ]

    def delete_values_not_unique(self, array, values):
        return array[ np.isin(array, values, invert=True) ]


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