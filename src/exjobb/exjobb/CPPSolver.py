from exjobb.Tree import Tree
import os
import numpy as np
from collections import deque
import timeit
import networkx as nx

import timeit
from heapq import heappush, heappop
import tracemalloc 
import linecache

from networkx.algorithms.shortest_paths.weighted import _weight_function
from exjobb.MotionPlanner import MotionPlanner
from exjobb.Parameters import ROBOT_SIZE, ROBOT_STEP_SIZE
ROBOT_RADIUS = ROBOT_SIZE/2
STEP_SIZE = ROBOT_SIZE
UNTRAVERSABLE_THRESHHOLD = 1.5*STEP_SIZE

class CPPSolver:

    def __init__(self, logger, motion_planner):
        self.name = "General CPP"
        self.logger = logger
        self.pcd = motion_planner.pcd
        self.motion_planner = motion_planner
        self.current_position = None
        self.path = np.empty((0,3))
    
    

    def start_tracking(self):
        tracemalloc.start()
        self.start_time = timeit.default_timer()

    def print_stats(self, path):
        end_time = timeit.default_timer()
        snapshot = tracemalloc.take_snapshot()
        nbr_of_points_in_path = len(path)

        def get_memory_consumption(snapshot, key_type='lineno'):
            #To see consumtion per line: https://docs.python.org/3/library/tracemalloc.html
            snapshot = snapshot.filter_traces((
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            ))
            top_stats = snapshot.statistics(key_type)
            total = sum(stat.size for stat in top_stats)
            return total / 1024
        
        def get_length_of_path(path):
            length = 0
            for point_idx in range(len(path) - 1):
                length += np.linalg.norm( path[point_idx] - path[point_idx + 1] )
            return length

        def get_total_rotation(path):
            rotation = 0
            for point_idx in range(len(path) - 2):
                prev = (path[point_idx+1] - path[point_idx]) / np.linalg.norm( path[point_idx] - path[point_idx + 1])
                next = (path[point_idx+2] - path[point_idx+1]) / np.linalg.norm( path[point_idx+2] - path[point_idx + 1])
                dot_product = np.dot(prev, next)
                curr_rotation = np.arccos(dot_product)
                if not np.isnan(curr_rotation):
                    rotation += abs(curr_rotation)
            return rotation

        length_of_path = get_length_of_path(path)
        rotation = get_total_rotation(path)
        unessecary_coverage_mean = self.pcd.get_multiple_coverage(path, ROBOT_RADIUS)
        computational_time = end_time - self.start_time
        coverage = self.pcd.get_coverage_efficiency()
        memory_consumption = get_memory_consumption(snapshot)

        print_text = "\n" + "=" * 20
        print_text += "\nAlgorithm: " + self.name
        print_text += "\nCoverage efficiency: " + str(round(coverage*100, 2)) + "%"
        print_text += "\nNumber of waypoints: " + str(nbr_of_points_in_path)
        print_text += "\nLength of path: " + str(round(length_of_path)) + " meter"
        print_text += "\nTotal rotation: " + str(round(rotation)) + " rad"
        print_text += "\nVisits per point: " + str(unessecary_coverage_mean)
        print_text += "\nComputational time: " + str(round(computational_time, 1)) + " sec" 
        print_text += "\nMemory consumption: " + str(round(memory_consumption, 1)) + " KiB"

        

        print_text += "\n" + "=" * 20
        self.logger.info(print_text)


    def follow_path(self, path):
        self.path = np.append( self.path, path, axis=0 )
        self.pcd.visit_path(path, ROBOT_RADIUS)

    def move_to(self, point):
        if len(self.path) > 0:
            curr_position = self.path[-1]
            self.pcd.visit_point(point, curr_position, ROBOT_RADIUS)
            
        self.path = np.append( self.path, [point], axis=0 )
        

    def print(self, object_to_print):
        self.logger.info(str(object_to_print))