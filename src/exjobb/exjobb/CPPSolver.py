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
ROBOT_RADIUS = 1
STEP_SIZE = 0.3
UNTRAVERSABLE_THRESHHOLD = 1.5*STEP_SIZE

class CPPSolver:

    def __init__(self, logger, ground_pcd):
        self.name = "General CPP"
        self.motion_planner = MotionPlanner(logger, ground_pcd)
        self.logger = logger
        self.pcd = ground_pcd
    
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
        computational_time = end_time - self.start_time
        coverage = self.pcd.get_coverage_efficiency()
        memory_consumption = get_memory_consumption(snapshot)

        print_text = "\n" + "=" * 20
        print_text += "\nAlgorithm: " + self.name
        print_text += "\nCoverage efficiency: " + str(round(coverage*100, 2)) + "%"
        print_text += "\nNumber of waypoints: " + str(nbr_of_points_in_path)
        print_text += "\nLength of path: " + str(round(length_of_path)) + " meter"
        print_text += "\nTotal rotation: " + str(round(rotation)) + " rad"
        print_text += "\nComputational time: " + str(round(computational_time, 1)) + " sec"
        print_text += "\nMemory consumption: " + str(round(memory_consumption, 1)) + " KiB"
        print_text += "\n" + "=" * 20
        self.logger.info(print_text)

    def new_point_towards(self, start_point, end_point, step_size):
        if np.linalg.norm(end_point - start_point) < step_size:
            return end_point
        direction = self.get_direction_vector(start_point, end_point)
        new_pos = start_point + step_size * direction
        return self.pcd.find_k_nearest(new_pos, 1)[0]   

    def get_direction_vector(self, start, goal):
        line_of_sight = goal - start
        return line_of_sight / np.linalg.norm(line_of_sight)

    def is_valid_step(self, from_point, to_point):
        total_step_size = np.linalg.norm(to_point - from_point)
        
        if total_step_size == 0:
            return False

        if total_step_size <= STEP_SIZE:
            return True

        nbr_of_steps = int(np.floor(total_step_size / STEP_SIZE))
         
        prev_point = from_point
        for step in range(nbr_of_steps):
            end_pos = prev_point + self.get_direction_vector(prev_point, to_point) * STEP_SIZE
            new_point = self.new_point_towards(prev_point, end_pos, STEP_SIZE)

            if np.linalg.norm(new_point - prev_point) > UNTRAVERSABLE_THRESHHOLD:
                return False

            if np.array_equal(new_point, to_point):
                return True

            prev_point = new_point

        return True

    def print(self, object_to_print):
        self.logger.info(str(object_to_print))