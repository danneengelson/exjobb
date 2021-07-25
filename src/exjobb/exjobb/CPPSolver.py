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
from exjobb.Parameters import ROBOT_SIZE, ROBOT_RADIUS

ROBOT_RADIUS = ROBOT_SIZE/2
STEP_SIZE = ROBOT_SIZE
UNTRAVERSABLE_THRESHHOLD = 1.5*STEP_SIZE

class CPPSolver:
    ''' Abstract class of a Coverage Path Problem Solver
    '''

    def __init__(self, print, motion_planner):
        '''
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        '''
        self.name = "General CPP"
        self.print = print
        self.pcd = motion_planner.traversable_pcd
        self.motion_planner = motion_planner
        self.current_position = None
        self.path = np.empty((0,3))    
        self.points_to_mark = np.empty((0,3))

    def start_tracking(self):
        ''' Start the tracking of computational time and memory consumption
        '''
        tracemalloc.start()
        self.start_time = timeit.default_timer()

    def print_stats(self, path):
        ''' Prints stats about the generated path
        Args:
            path: A Nx3 array with waypoints
        '''
        end_time = timeit.default_timer()
        snapshot = tracemalloc.take_snapshot()
        nbr_of_points_in_path = len(path)

        def get_memory_consumption(snapshot, key_type='lineno'):
            ''' Calculates memory consumption of the algorithm in KiB
            '''
            snapshot = snapshot.filter_traces((
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            ))
            top_stats = snapshot.statistics(key_type)
            total = sum(stat.size for stat in top_stats)
            return total / 1024
        
        def get_length_of_path(path):
            ''' Calculates length of the path in meters
            '''
            length = 0
            for point_idx in range(len(path) - 1):
                length += np.linalg.norm( path[point_idx] - path[point_idx + 1] )
            return length

        def get_total_rotation(path):
            ''' Calculates the total rotation made by the robot while executing the path
            '''
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
        unessecary_coverage_mean = self.pcd.get_coverage_count_per_point(path)
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
        self.print(print_text)


    def follow_path(self, path):
        ''' Makes the robot follow a path. Marks points along the way as visited.
        Args:
            path: A Nx3 array with waypoints
        '''
        if len(path) > 0 and len(self.path) > 0:
            if np.array_equal(self.path[-1], path[0]):
                path = path[1:]

            if len(path):
                self.path = np.append( self.path, path, axis=0 )
                self.pcd.visit_path(path)

    def move_to(self, position):
        ''' Makes the robot go to a specific position. Marks points along the way as visited.
        Args:
            position: A [x,y,z] array with the position
        '''
        if len(self.path) > 0:
            curr_position = self.path[-1]
            self.pcd.visit_path_to_position(position, curr_position)
        else:
            self.pcd.visit_position(position)
            
        self.path = np.append( self.path, [position], axis=0 )
        