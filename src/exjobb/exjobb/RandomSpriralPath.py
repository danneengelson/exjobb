from exjobb.Parameters import ROBOT_SIZE
from exjobb.PointCloud import PointCloud
from exjobb.Tree import Tree
import numpy as np
import networkx as nx
import timeit
import operator
import copy 
from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE
from exjobb.MotionPlanner import MotionPlanner
import pickle
CELL_STEP_SIZE = STEP_SIZE #1.25*ROBOT_RADIUS
VISITED_TRESHOLD = 0.66*STEP_SIZE #0.99*ROBOT_RADIUS
COVEREAGE_EFFICIENCY_GOAL = 0.95


class SpiralPath:
    def __init__(self, print, motion_planner, starting_point, visited_points):
        self.path = visited_points
        self.start = starting_point
        self.pcd = PointCloud(print, points=motion_planner.pcd.points)
        self.motion_planner = motion_planner
        current_angle = 0

        self.path, self.end = self.get_path_until_dead_zone(starting_point, current_angle)

        if len(self.path) > 1:
            self.pcd.visit_path(self.path, ROBOT_RADIUS)
            self.visited_points_idx = self.pcd.visited_points_idx

        self.coverage = self.pcd.get_coverage_efficiency()

        self.pcd = None
        self.motion_planner = None
        
            
    def get_path_until_dead_zone(self, current_position, current_angle):
        path = np.array([current_position])
        dead_zone_reached = False
        while not dead_zone_reached:
            dead_zone_reached =  True
            neighbours = self.get_neighbours_for_spiral(current_position, current_angle)

            for idx, neighbour in enumerate(neighbours): 
                current_path = np.append(self.path, path, axis=0)
                #self.print(current_path)
                if self.is_blocked(current_position, neighbour, current_path) :
                    continue

                path = np.append(path, [neighbour], axis=0)
                current_angle = self.get_angle(current_position, neighbour)
                current_position = neighbour
                
                dead_zone_reached = False
                break

        return path, current_position

    def get_neighbours_for_spiral(self, current_position, current_angle):
        directions = []
        for direction_idx in range(8):
            angle = direction_idx/8*np.pi*2 + current_angle
            x = current_position[0] + np.cos(angle) * STEP_SIZE
            y = current_position[1] + np.sin(angle) * STEP_SIZE
            z = current_position[2]
            pos = np.array([x, y, z])
            directions.append(self.pcd.find_k_nearest(pos, 1)[0])
        
        right, forwardright, forward, forwardleft, left, backleft, back, backright = directions
        return [backright, right, forwardright, forward, forwardleft, left, backleft]

    def get_angle(self, from_pos, to_pos):
        vec = to_pos[0:2] - from_pos[0:2]
        #self.print(vec[0] + vec[1]*1j)
        return np.angle( vec[0] + vec[1]*1j)

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