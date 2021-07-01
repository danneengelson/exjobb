from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx
import timeit
from numba import njit

from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE
from exjobb.MotionPlanner import MotionPlanner
CELL_STEP_SIZE = STEP_SIZE #1.25*ROBOT_RADIUS
VISITED_TRESHOLD = 0.66*STEP_SIZE #0.99*ROBOT_RADIUS
COVEREAGE_EFFICIENCY_GOAL = 0.95

TRAPPED = 0
ADVANCED = 1
REACHED = 2

class Spiral(CPPSolver):

    def __init__(self, logger, motion_planner):
        
        self.logger = logger
        super().__init__(logger, motion_planner)
        self.name = "Spiral"


    def get_cpp_path(self, start_point, path=False):
        self.start_tracking()
        coverage = 0
        
        if path is not False:
            self.print(path)
            self.path = np.append(path, [start_point], axis=0)
        else:
            self.path = np.array([start_point])

        self.points_to_mark = np.array([start_point])

        self.move_to(start_point)

        starting_point = start_point
        current_position = start_point
        self.motion_planner = MotionPlanner(self.logger, self.pcd)

        next_starting_point, current_angle = self.find_closest_wall(current_position)
        #self.print("next_starting_point" + str(next_starting_point))
        #self.points_to_mark = np.append(self.points_to_mark, [next_starting_point], axis=0)
        path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
        self.follow_path(path_to_next_starting_point)
        current_position = next_starting_point
        minimum = 5
        
        while coverage < COVEREAGE_EFFICIENCY_GOAL:
            
            next_starting_point, path_point_idx =  self.find_closest_free_pos_2(current_position, self.path)
            if next_starting_point is False:
                break

            path_until_dead_zone, new_current_position = self.get_path_until_dead_zone(next_starting_point, current_angle)

            while len(path_until_dead_zone) < minimum:
                next_starting_point, path_point_idx =  self.find_closest_free_pos_2(current_position, self.path, path_point_idx+1)
                if next_starting_point is False:
                    if minimum == 1:
                        break
                    minimum -= 1
                    self.print("MINSKAR!!!")
                    continue
                path_until_dead_zone, new_current_position = self.get_path_until_dead_zone(next_starting_point, current_angle)

            if next_starting_point is False:
                break

            self.print("Points in path: " + str(len(path_until_dead_zone)))
            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
            if path_to_next_starting_point is False:
                break

            self.follow_path(path_to_next_starting_point)
            self.follow_path(path_until_dead_zone)

            current_position = new_current_position
            current_angle = self.get_angle(self.path[-2], current_position)
            coverage = self.pcd.get_coverage_efficiency()
            self.print("coverage" + str(coverage))
        '''
        while coverage < COVEREAGE_EFFICIENCY_GOAL:
            #break
            #self.print("get_path_until_dead_zone")
            path_until_dead_zone, current_position = self.get_path_until_dead_zone(current_position, current_angle)

            self.print("Points in path: " + str(len(path_until_dead_zone)))
            self.follow_path(path_until_dead_zone)

            next_starting_point, idx =  self.find_closest_free_pos_2(current_position, self.path)

            if next_starting_point is False:
                break

            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)

            if path_to_next_starting_point is False:
                break

            
            self.follow_path(path_to_next_starting_point)
            
            current_position = next_starting_point
            current_angle = self.get_angle(self.path[-2], current_position)
            coverage = self.pcd.get_coverage_efficiency()
            self.print("coverage" + str(coverage))
        '''

        #self.motion_planner.print_times()
        self.print_stats(self.path)
        #self.print(self.path)
        
        return self.path

    def find_closest_free_pos_2(self, start_position, path, ignore_up_to_idx = 0):
        potenital_pos = np.empty((0,3))
        for idx, point in enumerate(np.flip(path, 0)[ignore_up_to_idx:]):
            neighbours = self.get_neighbours(point)
            for neighbour in neighbours:
                if not self.is_blocked(point, neighbour):
                    potenital_pos = np.append(potenital_pos, [neighbour], axis=0)
            if len(potenital_pos):
                closest = self.get_closest_to(start_position, potenital_pos)
                #self.print((closest, idx))
                return closest, ignore_up_to_idx + idx
        return False, False
    
    def find_closest_wall(self, start_position):
        queue = np.array([start_position])
        visited = np.array([start_position])
        while len(queue):
            current_position, queue = queue[0], queue[1:]
            neighbours = self.get_neighbours(current_position)
            for neighbour in neighbours:
                if np.linalg.norm(current_position - neighbour) < 0.5*CELL_STEP_SIZE:
                    current_angle = self.get_angle(start_position, current_position) + np.pi/2
                    return current_position, current_angle
                if self.is_in_list(visited, neighbour):
                    continue
                
                    
                queue = np.append(queue, [neighbour], axis=0)
                visited = np.append(visited, [neighbour], axis=0)

        return False, False
            

    def find_closest_free_pos(self, start_position):
        queue = np.array([start_position])
        visited = np.array([start_position])
        potenital_pos = np.empty((0,3))
        while len(queue):
            current_position, queue = queue[0], queue[1:]
            neighbours = self.get_neighbours(current_position)
            for neighbour in neighbours:
                if self.is_in_list(visited, neighbour):
                    continue

                if not self.is_blocked(current_position, neighbour):
                    potenital_pos = np.append(potenital_pos, [neighbour], axis=0)
                
                if self.motion_planner.is_valid_step(current_position, neighbour):
                    queue = np.append(queue, [neighbour], axis=0)
                
                visited = np.append(visited, [neighbour], axis=0)

            if len(potenital_pos):
                closest = self.get_closest_to(start_position, potenital_pos)
                current_angle = self.get_angle(current_position, closest) #+ np.pi/2
                return closest, current_angle

        self.print("Fail")
        return False, False

    def get_closest_to(self, reference_point, points):
        distances = np.linalg.norm(points - reference_point, axis=1)
        min_idx = np.argmin(distances)
        return points[min_idx]

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

    def get_angle(self, from_pos, to_pos):
        vec = to_pos[0:2] - from_pos[0:2]
        #self.print(vec[0] + vec[1]*1j)
        return np.angle( vec[0] + vec[1]*1j)

    def is_in_list(self, list, array):
        diffs =  np.linalg.norm(list - array, axis=1)
        return np.any(diffs < 0.01)


    def get_neighbours_for_spiral(self, current_position, current_angle):
        directions = []
        for direction_idx in range(8):
            angle = direction_idx/8*np.pi*2 + current_angle
            x = current_position[0] + np.cos(angle) * CELL_STEP_SIZE
            y = current_position[1] + np.sin(angle) * CELL_STEP_SIZE
            z = current_position[2]
            pos = np.array([x, y, z])
            directions.append(self.pcd.find_k_nearest(pos, 1)[0])
        
        right, forwardright, forward, forwardleft, left, backleft, back, backright = directions
        return [backright, right, forwardright, forward, forwardleft, left, backleft]

    def get_neighbours(self, current_position, angle_offset=0):
        directions = []
        for direction_idx in range(8):
            angle = direction_idx/8*np.pi*2 + angle_offset
            x = current_position[0] + np.cos(angle) * CELL_STEP_SIZE
            y = current_position[1] + np.sin(angle) * CELL_STEP_SIZE
            z = current_position[2]
            pos = np.array([x, y, z])
            directions.append(self.pcd.find_k_nearest(pos, 1)[0])

        east, northeast, north, northwest, west, southwest, south, southeast = directions

        return [north, south, northeast, northwest, southeast, southwest, east, west]

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