from exjobb.Parameters import ROBOT_SIZE
from exjobb.PointCloud import PointCloud
from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx
import timeit
from numba import njit
import operator
import copy 
from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE
from exjobb.MotionPlanner import MotionPlanner
import pickle

CELL_STEP_SIZE = STEP_SIZE #1.25*ROBOT_RADIUS
VISITED_TRESHOLD = 0.5*STEP_SIZE #0.8*ROBOT_RADIUS
COVEREAGE_EFFICIENCY_GOAL = 0.95
RRT_STEP_SIZE = 0.5*STEP_SIZE
RRT_COVEREAGE_EFFICIENCY_GOAL = 0.3
MAX_ITERATIONS = 75
GOAL_CHECK_FREQUENCY = 50


POINTS_FOR_BACKTRACK = 50

DO_BASTAR_PLANNING = True

TRAPPED = 0
ADVANCED = 1
REACHED = 2

class PlanePath():
    def __init__(self, print, motion_planner, starting_point, angle_offset, visited_waypoints):
        self.start = starting_point
        self.pcd = PointCloud(print, points=motion_planner.pcd.points)
        self.motion_planner = motion_planner
        self.path = np.array([starting_point])
        self.visited_waypoints = visited_waypoints
        next_starting_point_close = True
        self.print = print


        self.cover_time = 0
        self.backtrack_time = 0
        self.rest_time = 0
        self.total_time = 0

        #new_starting_point = self.find_closest_wall(starting_point)
        #if new_starting_point is not False:
        #    path_to_starting_point = self.motion_planner.Astar(starting_point, new_starting_point)
        #    self.path = np.append( self.path, path_to_starting_point, axis=0 )
        #    self.pcd.visit_path(self.path, ROBOT_RADIUS)
        #    starting_point = new_starting_point

        #current_position, angle_offset = self.cover_local_area(starting_point)
        
        #print("self.path" + str(self.path))

        total_time = timeit.default_timer()
        
        while next_starting_point_close:
            cover_t = timeit.default_timer()
            current_position, new_path = self.cover_local_area_with_angle(starting_point,angle_offset)
            if len(new_path) == 0:
                self.cover_time += timeit.default_timer() - cover_t
                break

            self.pcd.visit_path(new_path, ROBOT_RADIUS)
            self.cover_time += timeit.default_timer() - cover_t
            

            #print("numbers in path: " + str(len(self.path)))
            backtrack_time = timeit.default_timer()
            backtracking_list = self.get_sorted_backtracking_list(self.path)   
            self.backtrack_time += timeit.default_timer() - backtrack_time 


            rest_time = timeit.default_timer()
            if len(backtracking_list) == 0:
                break

            next_starting_point, backtracking_list = backtracking_list[0], backtracking_list[1:]

            distance_to_next_starting_point = np.linalg.norm(next_starting_point - current_position)
            #print("distance_to_next_starting_point:" + str(distance_to_next_starting_point))

            if distance_to_next_starting_point > 2 or distance_to_next_starting_point == 0:
                next_starting_point_close = False
                break

            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
            self.pcd.visit_path(path_to_next_starting_point, ROBOT_RADIUS)
            self.path = np.append( self.path, path_to_next_starting_point, axis=0 )
            starting_point = next_starting_point
            coverage = self.pcd.get_coverage_efficiency()
            #print("---- coverage: " + str(coverage))
            self.rest_time += timeit.default_timer() - rest_time 

            '''
            backtracking_list = self.get_sorted_backtracking_list(self.path, angle_offset)

            if len(backtracking_list) == 0:
                break   
            #print("backtracking_list" + str(backtracking_list))
            next_starting_point = backtracking_list[0]
            #print(current_position)
            #print(next_starting_point - current_position)
            #print(np.linalg.norm(next_starting_point - current_position))
            if np.array_equal(next_starting_point, current_position) or np.linalg.norm(next_starting_point - current_position) > 3:
                break

            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
            #print(len(path_to_next_starting_point))
            if path_to_next_starting_point is False:
                break

            self.pcd.visit_path(path_to_next_starting_point, ROBOT_RADIUS)
            self.path = np.append( self.path, path_to_next_starting_point, axis=0 )
            #print("visited Astar path")

            #current_position = self.cover_local_area_with_angle(next_starting_point, angle_offset)
            current_position = self.cover_local_area_with_angle(starting_point,angle_offset)
            self.pcd.visit_path(self.path, ROBOT_RADIUS)
            #print("visited cover_local_area_with_angle")
            '''
        self.end = current_position
        self.visited_points_idx = self.pcd.visited_points_idx
        self.coverage = self.pcd.get_coverage_efficiency()
        self.total_time = timeit.default_timer() - total_time
        #print("cover_time: " + str(self.cover_time/self.total_time))
        #print("backtrack_time: " + str(self.backtrack_time/self.total_time))
        #print("rest_time: " + str(self.rest_time/self.total_time))

        self.pcd = None
        self.motion_planner = None
        self.print = None


    def cover_local_area(self, start_point):
        
        
        path_found = False
        
        best_path = np.empty((0,3))
        path_before = self.path
        #self.print("start: " + str(start_point))
        for angle_idx in range(1):
            current_position = start_point
            angle_offset = angle_idx * np.pi/4
            current_path = path_before
            critical_point_found = False
            path_length = 0
            local_path = np.empty((0,3))

            while not critical_point_found:
                critical_point_found = True
                neighbours = self.get_neighbours(current_position, angle_offset)
                for index, neighbour in enumerate(neighbours):
                    #self.print("potential neighbour: " + str(neighbour))
                    if self.is_blocked(current_position, neighbour, current_path):
                        continue

                    #if index in [0,1]:
                    #    east = neighbours[6]
                    #    if not self.is_blocked(current_position, east, current_path):


                    current_position = neighbour
                    
                    current_path = np.append( current_path, [neighbour], axis=0 )
                    path_length += 1
                    critical_point_found  = False
                    #self.print("path: " + str(self.path))
                    break

            new_local_path = current_path [ len(path_before)-1: ]

            if len(best_path) < len(new_local_path):
                best_path = new_local_path
                
        self.pcd.visit_path(best_path , ROBOT_RADIUS)
        self.path = np.append( self.path, best_path, axis=0 )

        current_position = best_path[-1]
        return path_length, current_position
        #path_to_cover_local_area = []

    def get_sorted_backtracking_list(self, path):

        backtracking_list = np.empty((0,3))
        current_position = path[-1]
        
        close_points = np.linalg.norm(path - current_position, axis=1)
        path_close_points = path[ close_points < 2 ]


        if len(path_close_points) == 0:
            return []
        
        for point in path_close_points:
            def b(si, sj):

                if not self.is_blocked(point, si) and self.is_blocked(point, sj):

                    return True
                return False
            neighbours = self.get_neighbours(point)
            s1 = neighbours[6] #east
            s2 = neighbours[2] #northeast
            s3 = neighbours[0] #north
            s4 = neighbours[3] #northwest
            s5 = neighbours[7] #west
            s6 = neighbours[5] #southwest
            s7 = neighbours[1] #south
            s8 = neighbours[4] #southeast
            combinations =  [(s1, s8), (s1,s2), (s5,s6), (s5,s4), (s7,s6), (s7,s8)]
            for c in combinations:
                if b(c[0], c[1]):
                    backtracking_list = np.append( backtracking_list, [point], axis=0)
                    break

        
        
        distances = np.linalg.norm(backtracking_list - current_position, axis=1)
        distances = distances[ distances > 0.1 ]
        sorted_idx = np.argsort(distances)

        return backtracking_list[sorted_idx] #backtracking_list[closest_point_idx]


    def get_sorted_backtracking_list2(self, path, angle_offset):

        backtracking_list = np.empty((0,3))
        current_position = path[-1]
        for point in np.flip(path, 0):

            def b(si, sj):
                if not self.is_blocked(point, si) and self.is_blocked(point, sj):
                    return True
                return False

            neighbours = self.get_neighbours(point, angle_offset)
            s1 = neighbours[6] #east
            s2 = neighbours[2] #northeast
            s3 = neighbours[0] #north
            s4 = neighbours[3] #northwest
            s5 = neighbours[7] #west
            s6 = neighbours[5] #southwest
            s7 = neighbours[1] #south
            s8 = neighbours[4] #southeast

            combinations =  [(s1, s8), (s1,s2), (s5,s6), (s5,s4), (s7,s6), (s7,s8)]
            for c in combinations:
                if b(c[0], c[1]):
                    distance = np.linalg.norm(point - current_position)
                    if distance < 1 and distance > 0.1:
                        return [point]

                    backtracking_list = np.append( backtracking_list, [point], axis=0)
                    
                    break
            
        
        distances = np.linalg.norm(backtracking_list - current_position, axis=1)
        distances = distances[ distances > 0.1 ]
        sorted_idx = np.argsort(distances)

        return backtracking_list[sorted_idx]

    def has_been_visited(self, point, path=None):
        if path is None:
            path = self.path
        
        path = np.append(path, self.visited_waypoints, axis=0)
        distances = np.linalg.norm(path - point, axis=1)
        return np.any(distances <= VISITED_TRESHOLD) 


    def find_closest_wall(self, start_position):

        def is_in_list(list, array):
            diffs =  np.linalg.norm(list - array, axis=1)
            return np.any(diffs < 0.01)

        queue = np.array([start_position])
        visited = np.array([start_position])
        while len(queue):
            current_position, queue = queue[0], queue[1:]
            neighbours = self.get_neighbours(current_position)
            for neighbour in neighbours:
                if not self.motion_planner.is_valid_step(current_position, neighbour):
                    return current_position

                if is_in_list(visited, neighbour):
                    continue
                
                    
                queue = np.append(queue, [neighbour], axis=0)
                visited = np.append(visited, [neighbour], axis=0)

        return False

    def cover_local_area2(self, start_point):        
        best_path = np.empty((0,3))
        path_before = self.path

        for angle_idx in range(8):
            current_position = start_point
            angle_offset = angle_idx * np.pi/8
            current_path = path_before
            critical_point_found = False
            path_length = 0
            local_path = np.empty((0,3))

            while not critical_point_found:
                critical_point_found = True
                neighbours = self.get_neighbours(current_position, angle_offset)

                

                for index, neighbour in enumerate(neighbours):
                    if self.is_blocked(current_position, neighbour, current_path):
                        continue
                    
                    #if index in [0,1]: #If south or north
                    
                    current_position = neighbour
                    
                    current_path = np.append( current_path, [neighbour], axis=0 )
                    path_length += 1
                    critical_point_found  = False
                    break

            new_local_path = current_path [ len(path_before)-1: ]

            if len(best_path) < len(new_local_path):
                best_path = new_local_path
                best_angle_offset = angle_offset
                
        self.path = np.append( self.path, best_path, axis=0 )

        current_position = best_path[-1]
        return current_position, best_angle_offset

    def cover_local_area_with_angle(self, start_point, angle_offset):        
        current_position = start_point
        critical_point_found = False
        
        new_path = np.empty((0,3))

        #def gap_cell(point):
        #    neighours = self.get_neighbours(point, angle_offset)
        #    east_neighbour = neighours[]

        while not critical_point_found:
            critical_point_found = True
            for neighbour in self.get_neighbours(current_position, angle_offset):
                if self.is_blocked(current_position, neighbour, self.path):
                    continue
                current_position = neighbour
                
                self.path = np.append( self.path, [neighbour], axis=0 )
                new_path = np.append( new_path, [neighbour], axis=0 )
                critical_point_found  = False
                break
                
        current_position = self.path[-1]
        return current_position, new_path


    def get_neighbours(self, current_position, angle_offset=np.pi/4):
        directions = []
        for direction_idx in range(8):
            angle = direction_idx/8*np.pi*2 + angle_offset
            x = current_position[0] + np.cos(angle) * CELL_STEP_SIZE
            y = current_position[1] + np.sin(angle) * CELL_STEP_SIZE
            z = current_position[2]
            pos = np.array([x, y, z])
            directions.append(self.pcd.find_k_nearest(pos, 1)[0])
        east, northeast, north, northwest, west, southwest, south, southeast = directions

        #return [north, south, northeast, northwest, southeast, southwest, east, west]
        return [west, northwest, southwest, north, south, northeast, southeast, east]

    def is_blocked(self, from_point, to_point, path = None):
        if path is None:
            path = self.path

        if self.has_been_visited(to_point, path):
            return True
        

        if not self.motion_planner.is_valid_step(from_point, to_point):
            return True
        
        return False

    def get_angle(self, from_pos, to_pos):
        vec = to_pos[0:2] - from_pos[0:2]
        #self.print(vec[0] + vec[1]*1j)
        return np.angle( vec[0] + vec[1]*1j)

