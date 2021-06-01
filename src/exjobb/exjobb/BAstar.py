from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx
import timeit
from numba import njit

from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE
from exjobb.MotionPlanner import MotionPlanner
CELL_STEP_SIZE = 1.25*ROBOT_RADIUS
VISITED_TRESHOLD = 0.8*ROBOT_RADIUS
COVEREAGE_EFFICIENCY_GOAL = 0.95
MAX_ITERATIONS = 10000

TRAPPED = 0
ADVANCED = 1
REACHED = 2

class BAstar(CPPSolver):

    def __init__(self, logger, motion_planner):
        
        self.logger = logger
        super().__init__(logger, motion_planner)
        self.name = "BAstar"

    def get_cpp_path(self, start_point):
        self.start_tracking()
        coverage = 0
        self.path = np.array([start_point])
        self.starting_point_list = np.array([start_point])


        self.pcd.visit_point(start_point, ROBOT_RADIUS)


        starting_point = start_point
        current_position = start_point
        self.motion_planner = MotionPlanner(self.logger, self.pcd)
        self.cover_local_area_time = 0
        self.backtracking_time = 0
        self.motion_planner_time = 0
        self.rest_time = 0
        self.b_time = 0
        self.sort_time = 0
        self.visited_time = 0
        self.valid_time = 0
        self.next_starting_point_time = 0
        break_loop = False
        while coverage < COVEREAGE_EFFICIENCY_GOAL:
            
            cover_local_area = timeit.default_timer()
            path_length, current_position = self.cover_local_area(starting_point)
            self.cover_local_area_time += timeit.default_timer() - cover_local_area

            if path_length == 0:
                self.print("No path found when covering local area!")
                #break           
            
            

            backtracking = timeit.default_timer()
            backtracking_list = self.get_sorted_backtracking_list(self.path)     
            self.backtracking_time += timeit.default_timer() - backtracking

            if len(backtracking_list) == 0:
                break

            next_starting_point_time = timeit.default_timer()
            next_starting_point, backtracking_list = backtracking_list[0], backtracking_list[1:]
            #next_starting_point, next_starting_point_idx, visiting_rate  = self.get_next_starting_point(backtracking_list)
            self.next_starting_point_time += timeit.default_timer() - next_starting_point_time

            motion_planner = timeit.default_timer()
            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
            self.motion_planner_time += timeit.default_timer() - motion_planner
            
            
            #if path_to_next_starting_point is False:
                #go back
                #break
                #current_position = self.path[-2]
            
            while path_to_next_starting_point is False:
                self.print("No path found when planning Astar!")
                self.starting_point_list = np.append(self.starting_point_list, [current_position], axis=0 )
                self.starting_point_list = np.append(self.starting_point_list, [next_starting_point], axis=0 )
                self.print("backtracking_list: " + str(backtracking_list))
                backtracking_list = np.delete(backtracking_list, next_starting_point_idx, axis=0)
                if len(backtracking_list) == 0:
                    break_loop = True
                    break

                self.print("backtracking_list: " + str(backtracking_list))
                self.print("previous: " + str(next_starting_point))
                next_starting_point, next_starting_point_idx, visiting_rate = self.get_next_starting_point(backtracking_list, visiting_rate)
                self.print("new: " + str(next_starting_point))
                path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
                
            if break_loop:
                break
            
            rest = timeit.default_timer()
            self.path = np.append( self.path, path_to_next_starting_point, axis=0 )
            self.pcd.visit_path(path_to_next_starting_point, ROBOT_RADIUS)
            #path_to_next_starting_point = [next_starting_point]
            

            #self.starting_point_list = np.append(self.starting_point_list, [self.path[-1]], axis=0 )
            
            #self.print("next_starting_point" + str(next_starting_point))
            #self.print("path" + str(self.path))
            
            starting_point = next_starting_point
            
            #self.starting_point_list = np.append(self.starting_point_list, [next_starting_point], axis=0 )
            coverage = self.pcd.get_coverage_efficiency()
            self.print("coverage" + str(coverage))
            self.rest_time += timeit.default_timer() - rest
            #if new_coverage - coverage < 0.01 and len(backtracking_list) > 0:
            #    self.print("Too small area")
            #    self.path = self.path[0 : current_path_length]
            #    self.pcd.visited_points_idx = self.pcd.visited_points_idx[0 : current_path_length]
            #    starting_point, backtracking_list = backtracking_list[0], backtracking_list[1:]
            #    continue
            
            

            #if backtracking_list and path_to_cover_local_area:
            #    critical_point = path_to_cover_local_area[-1]
            #    next_starting_point = self.get_next_starting_point(backtracking_list)
            #    path_to_next_starting_point = self.find_path(critical_point, next_starting_point)
            #    path = np.append( path, path_to_next_starting_point )
            
        #self.print("path: " + str(self.path))
        self.print("cover_local_area_time" + str(self.cover_local_area_time))
        self.print("motion_planner_time" + str(self.motion_planner_time))
        self.print("backtracking_time" + str(self.backtracking_time))
        self.print("rest_time" + str(self.rest_time))
        self.print("sort_time" + str(self.sort_time))
        self.print("b_time" + str(self.b_time)) 
        self.print("visited_time" + str(self.visited_time))
        self.print("valid_time" + str(self.valid_time))
        self.print("next_starting_point_time: " + str(self.next_starting_point_time))
        self.motion_planner.print_times()
        
        
        self.print_stats(self.path)
        
        return self.path

    def get_next_starting_point(self, sorted_backtracking_list, visiting_rate_start = 0.7):
        next_starting_point = False
        visiting_rate = visiting_rate_start
        while next_starting_point is False:
            visiting_rate += 0.05
            for idx, potential_starting_point in enumerate(sorted_backtracking_list):
                visiting_rate_in_area = self.pcd.get_visiting_rate_in_area(potential_starting_point, 2*ROBOT_RADIUS)
                if visiting_rate_in_area < visiting_rate:
                    next_starting_point = potential_starting_point
                    break
        return next_starting_point, idx, visiting_rate



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
                for neighbour in self.get_neighbours(current_position, angle_offset):
                    #self.print("potential neighbour: " + str(neighbour))
                    if self.is_blocked(current_position, neighbour, current_path):
                        continue
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
        #self.print("Backtracking_list 1: " + str(backtracking_list))
        
        for point in path:
            #self.print("Backtracking_list 2: " + str(backtracking_list))
            def b(si, sj):
                b_t = timeit.default_timer()
                #self.print("si, sj: " + str((si, sj)))
                #self.print("si valid: " + str(self.is_valid_step(point, si)))
                #self.print("sj valid: " + str(self.is_valid_step(point, sj)))
                if not self.is_blocked(point, si) and self.is_blocked(point, sj):
                    #self.print("point: " + str(point))
                    #self.print("not bloacked: " + str(si))
                    #self.print("bloacked: " + str(sj))
                    self.b_time += timeit.default_timer() - b_t
                    return True
                #self.print("Return 0")
                self.b_time += timeit.default_timer() - b_t
                return False
            sort = timeit.default_timer()
            neighbours = self.get_neighbours(point)
            s1 = neighbours[6] #east
            s2 = neighbours[2] #northeast
            s3 = neighbours[0] #north
            s4 = neighbours[3] #northwest
            s5 = neighbours[7] #west
            s6 = neighbours[5] #southwest
            s7 = neighbours[1] #south
            s8 = neighbours[4] #southeast
            self.sort_time += timeit.default_timer() - sort
            combinations =  [(s1, s8), (s1,s2), (s5,s6), (s5,s4), (s7,s6), (s7,s8)]
            for c in combinations:
                if b(c[0], c[1]):
                    backtracking_list = np.append( backtracking_list, [point], axis=0)
                    break
            #my = b(s1, s8) + b(s1,s2) + b(s5,s6) + b(s5,s4) + b(s7,s6) + b(s7,s8)
            ##self.print("my" + str(my))
            #if my >= 1:
            #    backtracking_list = np.append( backtracking_list, [point], axis=0)
                #self.print("Backtracking_list 3: " + str(backtracking_list))

        #self.print("Length of backtracking_list: " + str(backtracking_list))

        current_position = path[-1]
        
        distances = np.linalg.norm(backtracking_list - current_position, axis=1)
        distances = distances[ distances > 0.1 ]
        sorted_idx = np.argsort(distances)
        
        #closest_point_idx = np.argmin(distances)
        #self.backtrack_list = backtracking_list

        return backtracking_list[sorted_idx] #backtracking_list[closest_point_idx]


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
        '''
        east = self.motion_planner.new_point_towards(current_position, current_position + directions, CELL_STEP_SIZE)
        northeast = self.motion_planner.new_point_towards(current_position, current_position +np.array([100,100,0]), CELL_STEP_SIZE)
        north = self.motion_planner.new_point_towards(current_position, current_position +np.array([0,100,0]), CELL_STEP_SIZE)
        northwest = self.motion_planner.new_point_towards(current_position,current_position + np.array([-100,100,0]), CELL_STEP_SIZE)
        west = self.motion_planner.new_point_towards(current_position, current_position +np.array([-100,0,0]), CELL_STEP_SIZE)
        southwest = self.motion_planner.new_point_towards(current_position, current_position +np.array([-100,-100,0]), CELL_STEP_SIZE)
        south = self.motion_planner.new_point_towards(current_position,current_position + np.array([0,-100,0]), CELL_STEP_SIZE)
        southeast = self.motion_planner.new_point_towards(current_position,current_position + np.array([100,-100,0]), CELL_STEP_SIZE)
        '''
        #return [north, east, south, west]
        return [north, south, northeast, northwest, southeast, southwest, east, west]

    def has_been_visited(self, point, path=None):
        if path is None:
            path = self.path
        #self.print("path: " + str(self.path) )
        #self.print("self.path - point" + str(self.path - point))

        distances = np.linalg.norm(path - point, axis=1)
        #self.print("distances" + str(distances))
        #self.print(np.any(distances <= STEP_SIZE) )
        return np.any(distances <= VISITED_TRESHOLD) 

    def is_blocked(self, from_point, to_point, path = None):
        if path is None:
            path = self.path

        visited = timeit.default_timer()
        if self.has_been_visited(to_point, path):
            self.visited_time += timeit.default_timer() - visited
            return True
        self.visited_time += timeit.default_timer() - visited
        

        valid = timeit.default_timer()
        if not self.motion_planner.is_valid_step(from_point, to_point):
            self.valid_time += timeit.default_timer() - valid
            return True
        self.valid_time += timeit.default_timer() - valid
        
        return False

    def get_points_to_mark(self):
        #return self.backtrack_list
        return self.starting_point_list