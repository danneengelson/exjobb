from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx
import timeit

from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE
from exjobb.MotionPlanner import MotionPlanner
CELL_STEP_SIZE = 1.25*ROBOT_RADIUS
VISITED_TRESHOLD = 0.8*ROBOT_RADIUS
COVEREAGE_EFFICIENCY_GOAL = 0.3
MAX_ITERATIONS = 10000
GOAL_CHECK_FREQUENCY = 50

TRAPPED = 0
ADVANCED = 1
REACHED = 2

class BAstar(CPPSolver):

    def __init__(self, logger, ground_pcd):
        self.name = "BAstar"
        self.logger = logger
        super().__init__(logger, ground_pcd)

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
            #next_starting_point, backtracking_list = backtracking_list[0], backtracking_list[1:]
            next_starting_point = False
            visiting_rate = 0.5
            while next_starting_point is False:
                visiting_rate += 0.05
                for potential_starting_point in backtracking_list:
                    visiting_rate_in_area = self.pcd.get_visiting_rate_in_area(potential_starting_point, 2*ROBOT_RADIUS)
                    #self.print(visiting_rate_in_area)
                    if visiting_rate_in_area < visiting_rate:
                        next_starting_point = potential_starting_point
                        break

            motion_planner = timeit.default_timer()
            path_to_next_starting_point = self.motion_planner.RRT(current_position, next_starting_point)
            self.motion_planner_time += timeit.default_timer() - motion_planner
            
            #self.print(path_to_next_starting_point)
            rest = timeit.default_timer()
            if path_to_next_starting_point is False:
                path_to_next_starting_point = []
                self.print("No path found when planning Astar!")
                self.starting_point_list = np.append(self.starting_point_list, [current_position], axis=0 )
                self.starting_point_list = np.append(self.starting_point_list, [next_starting_point], axis=0 )
                break

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
            
        self.print("path: " + str(self.path))
        self.print("cover_local_area_time" + str(self.cover_local_area_time))
        self.print("motion_planner_time" + str(self.motion_planner_time))
        self.print("backtracking_time" + str(self.backtracking_time))
        self.print("rest_time" + str(self.rest_time))
        self.print("sort_time" + str(self.sort_time))
        self.print("b_time" + str(self.b_time)) 
        self.print("visited_time" + str(self.visited_time))
        self.print("valid_time" + str(self.valid_time))

        self.print_stats(self.path)
        return self.path

    def cover_local_area(self, start_point):
        current_position = start_point
        critical_point_found = False
        path_found = False
        path_length = 0
        #self.print("start: " + str(start_point))
        while not critical_point_found:
            critical_point_found = True
            for neighbour in self.get_neighbours(current_position):
                #self.print("potential neighbour: " + str(neighbour))
                if self.is_blocked(current_position, neighbour):
                    continue
                
                #self.print("VALID")
                current_position = neighbour
                self.pcd.visit_point(neighbour, ROBOT_RADIUS)
                self.path = np.append( self.path, [neighbour], axis=0 )
                path_length += 1
                critical_point_found  = False
                #self.print("path: " + str(self.path))
                break

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
            s1 = self.new_point_towards(point, point + np.array([100,0,0]), CELL_STEP_SIZE)
            s2 = self.new_point_towards(point, point + np.array([100,100,0]), CELL_STEP_SIZE)
            #s3 = self.new_point_towards(point,point +  np.array([0,100,0]), CELL_STEP_SIZE)
            s4 = self.new_point_towards(point, point + np.array([-100,100,0]), CELL_STEP_SIZE)
            s5 = self.new_point_towards(point, point + np.array([-100,0,0]), CELL_STEP_SIZE)
            s6 = self.new_point_towards(point, point + np.array([-100,-100,0]), CELL_STEP_SIZE)
            s7 = self.new_point_towards(point, point + np.array([0,-100,0]), CELL_STEP_SIZE)
            s8 = self.new_point_towards(point, point + np.array([100,-100,0]), CELL_STEP_SIZE)
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
        sorted_idx = np.argsort(distances)
        
        #closest_point_idx = np.argmin(distances)
        #self.backtrack_list = backtracking_list

        return backtracking_list[sorted_idx] #backtracking_list[closest_point_idx]


    def get_neighbours(self, current_position):
        #north = self.new_point_towards(current_position, np.array([0,100,0]), CELL_STEP_SIZE)
        #south = self.new_point_towards(current_position, np.array([0,-100,0]), CELL_STEP_SIZE)
        #east = self.new_point_towards(current_position, np.array([100,0,0]), CELL_STEP_SIZE)
        #west = self.new_point_towards(current_position, np.array([-100,0,0]), CELL_STEP_SIZE)

        east = self.new_point_towards(current_position, current_position + np.array([100,0,0]), CELL_STEP_SIZE)
        northeast = self.new_point_towards(current_position, current_position +np.array([100,100,0]), CELL_STEP_SIZE)
        north = self.new_point_towards(current_position, current_position +np.array([0,100,0]), CELL_STEP_SIZE)
        northwest = self.new_point_towards(current_position,current_position + np.array([-100,100,0]), CELL_STEP_SIZE)
        west = self.new_point_towards(current_position, current_position +np.array([-100,0,0]), CELL_STEP_SIZE)
        southwest = self.new_point_towards(current_position, current_position +np.array([-100,-100,0]), CELL_STEP_SIZE)
        south = self.new_point_towards(current_position,current_position + np.array([0,-100,0]), CELL_STEP_SIZE)
        southeast = self.new_point_towards(current_position,current_position + np.array([100,-100,0]), CELL_STEP_SIZE)
        #return [north, east, south, west]
        return [north, south, northeast, northwest, southeast, southwest, east, west]

    def has_been_visited(self, point):
        #self.print("path: " + str(self.path) )
        #self.print("self.path - point" + str(self.path - point))

        distances = np.linalg.norm(self.path - point, axis=1)
        #self.print("distances" + str(distances))
        #self.print(np.any(distances <= STEP_SIZE) )
        return np.any(distances <= VISITED_TRESHOLD) 

    def is_blocked(self, from_point, to_point):
        visited = timeit.default_timer()
        if self.has_been_visited(to_point):
            self.visited_time += timeit.default_timer() - visited
            return True
        self.visited_time += timeit.default_timer() - visited
        

        valid = timeit.default_timer()
        if not self.is_valid_step(from_point, to_point):
            self.valid_time += timeit.default_timer() - valid
            return True
        self.valid_time += timeit.default_timer() - valid
        
        return False

    def get_points_to_mark(self):
        #return self.backtrack_list
        return self.starting_point_list