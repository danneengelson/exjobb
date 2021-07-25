import numpy as np
import timeit

from exjobb.CPPSolver import CPPSolver 
from exjobb.Parameters import ROBOT_RADIUS, BASTAR_STEP_SIZE, BASTAR_VISITED_TRESHOLD, COVEREAGE_EFFICIENCY_GOAL

class BAstar(CPPSolver):
    ''' Solving the Coverage Path Planning Problem with BAstar
    '''
    def __init__(self, print, motion_planner):
        '''
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        '''
        super().__init__(print, motion_planner)
        self.name = "BAstar"

    def get_cpp_path(self, start_point):
        """Generates a path that covers the area using BAstar Algorithm.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot

        Returns:
            Nx3 array with waypoints
        """
        
        self.start_tracking()
        coverage = 0
        self.move_to(start_point)

        current_position = start_point

        starting_point = self.find_closest_wall(start_point)

        if starting_point is False:
            starting_point = current_position
        else:
            path_to_starting_point = self.motion_planner.Astar(current_position, starting_point)
            self.follow_path(path_to_starting_point)

        while coverage < COVEREAGE_EFFICIENCY_GOAL:
            
            path_length, current_position = self.cover_local_area(starting_point)

            if path_length == 0:
                self.print("No path found when covering local area!")          
            
            backtracking_list = self.get_sorted_backtracking_list(self.path)     

            if len(backtracking_list) == 0:
                break

            next_starting_point = backtracking_list[0]

            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)

            self.follow_path(path_to_next_starting_point)
            
            starting_point = next_starting_point
            
            coverage = self.pcd.get_coverage_efficiency()
            self.print("coverage" + str(coverage))
        
        self.print_stats(self.path)
        
        return self.path

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



    def get_path_to_cover_local_area(self, start_point):
        """Generates BAstar paths to cover local area.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot

        Returns:
            [type]: [description]
        """
       
        path_before = self.path

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
                
        self.pcd.visit_path(best_path)
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
                #self.print("si, sj: " + str((si, sj)))
                #self.print("si valid: " + str(self.is_valid_step(point, si)))
                #self.print("sj valid: " + str(self.is_valid_step(point, sj)))
                if not self.is_blocked(point, si) and self.is_blocked(point, sj):
                    #self.print("point: " + str(point))
                    #self.print("not bloacked: " + str(si))
                    #self.print("bloacked: " + str(sj))
                    return True
                #self.print("Return 0")
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
            x = current_position[0] + np.cos(angle) * BASTAR_STEP_SIZE
            y = current_position[1] + np.sin(angle) * BASTAR_STEP_SIZE
            z = current_position[2]
            pos = np.array([x, y, z])
            directions.append(self.pcd.find_k_nearest(pos, 1)[0])
        east, northeast, north, northwest, west, southwest, south, southeast = directions

        return [east, north, south, northeast, northwest, southeast, southwest, west]
        #return [west, northwest, southwest, north, south, northeast, southeast, east]

    def has_been_visited(self, point, path=None):
        if path is None:
            path = self.path
        #self.print("path: " + str(self.path) )
        #self.print("self.path - point" + str(self.path - point))

        distances = np.linalg.norm(path - point, axis=1)
        #self.print("distances" + str(distances))
        #self.print(np.any(distances <= STEP_SIZE) )
        return np.any(distances <= BASTAR_VISITED_TRESHOLD) 

    def is_blocked(self, from_point, to_point, path = None):
        if path is None:
            path = self.path

        if self.has_been_visited(to_point, path):
            return True
        

        if not self.motion_planner.is_valid_step(from_point, to_point):
            return True
        
        return False
