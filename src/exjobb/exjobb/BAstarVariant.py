import numpy as np
import csv

from exjobb.CPPSolver import CPPSolver 
from exjobb.Parameters import ROBOT_SIZE, CURVED_BASTAR_STEP_SIZE, CURVED_BASTAR_VISITED_TRESHOLD, COVEREAGE_EFFICIENCY_GOAL

class BAstarVariant(CPPSolver):
    ''' Solving the Coverage Path Planning Problem with a variation of BAstar
    '''
    def __init__(self, print, motion_planner, coverable_pcd, time_limit = None, parameters = None):
        '''
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        '''
        super().__init__(print, motion_planner, coverable_pcd, time_limit)
        self.name = "BAstar Variant"

        if parameters is None:
            self.step_size = CURVED_BASTAR_STEP_SIZE
            self.visited_threshold = CURVED_BASTAR_VISITED_TRESHOLD
        else:
            self.step_size = parameters["step_size"] * ROBOT_SIZE
            self.visited_threshold =  parameters["visited_threshold"] * self.step_size

    def get_cpp_path(self, start_point, angle_offset=0, goal_coverage=None):
        """Generates a path that covers the area using BAstar Variant Algorithm.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot
            angle_offset (optional): Angle in radians of the main direction of the paths.

        Returns:
            Nx3 array with waypoints
        """
        if goal_coverage is None:
            goal_coverage = COVEREAGE_EFFICIENCY_GOAL
        
        self.start_tracking()
        coverage = 0
        self.move_to(start_point)

        current_position = start_point

        starting_point, _ = self.find_closest_wall(start_point, self.step_size)

        if starting_point is False:
            starting_point = current_position
        else:
            path_to_starting_point = self.motion_planner.Astar(current_position, starting_point)
            self.follow_path(path_to_starting_point)

        next_starting_point_not_found = False

        while coverage < goal_coverage and not self.time_limit_reached():
            
            path_to_cover_local_area, current_position = self.get_path_to_cover_local_area(starting_point, angle_offset)

            if len(path_to_cover_local_area) == 0:
                self.print("No path found when covering local area!")  

            self.follow_path(path_to_cover_local_area)   
            
            next_starting_point = self.get_next_starting_point(self.path, angle_offset)     
            if next_starting_point is False:
                self.print("No next_starting_point found")
                break

        

            #next_starting_point, evaluated_points =  self.get_next_starting_point_wavefront(current_position)
            #visited = np.empty((0,3))
            #while next_starting_point is False:
            #    self.print("No next_starting_point")
            #    if len(evaluated_points) == 1:
            #        visited = np.append(visited, [current_position], axis=0)
            #        current_position = self.step_back()
            #        next_starting_point, evaluated_points =  self.get_next_starting_point_wavefront(current_position, visited)
            #    else:
            #        next_starting_point_not_found = True
            #        break
            #
            #if next_starting_point_not_found:
            #    break
            
            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)

            while path_to_next_starting_point is False:
                current_position = self.step_back()
                path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)                

            self.follow_path(path_to_next_starting_point)
            
            starting_point = next_starting_point
            
            coverage = self.coverable_pcd.get_coverage_efficiency()
            self.save_sample_for_results(coverage)
            self.print_update(coverage)        
        #self.print_stats(self.path)
        
        return self.path


    def get_path_to_cover_local_area(self, start_point, angle_offset = 0):
        """Generates BAstar paths to cover local area.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot
            angle_offset (optional): Angle in radians of the main direction of the paths.

        Returns:
            Generated path with waypoints and the current position of the robot at the end
        """
       
        path_before = self.path

        current_position = start_point
        current_full_path = path_before
        critical_point_found = False
        local_path = np.empty((0,3))

        while not critical_point_found:
            critical_point_found = True
            neighbours = self.get_neighbours_for_bastar_variant(current_position, angle_offset)
            for neighbour in neighbours:
                
                if self.is_blocked(current_position, neighbour, self.visited_threshold, current_full_path):
                    continue

                current_position = neighbour
                
                current_full_path = np.append( current_full_path, [neighbour], axis=0 )
                local_path = np.append( local_path, [neighbour], axis=0 )

                critical_point_found  = False

                break

        return local_path, current_position
        

    def get_next_starting_point(self, path, angle_offset = 0, lower_criteria = False):
        """Finds the next starting point by creating a backtrack list of possible points
        and choose the closest one.

        Args:
            path: Waypoint with the path that has been made so far in a Nx3 array
            angle_offset (optional): Angle in radians of the main direction of the paths.

        Returns:
            A position with an obstacle free uncovered point.
        """

        current_position = path[-1]
        distances = np.linalg.norm(path - current_position, axis=1)
        sorted_path_by_distance = path[np.argsort(distances)]
        
        for point in sorted_path_by_distance:
            def b(si, sj):
                                
                if not self.is_blocked(point, si, self.visited_threshold) and self.is_blocked(point, sj, self.visited_threshold):
                    return True

                if lower_criteria and not self.is_blocked(point, si, self.visited_threshold):
                    return True

                return False

            neighbours = self.get_neighbours(point, self.step_size, angle_offset)
            
            s1 = neighbours[0] #east
            s2 = neighbours[1] #northeast
            s3 = neighbours[2] #north
            s4 = neighbours[3] #northwest
            s5 = neighbours[4] #west
            s6 = neighbours[5] #southwest
            s7 = neighbours[6] #south
            s8 = neighbours[7] #southeast

            combinations =  [(s1, s8), (s1,s2), (s5,s6), (s5,s4), (s7,s6), (s7,s8)]
            for c in combinations:
                if b(c[0], c[1]):
                    return point

        if not lower_criteria:
            self.print("WARNING: Lowered criteria")
            return self.get_next_starting_point(path, angle_offset, lower_criteria=True)
        return False

    def get_next_starting_point_wavefront(self, start_position, previously_visited = None,  max_distance = False):
        """Using Wavefront algorithm to find the closest obstacle free uncovered position.

        Args:
            start_position: A [x,y,z] np.array of the start position of the search

        Returns:
            An obstacle free uncovered position.
        """
        last_layer = np.array([start_position])
        visited = np.array([start_position])
        if previously_visited is not None:
            visited = np.append(visited, previously_visited, axis=0)

        while len(last_layer):
            new_layer = np.empty((0,3))
            for pos in last_layer:
                neighbours = self.get_neighbours(pos, self.step_size)
                #self.print("neighbours" + str(neighbours))
                for neighbour in neighbours:

                    if max_distance and np.linalg.norm(start_position - neighbour) > max_distance:
                        continue

                    if self.has_been_visited(neighbour, self.visited_threshold, visited):
                        #self.print("visited")
                        continue                    

                    if not self.motion_planner.is_valid_step(pos, neighbour):
                        #self.print("invalid steo")
                        continue

                    if not self.has_been_visited(neighbour, self.visited_threshold) and self.accessible(neighbour, self.path):
                        
                        return neighbour, visited
                    #self.print("unvisited lets go")
                    visited = np.append(visited, [neighbour], axis=0)
                    new_layer = np.append(new_layer, [neighbour], axis=0)

            last_layer = new_layer

        self.print("FAIL. No new uncovered obstacle free positions could be found from " + str(start_position))
        self.points_to_mark = [start_position]
        return False, visited


    def get_neighbours_for_bastar_variant(self, current_position, angle_offset = 0):
        """Finds all neighbours of a given position and return them in the order
        to create the bastar zig-zag motion.

        Args:
            current_position: A [x,y,z] np.array of the start position 
            angle_offset: Angle offset in radians

        Returns:
            All 8 neighbours of the given position in order:
            north, south, northeast, northwest, southeast, southwest, east, west
        """

        east, northeast, north, northwest, west, southwest, south, southeast = self.get_neighbours(current_position, self.step_size, angle_offset)
        return [west, northwest, southwest, north, south, northeast, southeast, east]
