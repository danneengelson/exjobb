import numpy as np

from exjobb.CPPSolver import CPPSolver 
from exjobb.Parameters import BASTAR_VARIANT_DISTANCE, BASTAR_STEP_SIZE, BASTAR_VISITED_TRESHOLD, COVEREAGE_EFFICIENCY_GOAL

class BAstarVariant(CPPSolver):
    ''' Solving the Coverage Path Planning Problem with a variation of BAstar
    '''
    def __init__(self, print, motion_planner):
        '''
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        '''
        super().__init__(print, motion_planner)
        self.name = "BAstar"

    def get_cpp_path(self, start_point, angle_offset=0):
        """Generates a path that covers the area using BAstar Algorithm.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot
            angle_offset (optional): Angle in radians of the main direction of the paths.

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
            
            path_to_cover_local_area, current_position = self.get_path_to_cover_local_area(starting_point, angle_offset)

            if len(path_to_cover_local_area) == 0:
                self.print("No path found when covering local area!")  

            self.follow_path(path_to_cover_local_area)        
            
            next_starting_point = self.get_next_starting_point_fast(self.path)

            if next_starting_point is False:
                next_starting_point = self.get_next_starting_point(self.path, angle_offset) 

            if next_starting_point is False:
                self.print("No next_starting_point found")
                break

            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)

            self.follow_path(path_to_next_starting_point)
            
            starting_point = next_starting_point
            
            coverage = self.pcd.get_coverage_efficiency()
            self.print("coverage" + str(coverage))
        
        self.print_stats(self.path)
        
        return self.path

    def find_closest_wall(self, start_position):
        """Using Breadth First Search to find the closest wall or obstacle.

        Args:
            start_position: A [x,y,z] np.array of the start position of the search

        Returns:
            The position right before wall
        """

        queue = np.array([start_position])
        visited = np.array([start_position])
        while len(queue):
            current_position, queue = queue[0], queue[1:]
            neighbours = self.get_neighbours(current_position)
            for neighbour in neighbours:
                if not self.motion_planner.is_valid_step(current_position, neighbour):
                    return current_position
                if self.is_in_list(visited, neighbour):
                    continue
                
                queue = np.append(queue, [neighbour], axis=0)
                visited = np.append(visited, [neighbour], axis=0)

        return False


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
            neighbours = self.get_neighbours(current_position, angle_offset)
            for neighbour in neighbours:
                
                if self.is_blocked(current_position, neighbour, current_full_path):
                    continue

                current_position = neighbour
                
                current_full_path = np.append( current_full_path, [neighbour], axis=0 )
                local_path = np.append( local_path, [neighbour], axis=0 )

                critical_point_found  = False

                break

        return local_path, current_position
        
    def get_next_starting_point(self, path, angle_offset = 0):
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
                    return point

        return False



    def get_neighbours(self, current_position, angle_offset = 0):
        """Finds all neighbours of a given position. 

        Args:
            current_position: A [x,y,z] np.array of the start position 
            angle_offset: Angle offset in radians

        Returns:
            All 8 neighbours of the given position
        """
        directions = []
        for direction_idx in range(8):
            angle = direction_idx/8*np.pi*2 + angle_offset
            x = current_position[0] + np.cos(angle) * BASTAR_STEP_SIZE
            y = current_position[1] + np.sin(angle) * BASTAR_STEP_SIZE
            z = current_position[2]
            pos = np.array([x, y, z])
            directions.append(self.pcd.find_k_nearest(pos, 1)[0])

        east, northeast, north, northwest, west, southwest, south, southeast = directions

        return [west, northwest, southwest, north, south, northeast, southeast, east]

    def is_in_list(self, list, array):
        """Checks if an array is in a list by checking if it has 
        values close to it. 

        Args:
            list: list with arrays
            array: array to check

        Returns:
            True if it finds the array in the list 
        """
        diffs =  np.linalg.norm(list - array, axis=1)
        return np.any(diffs < 0.05)

    def has_been_visited(self, point, path=None):
        """Checks if a point has been visited. Looks if the distance to a point in the
        path is smaller than SPIRAL_VISITED_TRESHOLD.

        Args:
            point: A [x,y,z] np.array of the point that should be checked.
            path (optional): Specific path. Defaults to None.

        Returns:
            True if the point has been classified as visited
        """
        if path is None:
            path = self.path

        distances = np.linalg.norm(path - point, axis=1)

        return np.any(distances <= BASTAR_VISITED_TRESHOLD) 

    def is_blocked(self, from_point, to_point, path = None):
        """Checks if a step is valid by looking if the end point has been visited 
        or is an obstacle.

        Args:
            from_point: A [x,y,z] np.array of the start position
            to_point: A [x,y,z] np.array of the end position
            path (optional): Specific path. Defaults to None.

        Returns:
            True if the point has been classified as blocked
        """

        if path is None:
            path = self.path

        if self.has_been_visited(to_point, path):
            return True
        
        if not self.motion_planner.is_valid_step(from_point, to_point):
            return True
        
        return False
