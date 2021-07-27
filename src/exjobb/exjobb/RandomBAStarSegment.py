from exjobb.Parameters import BASTAR_STEP_SIZE, RANDOM_BASTAR_VISITED_TRESHOLD, RANDOM_BASTAR_VARIANT_DISTANCE
from exjobb.PointCloud import PointCloud
import numpy as np

class BAStarSegment():
    """A class to generate a segment of BAstar path. Used in Sample-Based BAstar CPP Algorithm.
    """
    def __init__(self, print, motion_planner, starting_point, angle_offset, visited_waypoints):
        """
        Args:
            print: function for printing messages
            motion_planner:  Motion Planner of the robot wihch also has the Point Cloud
            starting_point: A [x,y,z] np.array of the start position of the robot
            angle_offset: An angle in radians, representing the primary direction of the paths
            visited_waypoints: A Nx3 array with points that has been visited and should be avoided
        """

        self.start = starting_point
        self.pcd = PointCloud(print, points=motion_planner.traversable_points)
        self.motion_planner = motion_planner
        self.path = np.array([starting_point])
        self.visited_waypoints = visited_waypoints
        next_starting_point_close = True
        self.print = print
        next_starting_point = starting_point
        
        
        while next_starting_point_close:
           
            
            current_position, path_to_cover_local_area = self.get_path_to_cover_local_area(next_starting_point, angle_offset)
            
            if len(path_to_cover_local_area) == 0:
                break

            self.path = np.append(self.path, path_to_cover_local_area, axis=0)
            self.pcd.visit_path(path_to_cover_local_area)            

            next_starting_point = self.get_next_starting_point(self.path, angle_offset)   

            if next_starting_point is False:
                break

            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
            self.pcd.visit_path(path_to_next_starting_point)
            self.path = np.append( self.path, path_to_next_starting_point, axis=0 )

        self.end = current_position
        self.covered_points_idx = self.pcd.covered_points_idx
        self.coverage = self.pcd.get_coverage_efficiency()

        self.pcd = None
        self.motion_planner = None
        self.print = None

        
    
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
        distances = distances[ distances < RANDOM_BASTAR_VARIANT_DISTANCE ]
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
        
        path = np.append(path, self.visited_waypoints, axis=0)
        distances = np.linalg.norm(path - point, axis=1)
        return np.any(distances <= RANDOM_BASTAR_VISITED_TRESHOLD) 


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

    def get_path_to_cover_local_area(self, start_point, angle_offset):
        """"Generates BAstar paths to cover local area.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot
            angle_offset: Angle offset in radians

        Returns:
            Generated path with waypoints and the current position of the robot at the end
        """
        
        current_position = start_point
        critical_point_found = False
        
        new_path = np.empty((0,3))
        visited_points = np.append(self.visited_waypoints, self.path, axis=0)

        while not critical_point_found:
            
            critical_point_found = True
            for neighbour in self.get_neighbours(current_position, angle_offset):
                
                if self.is_blocked(current_position, neighbour, visited_points):
                    continue

                current_position = neighbour
                new_path = np.append( new_path, [neighbour], axis=0 )
                visited_points = np.append( visited_points, [neighbour], axis=0 )
                critical_point_found  = False
                break
                
        return current_position, new_path


    def get_neighbours(self, current_position, angle_offset=0):
        """Finds all neighbours of a given position. 

        Args:
            current_position: A [x,y,z] np.array of the start position 
            angle_offset: Angle offset in radians
            angle_offset (optional): Angle offset in radians. Defaults to 0.

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

        return [north, south, northeast, northwest, southeast, southwest, east, west]

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

    def get_angle(self, from_pos, to_pos):
        """Calculates the angle of the robot after making a step

        Args:
            from_pos: A [x,y,z] np.array of the start position 
            to_pos: A [x,y,z] np.array of the end position

        Returns:
            An angle in radians
        """

        vec = to_pos[0:2] - from_pos[0:2]
        return np.angle( vec[0] + vec[1]*1j)

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

