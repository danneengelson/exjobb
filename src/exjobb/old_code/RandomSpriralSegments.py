from exjobb.Parameters import SPIRAL_STEP_SIZE, RANDOM_BASTAR_VISITED_TRESHOLD, RANDOM_BASTAR_VARIANT_DISTANCE
from exjobb.PointCloud import PointCloud
import numpy as np

class RandomSpiralSegment:
    """A class to generate a segment of Spiral path. Used in Sample-Based BAstar 
    CPP Algorithm.
    """
    def __init__(self, print, motion_planner, starting_point, visited_waypoints):
        """
        Args:
            print: function for printing messages
            motion_planner:  Motion Planner of the robot wihch also has the Point Cloud
            starting_point: A [x,y,z] np.array of the start position of the robot
            visited_waypoints: A Nx3 array with points that has been visited and should be avoided
        """
        
        self.visited_waypoints = visited_waypoints
        self.start = starting_point
        self.print = print 
        self.pcd = PointCloud(print, points=motion_planner.traversable_points)
        self.motion_planner = motion_planner
        current_angle = 0

        self.path = np.empty((0,3))
        next_starting_point = starting_point

        while True:

            local_spiral_path, current_position = self.get_path_until_dead_zone(next_starting_point, current_angle)
            self.path = np.append(self.path, local_spiral_path, axis=0)
            self.visited_waypoints = np.append(self.visited_waypoints, local_spiral_path, axis=0)
            
            next_starting_point =  self.wavefront_algorithm(current_position)

            if next_starting_point is False:
                break

            if np.linalg.norm(next_starting_point - current_position) > RANDOM_BASTAR_VARIANT_DISTANCE:
                break
            
            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
            
            
            if path_to_next_starting_point is False:
                break

            self.path = np.append(self.path, path_to_next_starting_point, axis=0)
            current_position = next_starting_point

            if len(self.path) >= 2:
                current_angle = self.get_angle(self.path[-2], current_position)
            
        self.end = current_position
        self.print("Length of path: " + str(len(self.path)))

        if len(self.path) > 1:
            self.pcd.visit_path(self.path)
            self.covered_points_idx = self.pcd.covered_points_idx

        self.coverage = self.pcd.get_coverage_efficiency()

        self.pcd = None
        self.motion_planner = None

    def wavefront_algorithm(self, start_position):
        """Using Wavefront algorithm to find the closest obstacle free uncovered position.

        Args:
            start_position: A [x,y,z] np.array of the start position of the search

        Returns:
            An obstacle free uncovered position.
        """
        last_layer = np.array([start_position])
        visited = np.array([start_position])
        visited_points = np.append(self.visited_waypoints, self.path, axis=0)
        while len(last_layer):
            new_layer = np.empty((0,3))
            for pos in last_layer:
                neighbours = self.get_neighbours(pos)
                
                for neighbour in neighbours:

                    if self.has_been_visited(neighbour, visited):
                        continue                    

                    if not self.motion_planner.is_valid_step(pos, neighbour):
                        continue

                    if not self.has_been_visited(neighbour, visited_points):
                        return neighbour

                    visited = np.append(visited, [neighbour], axis=0)
                    new_layer = np.append(new_layer, [neighbour], axis=0)

            last_layer = new_layer

        self.print("FAIL. No new uncovered obstacle free positions could be found.")
        return False

    def get_path_until_dead_zone(self, current_position, current_angle):
        """Covers the area in an inward spiral motion until  a dead zone is reached.

        Args:
            current_position: A [x,y,z] np.array of the start position of the search
            current_angle: A float value representing the starting angle in radians 

        Returns:
            New part of the path with waypoints and the position of the robot at the 
            end of the path.
        """

        local_path = np.array([current_position])
        dead_zone_reached = False
        visited_points = np.append(self.visited_waypoints, self.path, axis=0)
        
        while not dead_zone_reached:
            dead_zone_reached =  True
            neighbours = self.get_neighbours_for_spiral(current_position, current_angle)

            for neighbour in neighbours: 
                

                if self.is_blocked(current_position, neighbour, visited_points) :
                    continue

                local_path = np.append(local_path, [neighbour], axis=0)
                visited_points = np.append(visited_points, [neighbour], axis=0)
                current_angle = self.get_angle(current_position, neighbour)
                current_position = neighbour
                
                dead_zone_reached = False
                break

        return local_path, current_position

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


    def get_neighbours_for_spiral(self, current_position, current_angle):
        """Finds neighbours of a given position. And return them in the order
        to create the inward spiral motion.

        Args:
            current_position: A [x,y,z] np.array of the start position 
            current_angle: A float value representing the starting angle in radians

        Returns:
            List of neighbours in specific order to get the inward spiral motion,
        """
        directions = []
        for direction_idx in range(8):
            angle = direction_idx/8*np.pi*2 + current_angle
            x = current_position[0] + np.cos(angle) * SPIRAL_STEP_SIZE
            y = current_position[1] + np.sin(angle) * SPIRAL_STEP_SIZE
            z = current_position[2]
            pos = np.array([x, y, z])
            directions.append(self.pcd.find_k_nearest(pos, 1)[0])
        
        right, forwardright, forward, forwardleft, left, backleft, back, backright = directions
        return [backright, right, forwardright, forward, forwardleft, left, backleft]

    def get_neighbours(self, current_position):
        """Finds all neighbours of a given position. 

        Args:
            current_position: A [x,y,z] np.array of the start position 

        Returns:
            All 8 neighbours of the given position
        """
        directions = []
        for direction_idx in range(8):
            angle = direction_idx/8*np.pi*2
            x = current_position[0] + np.cos(angle) * SPIRAL_STEP_SIZE
            y = current_position[1] + np.sin(angle) * SPIRAL_STEP_SIZE
            z = current_position[2]
            pos = np.array([x, y, z])

            directions.append(self.pcd.find_k_nearest(pos, 1)[0])

        return directions

    def has_been_visited(self, point, path=None):
        """Checks if a point has been visited. Looks if the distance to a point in the
        path is smaller than RANDOM_BASTAR_VISITED_TRESHOLD.

        Args:
            point: A [x,y,z] np.array of the point that should be checked.
            path (optional): Specific path. Defaults to None.

        Returns:
            True if the point has been classified as visited
        """
        if path is None:
            path = self.path


        distances = np.linalg.norm(path - point, axis=1)
        return np.any(distances <= RANDOM_BASTAR_VISITED_TRESHOLD) 

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
