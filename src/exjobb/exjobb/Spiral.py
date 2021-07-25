import numpy as np

from exjobb.CPPSolver import CPPSolver
from exjobb.MotionPlanner import MotionPlanner
from exjobb.Parameters import COVEREAGE_EFFICIENCY_GOAL, SPIRAL_STEP_SIZE, SPIRAL_VISITED_TRESHOLD

class Spiral(CPPSolver):
    """ Implementation of the Inward Spiral Coverage Path Planning Algorithm
    """

    def __init__(self, print, motion_planner):
        """
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        """
        
        self.print = print
        super().__init__(print, motion_planner)
        self.name = "Inward Spiral"


    def get_cpp_path(self, start_point):
        """Generates a path that covers the area using Inward Spiral Algorithm.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot

        Returns:
            Nx3 array with waypoints
        """

        self.start_tracking()
        self.move_to(start_point)

        next_starting_point, current_angle = self.find_closest_wall(start_point)
        path_to_next_starting_point = self.motion_planner.Astar(start_point, next_starting_point)
        self.follow_path(path_to_next_starting_point)

        current_position = next_starting_point
        coverage = 0
        
        while coverage < COVEREAGE_EFFICIENCY_GOAL:

            next_starting_point =  self.wavefront_algorithm(current_position)
            
            if next_starting_point is False:
                break

            path_until_dead_zone, new_current_position = self.get_path_until_dead_zone(next_starting_point, current_angle)

            path_to_next_starting_point = self.motion_planner.Astar(current_position, next_starting_point)
            
            if path_to_next_starting_point is False:
                break

            self.follow_path(path_to_next_starting_point)

            self.follow_path(path_until_dead_zone)

            current_position = new_current_position
            current_angle = self.get_angle(self.path[-2], current_position)
            coverage = self.pcd.get_coverage_efficiency()
            self.print("coverage" + str(coverage))
  
        self.print_stats(self.path)
        
        return self.path


    def find_closest_wall(self, start_position):
        """Using Breadth First Search to find the closest wall or obstacle.

        Args:
            start_position: A [x,y,z] np.array of the start position of the search

        Returns:
            The position right before wall and the angle of the robot, which will be 
            along the wall.
        """
        queue = np.array([start_position])
        visited = np.array([start_position])
        while len(queue):
            current_position, queue = queue[0], queue[1:]
            neighbours = self.get_neighbours(current_position)
            for neighbour in neighbours:
                if not self.motion_planner.is_valid_step(current_position, neighbour):
                    current_angle = self.get_angle(start_position, current_position) + np.pi/2
                    return current_position, current_angle
                if self.is_in_list(visited, neighbour):
                    continue
                
                queue = np.append(queue, [neighbour], axis=0)
                visited = np.append(visited, [neighbour], axis=0)

        return False, False

 
    def wavefront_algorithm(self, start_position):
        """Using Wavefront algorithm to find the closest obstacle free uncovered position.

        Args:
            start_position: A [x,y,z] np.array of the start position of the search

        Returns:
            An obstacle free uncovered position.
        """
        last_layer = np.array([start_position])
        visited = np.array([start_position])
        while len(last_layer):
            new_layer = np.empty((0,3))
            for pos in last_layer:
                neighbours = self.get_neighbours(pos)
                
                for neighbour in neighbours:

                    if self.has_been_visited(neighbour, visited):
                        continue                    

                    if not self.motion_planner.is_valid_step(pos, neighbour):
                        continue

                    if not self.has_been_visited(neighbour):
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
        while not dead_zone_reached:
            dead_zone_reached =  True
            neighbours = self.get_neighbours_for_spiral(current_position, current_angle)

            for neighbour in neighbours: 
                current_path = np.append(self.path, local_path, axis=0)

                if self.is_blocked(current_position, neighbour, current_path) :
                    continue

                local_path = np.append(local_path, [neighbour], axis=0)
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
        return np.any(distances <= SPIRAL_VISITED_TRESHOLD) 

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
