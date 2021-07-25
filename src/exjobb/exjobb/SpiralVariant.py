import numpy as np

from exjobb.CPPSolver import CPPSolver
from exjobb.MotionPlanner import MotionPlanner
from exjobb.Parameters import COVEREAGE_EFFICIENCY_GOAL, SPIRAL_STEP_SIZE, SPIRAL_VISITED_TRESHOLD

class SpiralVariant(CPPSolver):
    """ Implementation of a variation of Inward Spiral Coverage Path Planning Algorithm
    """

    def __init__(self, print, motion_planner):
        """
        Args: 
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        """
        
        self.print = print
        super().__init__(print, motion_planner)
        self.name = "Inward Spiral Variant"


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
        minimum = 3
        coverage = 0
        
        while coverage < COVEREAGE_EFFICIENCY_GOAL:

            next_starting_point, path_point_idx =  self.find_closest_free_pos_2(current_position, self.path)
            
            if next_starting_point is False:
                break

            path_until_dead_zone, new_current_position = self.get_path_until_dead_zone(next_starting_point, current_angle)
            
            while len(path_until_dead_zone) < minimum:
                next_starting_point, path_point_idx =  self.find_closest_free_pos_2(current_position, self.path, path_point_idx+1)
                #self.print(next_starting_point)
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
            #break
            #self.print("path_until_dead_zone" + str(path_until_dead_zone))
            self.follow_path(path_until_dead_zone)

            current_position = new_current_position
            current_angle = self.get_angle(self.path[-2], current_position)
            coverage = self.pcd.get_coverage_efficiency()
            self.print("coverage" + str(coverage))
  
        self.print_stats(self.path)
        #self.print(self.path)
        
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

    def find_closest_free_pos_2(self, start_position, path, ignore_up_to_idx = 0):
        potenital_pos = np.empty((0,3))
        for idx, point in enumerate(np.flip(path, 0)[ignore_up_to_idx:]):
            neighbours = self.get_neighbours(point)
            for neighbour in neighbours:
                if not self.is_blocked(point, neighbour):
                    potenital_pos = np.append(potenital_pos, [neighbour], axis=0)
            if len(potenital_pos):
                closest = self.get_closest_to(start_position, potenital_pos)
                return closest, ignore_up_to_idx + idx
        return False, False
    
    
    def wavefront_algorithm(self, start_position, path, a=False):
        last_layer = np.array([start_position])
        visited = np.array([start_position])
        while len(last_layer):
            #self.print(last_layer)
            new_layer = np.empty((0,3))
            for pos in last_layer:
                neighbours = self.get_neighbours(pos)
                #self.print("pos: " + str(pos))
                
                for neighbour in neighbours:
                    #self.print("neigbour: " + str(neighbour))

                    if self.has_been_visited(neighbour, visited):
                        #self.print("Visited")
                        continue                    

                    if not self.motion_planner.is_valid_step(pos, neighbour):
                        #self.print("Obstacle")
                        continue

                    if not self.has_been_visited(neighbour):
                        #self.print("OK!")
                        #current_angle = self.get_angle(start_position, neighbour)
                        return neighbour, 1

                    #self.print("Not free")
                    visited = np.append(visited, [neighbour], axis=0)
                    new_layer = np.append(new_layer, [neighbour], axis=0)

            last_layer = new_layer
        self.print("FAIL")
        #self.path = np.append(self.path, visited, axis=0)
        return False, False
                                    



                    


    def find_closest_free_pos(self, start_position, path, a=False):
        queue = np.array([start_position])
        visited = np.array([start_position])
        
        while len(queue):
            potenital_pos = np.empty((0,3))
            current_position, queue = queue[0], queue[1:]
            #self.print(queue)

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
        #self.print(np.min(diffs))
        return np.any(diffs < 0.05)


    def get_neighbours_for_spiral(self, current_position, current_angle):
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

    def get_neighbours(self, current_position, angle_offset=0):
        directions = []
        for direction_idx in range(8):
            angle = direction_idx/8*np.pi*2 + angle_offset
            x = current_position[0] + np.cos(angle) * SPIRAL_STEP_SIZE
            y = current_position[1] + np.sin(angle) * SPIRAL_STEP_SIZE
            z = current_position[2]
            pos = np.array([x, y, z])

            directions.append(self.pcd.find_k_nearest(pos, 1)[0])

        east, northeast, north, northwest, west, southwest, south, southeast = directions

        return [north, south, northeast, northwest, southeast, southwest, east, west]

    def has_been_visited(self, point, path=None):
        if path is None:
            path = self.path


        distances = np.linalg.norm(path - point, axis=1)
        return np.any(distances <= SPIRAL_VISITED_TRESHOLD) 

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