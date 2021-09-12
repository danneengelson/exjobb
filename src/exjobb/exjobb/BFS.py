import numpy as np
import timeit
from exjobb.CPPSolver import CPPSolver 
from exjobb.Parameters import BASTAR_STEP_SIZE, BASTAR_VISITED_TRESHOLD, COVEREAGE_EFFICIENCY_GOAL, ROBOT_SIZE

class BFS(CPPSolver):
    ''' Solving the Coverage Path Planning Problem with BAstar
    '''
    def __init__(self, print, motion_planner, coverable_pcd, time_limit=None, parameters=None):
        '''
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        '''
        super().__init__(print, motion_planner, coverable_pcd, time_limit)
        self.name = "BFS"
        if parameters is None:
            self.step_size = BASTAR_STEP_SIZE
            self.visited_threshold = BASTAR_VISITED_TRESHOLD
        else:
            self.step_size = parameters["step_size"]
            self.visited_threshold =  parameters["visited_threshold"]

        self.print("step_size: " + str(self.step_size))
        self.print("visited_threshold: " + str(self.visited_threshold))


    def breadth_first_search(self, start_pos, goal_coverage = 1):
        
        start_node = 0
        self.points = np.array([start_pos])
        self.neighbours = {}
        queue = np.array([0])
        self.start_tracking()
        self.move_to(start_pos)
        current_percent = 0

        while len(queue):

            if self.time_limit_reached():
                break
            #self.print(len(self.points))
            #self.print(len(self.neighbours))
            curr_node, queue = queue[0], queue[1:]
            curr_point = self.points[curr_node]
            self.coverable_pcd.visit_position(curr_point, apply_unique=True)
            coverage = self.coverable_pcd.get_coverage_efficiency()
            if coverage > current_percent:
                self.print(coverage)
                current_percent += 0.01
            if goal_coverage < coverage:
                break

            neighbours = {}

            for direction, neighbour_point in enumerate(self.get_cell_neighbours(curr_point)):

                if not self.motion_planner.is_valid_step(curr_point, neighbour_point):
                    continue
                
                neighbour_node = self.get_node(neighbour_point)

                if neighbour_node is False:
                    neighbour_node = len(self.points)
                    self.points = np.append(self.points, [neighbour_point], axis=0)
                    queue = np.append(queue, neighbour_node)
                elif neighbour_node == curr_node:
                    continue

                neighbours[direction] = neighbour_node
            
            self.neighbours[curr_node] = neighbours
        self.path = self.points
        return self.points

    def get_cell_neighbours(self, current_position):
        """Finds all neighbours of a given position. 

        Args:
            current_position: A [x,y,z] np.array of the start position 
            step_size: The approximate distance to the neighbours
            angle_offset: Optional. The yaw angle of the robot.

        Returns:
            All 8 neighbours of the given position in following order:
            right, forwardright, forward, forwardleft, left, backleft, back, backright
        """
        directions = []
        for direction_idx in range(8):
            angle = direction_idx/8*np.pi*2 + np.pi/2
            x = current_position[0] + np.cos(angle) * self.step_size
            y = current_position[1] + np.sin(angle) * self.step_size 
            z = current_position[2]
            pos = np.array([x, y, z])

            directions.append(self.motion_planner.traversable_pcd.find_k_nearest(pos, 1)[0])

        return directions

    def get_node(self, point):
        ''' Finds the node (index in self.astar_points) that is close to a given point.
        Args:
            point: A [x,y,z] array with the position of the point
        Returns:
            Index of the closest point in self.astar_points or False if no point found nearby.
        '''
        distance_to_existing_nodes = np.linalg.norm(self.points - point, axis=1)
        closest_node = np.argmin(distance_to_existing_nodes)
        distance = distance_to_existing_nodes[closest_node]
        if distance < self.step_size * 0.5:
            return closest_node

        return False
        

    def print_results(self):
        #for idx, point in enumerate(self.points):
        #    self.coverable_pcd.visit_position(point, apply_unique=False)
        #self.coverable_pcd.covered_points_idx = np.unique(self.coverable_pcd.covered_points_idx )
        coverage = self.coverable_pcd.get_coverage_efficiency()
        time = timeit.default_timer() - self.start_time
        self.print("coverage: " + str(coverage*100))
        self.coverable_pcd.covered_points_idx = np.array([])
        self.print("time: " + str(timeit.default_timer() - self.start_time))
        return {
            "name": "BFS",
            "no_of_nodes": len(self.points),
            "coverage": coverage*100,
            "time": time,
            "step_size": self.step_size,
            "visited_threshold": self.visited_threshold
        }
