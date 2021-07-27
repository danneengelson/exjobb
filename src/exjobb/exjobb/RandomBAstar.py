
import numpy as np
import operator
import pickle

from exjobb.CPPSolver import CPPSolver
from exjobb.RandomBAStarSegment import BAStarSegment
from exjobb.RandomSpriralSegment import RandomSpiralSegment
from exjobb.Parameters import RANDOM_BASTAR_VISITED_TRESHOLD, COVEREAGE_EFFICIENCY_GOAL, RANDOM_BASTAR_MAX_ITERATIONS, RANDOM_BASTAR_NUMBER_OF_ANGLES, RANDOM_BASTAR_PART_I_COVERAGE, RANDOM_BASTAR_MIN_COVERAGE, RANDOM_BASTAR_MIN_SPIRAL_LENGTH
from exjobb.PointCloud import PointCloud
from exjobb.Tree import Tree

DO_BASTAR_PLANNING = True

class RandomBAstar(CPPSolver):
    ''' Solving the Coverage Path Planning Problem with Random Sample BAstar with Inward Spiral
    '''
    def __init__(self, print, motion_planner):
        '''
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        '''
        self.print = print
        super().__init__(print, motion_planner)
        self.name = "Random BAstar"

    def get_cpp_path(self, start_point):
        """Generates a path that covers the area using BAstar Algorithm.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot

        Returns:
            Nx3 array with waypoints
        """

        self.start_tracking()
        
        self.path = np.array([start_point])
        self.move_to(start_point)

        current_position = start_point
        total_nbr_of_points = len(self.pcd.points)
        iter = 0

        if DO_BASTAR_PLANNING:
            Paths = []
            uncovered_points = np.arange(total_nbr_of_points)
            visited_waypoints = np.empty((0,3))
            coverage_part_I = 0
            #len(uncovered_points)/total_nbr_of_points > 0.05
            while coverage_part_I < RANDOM_BASTAR_PART_I_COVERAGE and iter <= RANDOM_BASTAR_MAX_ITERATIONS:
                iter += 1
                
                random_point = self.get_random_uncovered_point(visited_waypoints, iter)
                BAstar_paths_from_point = []

                for angle_idx in range(RANDOM_BASTAR_NUMBER_OF_ANGLES):
                    angle_offset = angle_idx * np.pi/RANDOM_BASTAR_NUMBER_OF_ANGLES
                    new_BAstar_path = BAStarSegment(self.print, self.motion_planner, random_point, angle_offset, visited_waypoints)
                    
                    BAstar_paths_from_point.append(new_BAstar_path)
                    
                    if new_BAstar_path.coverage == 0:
                        break

                best_BAstar_paths_from_point = max(BAstar_paths_from_point, key=operator.attrgetter("coverage"))
                self.print(str(iter) + "- coverage: " + str(best_BAstar_paths_from_point.coverage))
                
                if best_BAstar_paths_from_point.coverage > RANDOM_BASTAR_MIN_COVERAGE:
                    Paths.append( best_BAstar_paths_from_point )
                    visited_waypoints = np.append(visited_waypoints, best_BAstar_paths_from_point.path, axis=0)
                    uncovered_points = self.delete_values(uncovered_points, best_BAstar_paths_from_point.covered_points_idx)
                    coverage_part_I = 1 - len(uncovered_points)/ total_nbr_of_points
                    self.print("Coverage part I: " + str(coverage_part_I))

            
            self.print("Number of found paths: " + str(len(Paths)))

            covered_points_idx = self.get_data_from_paths(Paths)

            with open('cached_sample_based_bastar.dictionary', 'wb') as cached_pcd_file:
                cache_data = {  "covered_points_idx": covered_points_idx, 
                                "visited_waypoints": visited_waypoints,
                                "paths": Paths,
                                }
                pickle.dump(cache_data, cached_pcd_file)
        else:
            with open('cached_sample_based_bastar.dictionary', 'rb') as cached_pcd_file:
                cache_data = pickle.load(cached_pcd_file)
                covered_points_idx = np.unique(cache_data["covered_points_idx"])
                visited_waypoints = cache_data["visited_waypoints"]
                Paths = cache_data["paths"]
        
        self.pcd.covered_points_idx = covered_points_idx
        coverage_part_II = len(covered_points_idx)/ total_nbr_of_points
        
        self.print("Coverage part II: " + str(coverage_part_II))
        self.print("visited_waypoints: " + str(visited_waypoints))

        while coverage_part_II < COVEREAGE_EFFICIENCY_GOAL: 
            iter += 1
            random_uncovered_point = self.get_random_uncovered_point(visited_waypoints, iter)
            spiral_path = RandomSpiralSegment(self.print, self.motion_planner, random_uncovered_point, visited_waypoints)
            
            if len(spiral_path.path) < RANDOM_BASTAR_MIN_SPIRAL_LENGTH:
                continue

            Paths.append(spiral_path)    
            visited_waypoints = np.append(visited_waypoints, spiral_path.path, axis=0)
            covered_points_idx = np.unique(np.append(covered_points_idx, spiral_path.covered_points_idx))
            self.print(len(covered_points_idx) / total_nbr_of_points)
            coverage_part_II = len(covered_points_idx) / total_nbr_of_points

        paths_to_visit_in_order  = self.traveling_salesman(Paths)

        self.follow_given_path(current_position, paths_to_visit_in_order)

        self.print_stats(self.path)

        return self.path

    

    


    def traveling_salesman(self, paths):
        """Using Traveling Salesman Algorithm to order the path in an order
        that would minimise the total length of the path.

        Args:
            paths: List of paths of types RandomSpiralSegment and BAStarSegment

        Returns:
            Ordered list of paths
        """
        tree = Tree("BAstar paths")      
        start_nodes_idx = []  
        end_nodes_idx = []  
        
        def get_weight(from_idx, to_idx):
            from_point = tree.nodes[from_idx]
            to_point = tree.nodes[to_idx]
            return 100 + np.linalg.norm( to_point[0:2] - from_point[0:2] ) + 10 * abs( to_point[2] - from_point[2] )

        for path in paths:
            start_point = path.start
            end_point = path.end
            start_point_node_idx = tree.add_node(start_point)
            start_nodes_idx.append(start_point_node_idx)

            for node_idx, point in enumerate(tree.nodes[:-1]):
                tree.add_edge(start_point_node_idx, node_idx, get_weight(start_point_node_idx, node_idx))
            
            end_point_node_idx = tree.add_node(end_point)
            end_nodes_idx.append(end_point_node_idx)
            for node_idx, point in enumerate(tree.nodes[:-2]):
                tree.add_edge(end_point_node_idx, node_idx, get_weight(end_point_node_idx, node_idx))

            tree.add_edge(start_point_node_idx, end_point_node_idx, 0)

        traveling_Salesman_path =  tree.get_traveling_salesman_path()
        self.print(traveling_Salesman_path)

        paths_in_order = []
        current_position = np.array([0,0,0])

        for node_idx in traveling_Salesman_path:

            if np.array_equal(tree.nodes[node_idx], current_position):
                continue

            if node_idx in start_nodes_idx:
                path_idx = start_nodes_idx.index(node_idx)
                current_path = paths[path_idx]
                paths_in_order.append(current_path)
            elif node_idx in end_nodes_idx:
                path_idx = end_nodes_idx.index(node_idx)
                current_path = paths[path_idx]
                current_path.path = np.flip(current_path.path, 0)
                current_path.end = current_path.start
                current_path.start = tree.nodes[node_idx]
                paths_in_order.append(current_path)
            else:
                self.print("Not start or end point..")

            current_position = current_path.end

        return paths_in_order

        

    def get_data_from_paths(self, paths):
        """Splitting up data from the generating paths in the list for saving and
        future calculations

        Args:
            paths: Generated Paths of class RandomBAstarSegment

        Returns:
            List of start points, end points and points indices that has been covered
        """
        pcd = PointCloud(self.print, points=self.motion_planner.traversable_points)

        for path in paths:
            pcd.covered_points_idx = np.unique(np.append(pcd.covered_points_idx, path.covered_points_idx, axis=0))

        return pcd.covered_points_idx

    def follow_given_path(self, start_position, paths_to_visit_in_order):
        current_position = start_position

        def get_length_of_path(path):
            ''' Calculates length of the path in meters
            '''
            return np.linalg.norm( path[0, 0:2] - path[-1, 0:2] ), np.linalg.norm( path[0, 2] - path[-1, 2] )

        for idx, path in enumerate(paths_to_visit_in_order):
            
            self.print("Moving to start of path " + str(idx+1) + " out of " + str(len(paths_to_visit_in_order)))
            path_to_next_starting_point = self.motion_planner.Astar(current_position, path.start)
            self.follow_path(path_to_next_starting_point)
            self.path = np.append(self.path, path.path, axis=0)
            self.pcd.covered_points_idx = np.unique(np.append(self.pcd.covered_points_idx, path.covered_points_idx, axis=0))
            current_position = self.path[-1]

            

    def get_random_uncovered_point(self, visited_waypoints , iter = False ):
        """Returns a random uncovered point

        Args:
            visited_waypoints: Nx3 array with waypoints that has been visited
            iter (bool, optional): Integer for random seed. Defaults to False.

        Returns:
            A [x,y,z] position of an unvisited point.
        """
        all_points_idx = np.arange(len(self.pcd.points))
        if iter is False:
            np.random.seed(20*iter)

        random_idx = np.random.choice(len(all_points_idx), 1, replace=False)[0]
        random_point = self.pcd.points[all_points_idx[random_idx]]

        while self.has_been_visited(random_point, visited_waypoints) or not self.accessible(random_point, visited_waypoints):
            random_idx = np.random.choice(len(all_points_idx), 1, replace=False)[0]
            random_point = self.pcd.points[all_points_idx[random_idx]]

        return random_point

    def accessible(self, point, visited_waypoints):
        ''' Checks if a point is accessible by trying to make a path from the point
        to the closest visited point.
        Args:
            point:  A [x,y,z] np.array of the point position
            visited_waypoints: A Nx3 NumPy array with positions that has been visited.
        Returns:
            True if the point is accessible.
        '''
        if len(visited_waypoints) == 0:
            return True
        closest_point = visited_waypoints[np.argmin(np.linalg.norm(visited_waypoints - point, axis=1))]
        path_to_point = self.motion_planner.RRT(closest_point, point)
        return path_to_point is not False


    def delete_values(self, array, values):
        ''' Removes specific values from an array with unique values
        Args:
            array: NumPy array with unique values to remove values from
            values: NumPy array with values that should be removed.
        '''
        return array[ np.isin(array, values, assume_unique=True, invert=True) ]

    def delete_values_not_unique(self, array, values):
        ''' Removes specific values from an array
        Args:
            array: NumPy array to remove values from
            values: NumPy array with values that should be removed.
        '''
        return array[ np.isin(array, values, invert=True) ]

    def has_been_visited(self, point, path=None):
        ''' Removes specific values from an array
        Args:
            array: NumPy array to remove values from
            values: NumPy array with values that should be removed.
        '''
        if path is None:
            path = self.path

        distances = np.linalg.norm(path - point, axis=1)
        return np.any(distances <= RANDOM_BASTAR_VISITED_TRESHOLD) 