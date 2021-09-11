
import timeit
import numpy as np
import operator
import pickle

from exjobb.CPPSolver import CPPSolver
from exjobb.RandomBAStarSegment import BAStarSegment
from exjobb.RandomSpriralSegment import RandomSpiralSegment
from exjobb.Parameters import RANDOM_BASTAR_VISITED_TRESHOLD, COVEREAGE_EFFICIENCY_GOAL, RANDOM_BASTAR_MAX_ITERATIONS, RANDOM_BASTAR_NUMBER_OF_ANGLES, RANDOM_BASTAR_PART_I_COVERAGE, RANDOM_BASTAR_MIN_COVERAGE, RANDOM_BASTAR_MIN_SPIRAL_LENGTH, RANDOM_BASTAR_VARIANT_DISTANCE, RANDOM_BASTAR_VARIANT_DISTANCE_PART_II, ROBOT_SIZE
from exjobb.PointCloud import PointCloud
from exjobb.Tree import Tree

DO_BASTAR_PLANNING = True
ONLY_PART_I = False

class RandomBAstar(CPPSolver):
    ''' Solving the Coverage Path Planning Problem with Random Sample BAstar with Inward Spiral
    '''
    def __init__(self, print, motion_planner, coverable_pcd, time_limit=None, parameters=None):
        '''
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        '''
        self.print = print
        super().__init__(print, motion_planner, coverable_pcd, time_limit)
        self.name = "Random BAstar"
        

        if parameters is None:
            self.max_distance = RANDOM_BASTAR_VARIANT_DISTANCE
            self.max_distance_part_II = RANDOM_BASTAR_VARIANT_DISTANCE_PART_II
            self.nbr_of_angles = RANDOM_BASTAR_NUMBER_OF_ANGLES
            self.coverage_1 = RANDOM_BASTAR_PART_I_COVERAGE
            self.coverage_2 = COVEREAGE_EFFICIENCY_GOAL
            self.min_spiral_length = RANDOM_BASTAR_MIN_SPIRAL_LENGTH
            self.min_bastar_coverage = RANDOM_BASTAR_MIN_COVERAGE
            self.max_iterations = RANDOM_BASTAR_MAX_ITERATIONS
        else:
            self.max_distance = parameters["max_distance"]
            self.max_distance_part_II = parameters["max_distance_part_II"]
            self.nbr_of_angles = int(round(parameters["nbr_of_angles"]))
            self.coverage_1 = parameters["coverage_1"]
            self.coverage_2 = parameters["coverage_2"]
            self.min_spiral_length = parameters["min_spiral_length"]
            self.min_bastar_coverage = parameters["min_bastar_coverage"]
            self.max_iterations = parameters["max_iterations"]
            self.step_size = parameters["step_size"] * ROBOT_SIZE
            self.visited_threshold = parameters["visited_threshold"] * self.step_size


        self.randombastar_stats = {}
        self.randombastar_stats_over_time = []


    def get_cpp_path(self, start_point, angle_offset_fake = None,  goal_coverage = None):
        """Generates a path that covers the area using BAstar Algorithm.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot

        Returns:
            Nx3 array with waypoints
        """

        if goal_coverage is not None:
            self.coverage_2 = goal_coverage
            

        self.start_tracking()
        
        self.path = np.array([start_point])
        self.move_to(start_point)

        current_position = start_point
        total_nbr_of_points = len(self.coverable_pcd.points)
        iter = 0

        if DO_BASTAR_PLANNING:
            Paths = []
            uncovered_points = np.arange(total_nbr_of_points)
            visited_waypoints = np.empty((0,3))
            coverage_part_I = 0
            #len(uncovered_points)/total_nbr_of_points > 0.05
            while coverage_part_I < self.coverage_1 and iter <= self.max_iterations and coverage_part_I < self.coverage_2:
                iter += 1
                
                random_point = self.get_random_uncovered_point(visited_waypoints)

                
                BAstar_paths_from_point = []

                for angle_idx in range(self.nbr_of_angles):
                    angle_offset = angle_idx * 2*np.pi/self.nbr_of_angles
                    coverable_pcd = PointCloud(self.print, points=self.coverable_pcd.points)
                    new_BAstar_path = BAStarSegment(self.print, self.motion_planner, random_point, angle_offset, visited_waypoints, coverable_pcd, self.max_distance, self.step_size, self.visited_threshold)
                    
                    BAstar_paths_from_point.append(new_BAstar_path)
                    
                    if new_BAstar_path.coverage == 0:
                        break

                best_BAstar_paths_from_point = max(BAstar_paths_from_point, key=operator.attrgetter("coverage"))
                self.print(str(iter) + "- coverage: " + str(best_BAstar_paths_from_point.coverage))
                
                if best_BAstar_paths_from_point.coverage > self.min_bastar_coverage:
                    Paths.append( best_BAstar_paths_from_point )
                    visited_waypoints = np.append(visited_waypoints, best_BAstar_paths_from_point.path, axis=0)
                    uncovered_points = self.delete_values(uncovered_points, best_BAstar_paths_from_point.covered_points_idx)
                    coverage_part_I = 1 - len(uncovered_points)/ total_nbr_of_points
                    self.print("Coverage part I: " + str(coverage_part_I))
                    self.randombastar_stats_over_time.append({
                        "time": timeit.default_timer() - self.start_time,
                        "coverage": coverage_part_I,
                        "iteration": iter,
                        "path": best_BAstar_paths_from_point.path,
                        "segment": best_BAstar_paths_from_point
                    })

            
            self.print("Number of found paths: " + str(len(Paths)))

           
            

            covered_points_idx = self.get_covered_points_idx_from_paths(Paths)

        #    with open('cached_sample_based_bastar.dictionary', 'wb') as cached_pcd_file:
        #        cache_data = {  "covered_points_idx": covered_points_idx, 
        #                        "visited_waypoints": visited_waypoints,
        #                        "paths": Paths,
        #                        }
        #        pickle.dump(cache_data, cached_pcd_file)
        #else:
        #    with open('cached_sample_based_bastar.dictionary', 'rb') as cached_pcd_file:
        #        cache_data = pickle.load(cached_pcd_file)
        #        covered_points_idx = np.unique(cache_data["covered_points_idx"])
        #        visited_waypoints = cache_data["visited_waypoints"]
        #        Paths = cache_data["paths"]
        
        if not ONLY_PART_I:

            self.coverable_pcd.covered_points_idx = covered_points_idx
            coverage_part_II = len(covered_points_idx)/ total_nbr_of_points

            self.randombastar_stats["Part1_segments"] = len(Paths)
            self.randombastar_stats["Part1_coverage"] = coverage_part_II
            self.randombastar_stats["Part1_iterations"] = iter
            self.randombastar_stats["Part1_time"] = timeit.default_timer() - self.start_time
            
            self.print("Coverage part II: " + str(coverage_part_II))
            self.print("visited_waypoints: " + str(visited_waypoints))
            iter = 0
            failed_tries = 0
            while coverage_part_II < self.coverage_2 and not self.time_limit_reached(): 
                iter += 1
                if failed_tries > 100:
                    break

                random_uncovered_point = self.get_random_uncovered_point(visited_waypoints)

                coverable_pcd = PointCloud(self.print, points=self.coverable_pcd.points)
                spiral_path = RandomSpiralSegment(self.print, self.motion_planner, random_uncovered_point, visited_waypoints, coverable_pcd, self.max_distance_part_II, self.step_size, self.visited_threshold)
                
                if len(spiral_path.path) < self.min_spiral_length:
                    visited_waypoints = np.append(visited_waypoints, [random_uncovered_point], axis=0)
                    failed_tries += 1
                    continue
                failed_tries = 0

                self.print("length: " + str(len(spiral_path.path)))
                Paths.append(spiral_path)    
                visited_waypoints = np.append(visited_waypoints, spiral_path.path, axis=0)
                covered_points_idx = np.unique(np.append(covered_points_idx, spiral_path.covered_points_idx))
                self.print(len(covered_points_idx) / total_nbr_of_points)
                coverage_part_II = len(covered_points_idx) / total_nbr_of_points
                self.randombastar_stats_over_time.append({
                        "time": timeit.default_timer() - self.start_time,
                        "coverage": coverage_part_II,
                        "iteration": iter,
                        "path": spiral_path.path,
                        "segment": spiral_path
                    })

            self.randombastar_stats["Part2_segments"] = len(Paths) - self.randombastar_stats["Part1_segments"]
            self.randombastar_stats["Part2_coverage"] = coverage_part_II
            self.randombastar_stats["Part2_iterations"] = iter
            self.randombastar_stats["Part2_time"] = timeit.default_timer() - self.start_time

            paths_to_visit_in_order  = self.traveling_salesman(Paths)

            self.follow_paths(current_position, paths_to_visit_in_order)

            #self.print_stats(self.path)
            #self.print(self.randombastar_stats)

        #self.print(self.randombastar_stats_over_time)
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

        

    def get_covered_points_idx_from_paths(self, paths):
        """Finds indices of all covered points by the given paths

        Args:
            paths: Generated Paths of class RandomBAstarSegment

        Returns:
            List of points indices that has been covered
        """
        covered_points_idx = np.array([])

        for path in paths:
            covered_points_idx = np.unique(np.append(covered_points_idx, path.covered_points_idx, axis=0))

        return covered_points_idx

    def follow_paths(self, start_position, paths_to_visit_in_order):
        """Connects all paths with Astar and make the robot walk through the paths.

        Args:
            start_position: A [x,y,z] np.array of the start position of the robot
            paths_to_visit_in_order:    Ordered list of paths of types RandomSpiralSegment 
                                        and BAStarSegment
        """
        current_position = start_position

        for idx, path in enumerate(paths_to_visit_in_order):
            
            self.print("Moving to start of path " + str(idx+1) + " out of " + str(len(paths_to_visit_in_order)))
            path_to_next_starting_point = self.motion_planner.Astar(current_position, path.start)
            self.follow_path(path_to_next_starting_point)
            self.path = np.append(self.path, path.path, axis=0)
            self.coverable_pcd.covered_points_idx = np.unique(np.append(self.coverable_pcd.covered_points_idx, path.covered_points_idx, axis=0))
            current_position = self.path[-1]

    def get_random_uncovered_point(self, visited_waypoints , iter = False ):
        """Returns a random uncovered point

        Args:
            visited_waypoints: Nx3 array with waypoints that has been visited
            iter (bool, optional): Integer for random seed. Defaults to False.

        Returns:
            A [x,y,z] position of an unvisited point.
        """
        all_points_idx = np.arange(len(self.traversable_pcd.points))
        if iter is not False:
            np.random.seed(20*iter)

        random_idx = np.random.choice(len(all_points_idx), 1, replace=False)[0]
        random_point = self.traversable_pcd.points[all_points_idx[random_idx]]

        while self.has_been_visited(random_point, RANDOM_BASTAR_VISITED_TRESHOLD,  visited_waypoints) or not self.accessible(random_point, visited_waypoints):
            random_idx = np.random.choice(len(all_points_idx), 1, replace=False)[0]
            random_point = self.traversable_pcd.points[all_points_idx[random_idx]]
            

        return random_point




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