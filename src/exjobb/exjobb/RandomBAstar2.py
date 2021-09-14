
import timeit
import numpy as np
import operator
import pickle

from exjobb.CPPSolver import CPPSolver
from exjobb.RandomBAStarSegment import BAStarSegment
from exjobb.RandomSpriralSegment import RandomSpiralSegment
from exjobb.Parameters import BASTAR_STEP_SIZE, BASTAR_VISITED_TRESHOLD, RANDOM_BASTAR_VISITED_TRESHOLD, COVEREAGE_EFFICIENCY_GOAL, RANDOM_BASTAR_MAX_ITERATIONS, RANDOM_BASTAR_NUMBER_OF_ANGLES, RANDOM_BASTAR_PART_I_COVERAGE, RANDOM_BASTAR_MIN_COVERAGE, RANDOM_BASTAR_MIN_SPIRAL_LENGTH, RANDOM_BASTAR_VARIANT_DISTANCE, RANDOM_BASTAR_VARIANT_DISTANCE_PART_II, ROBOT_RADIUS, ROBOT_SIZE
from exjobb.PointCloud import PointCloud
from exjobb.Tree import Tree
from exjobb.RandomBorderSegment import RandomBorderSegment

DO_BASTAR_PLANNING = True
ONLY_PART_I = False

RADIUS_TO_REMOVE_WHEN_UNACCESSIBLE = 4*ROBOT_RADIUS
#ADIUS_TO_REMOVE_WHEN_CLOSE_TO_VISITED= visited
#RADIUS_TO_REMOVE_WHEN_FAILED_SPIRAL = 4*ROBOT_RADIUS

class RandomBAstar2(CPPSolver):
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
            self.ba_exploration = RANDOM_BASTAR_PART_I_COVERAGE
            self.coverage_2 = COVEREAGE_EFFICIENCY_GOAL
            self.min_spiral_coverage = RANDOM_BASTAR_MIN_SPIRAL_LENGTH
            self.min_bastar_coverage = RANDOM_BASTAR_MIN_COVERAGE
            self.max_iterations = RANDOM_BASTAR_MAX_ITERATIONS
            self.step_size = BASTAR_STEP_SIZE
            self.visited_threshold = BASTAR_VISITED_TRESHOLD
        else:
            self.max_distance = parameters["max_distance"]
            self.max_distance_part_II = parameters["max_distance_part_II"]
            self.ba_exploration = parameters["ba_exploration"]
            self.min_spiral_coverage = parameters["min_spiral_coverage"]
            self.min_bastar_coverage = parameters["min_bastar_coverage"]
            self.step_size = parameters["step_size"]
            self.visited_threshold = parameters["visited_threshold"]

        
        self.randombastar_stats = {}
        self.randombastar_stats_over_time = []


    def get_cpp_path(self, start_point, goal_coverage = None):
        """Generates a path that covers the area using BAstar Algorithm.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot

        Returns:
            Nx3 array with waypoints
        """
            
        self.start_tracking()
        self.move_to(start_point)
        self.all_segments = []
        self.visited_waypoints = np.empty((0,3))
        
        self.tmp_coverable_pcd = PointCloud(self.print, points=self.coverable_pcd.points)
        self.explored_pcd = PointCloud(self.print, points=self.coverable_pcd.points)
        self.uncovered_coverable_points_idx = np.arange(len(self.tmp_coverable_pcd.points))
        iter = 0

        #### PART 0 - Border ####    
        self.print("PART 0 - Covering border")
        coverable_pcd = PointCloud(self.print, points=self.coverable_pcd.points)
        border_segment = RandomBorderSegment(self.print, self.motion_planner, start_point, self.visited_waypoints, coverable_pcd, self.max_distance_part_II, self.step_size, self.visited_threshold, self.time_left())
        self.add_segment(border_segment)
        self.explored_pcd.covered_points_idx = np.unique(np.append(self.explored_pcd.covered_points_idx, border_segment.covered_points_idx))

        self.print("Coverage of border: " + str(border_segment.coverage))

        #### PART 1 - BA* ####    
        self.print("PART 1 - Covering with BA*")
        coverage = self.tmp_coverable_pcd.get_coverage_efficiency()        
        exploration = self.explored_pcd.get_coverage_efficiency()        
        while exploration < self.ba_exploration and coverage < goal_coverage and not self.time_limit_reached():
            iter += 1
            #self.print("Uncovered points: " + str(len(self.uncovered_coverable_points_idx)))
            
            random_point = self.get_random_uncovered_point(ignore_list = self.explored_pcd.covered_points_idx)
            
            if random_point is False:
                break
            
            closest_border_point, _ = self.find_closest_border(random_point, self.step_size, self.visited_threshold, self.visited_waypoints)

            BA_segments_from_point = []

            for angle_idx in range(8):

                angle_offset = angle_idx * 2*np.pi/8
                coverable_pcd = PointCloud(self.print, points=self.coverable_pcd.points)
                new_BAstar_path = BAStarSegment(self.print, self.motion_planner, closest_border_point, angle_offset, self.visited_waypoints, coverable_pcd, self.max_distance, self.step_size, self.visited_threshold, self.time_left())
                
                BA_segments_from_point.append(new_BAstar_path)
                
                #if new_BAstar_path.coverage == 0:
                #    break
            accepted_segments = list(filter(lambda x: x.coverage > self.min_bastar_coverage, BA_segments_from_point))
            #self.print([a.coverage for a in accepted_segments])
            if not accepted_segments:
                best_BA_segment = max(BA_segments_from_point, key=operator.attrgetter("coverage"))           
            else:
                #best_BA_segment = max(BA_segments_from_point, key=operator.attrgetter("coverage"))
                costs = [segment.get_cost_per_coverage() for segment in accepted_segments]
                #self.print(costs)
                best_BA_segment_idx = np.argmin(costs)  
                best_BA_segment =  accepted_segments[best_BA_segment_idx]
                self.add_segment(best_BA_segment)
                coverage = self.tmp_coverable_pcd.get_coverage_efficiency()      
                self.print_update(coverage)
                self.randombastar_stats_over_time.append({
                    "time": timeit.default_timer() - self.start_time,
                    "coverage": coverage,
                    "iteration": iter,
                    "path": best_BA_segment.path,
                    "segment": best_BA_segment
                })

            self.print(str(iter) + "- bastar coverage: " + str(best_BA_segment.coverage))
            self.explored_pcd.covered_points_idx = np.unique(np.append(self.explored_pcd.covered_points_idx, best_BA_segment.covered_points_idx))
            exploration = self.explored_pcd.get_coverage_efficiency()
            self.print("exploration: " + str(exploration))
            
            

        self.print("Number of found paths: " + str(len(self.all_segments)))


        self.randombastar_stats["Part1_segments"] = len(self.all_segments)
        self.randombastar_stats["Part1_coverage"] = coverage
        self.randombastar_stats["Part1_iterations"] = iter
        self.randombastar_stats["Part1_time"] = timeit.default_timer() - self.start_time


        #### PART 2 - Inward Spiral ####    
        self.print("PART 2 - Covering with Inward spiral")
        self.explored_pcd.covered_points_idx = self.tmp_coverable_pcd.covered_points_idx
        
        iter = 0
        while coverage < goal_coverage and not self.time_limit_reached(): 
            iter += 1
            #self.print("Uncovered points: " + str(len(self.uncovered_coverable_points_idx)))

            random_point = self.get_random_uncovered_point()
            if random_point is False:
                break

            closest_border_point, _ = self.find_closest_border(random_point, self.step_size, self.visited_threshold, self.visited_waypoints)
            coverable_pcd = PointCloud(self.print, points=self.coverable_pcd.points)
            spiral_segment = RandomSpiralSegment(self.print, self.motion_planner, closest_border_point, self.visited_waypoints, coverable_pcd, self.max_distance_part_II, self.step_size, self.visited_threshold, self.time_left())
            
            self.print(str(iter) + "- spiral coverage: " + str(spiral_segment.coverage))
            
            if spiral_segment.coverage < 0.001:
                close_coverable_points_idx = spiral_segment.covered_points_idx
                if not len(close_coverable_points_idx):
                    close_coverable_points_idx =  self.tmp_coverable_pcd.points_idx_in_radius(closest_border_point, self.visited_threshold)
                self.uncovered_coverable_points_idx = self.delete_values(self.uncovered_coverable_points_idx, close_coverable_points_idx)
                #self.print("Too short spiral")
                continue


            
            self.add_segment(spiral_segment)
            coverage = self.tmp_coverable_pcd.get_coverage_efficiency() 
            self.randombastar_stats_over_time.append({
                    "time": timeit.default_timer() - self.start_time,
                    "coverage": coverage,
                    "iteration": iter,
                    "path": spiral_segment.path,
                    "segment": spiral_segment
                })

            #self.print("length: " + str(len(spiral_segment.path)))
            self.print_update(coverage)
            #self.print("Uncovered points: " + str(len(self.uncovered_coverable_points_idx)))

        self.randombastar_stats["Part2_segments"] = len(self.all_segments) - self.randombastar_stats["Part1_segments"]
        self.randombastar_stats["Part2_coverage"] = coverage
        self.randombastar_stats["Part2_iterations"] = iter
        self.randombastar_stats["Part2_time"] = timeit.default_timer() - self.start_time

        paths_to_visit_in_order  = self.traveling_salesman(self.all_segments)

        self.follow_paths(start_point, paths_to_visit_in_order)

        #self.print_stats(self.path)
        self.print(self.randombastar_stats)

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

        
    def add_segment(self, segment):
        self.all_segments.append(segment)
        self.visited_waypoints = np.append(self.visited_waypoints, segment.path, axis=0)
        self.tmp_coverable_pcd.covered_points_idx = np.unique(np.append(self.tmp_coverable_pcd.covered_points_idx, segment.covered_points_idx))
        self.uncovered_coverable_points_idx = self.delete_values(self.uncovered_coverable_points_idx, segment.covered_points_idx)


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

    def get_random_uncovered_point(self, ignore_list = None, iter = False ):
        """Returns a random uncovered point

        Args:
            visited_waypoints: Nx3 array with waypoints that has been visited
            iter (bool, optional): Integer for random seed. Defaults to False.

        Returns:
            A [x,y,z] position of an unvisited point.
        """
        uncovered_coverable_points_idx = self.uncovered_coverable_points_idx

        if iter is not False:
            np.random.seed(20*iter)

        if ignore_list is not None:
            uncovered_coverable_points_idx = self.delete_values(uncovered_coverable_points_idx, ignore_list)
        while len(uncovered_coverable_points_idx) and not self.time_limit_reached():
            random_idx = np.random.choice(len(uncovered_coverable_points_idx), 1, replace=False)[0]
            random_uncovered_coverable_point_idx = uncovered_coverable_points_idx[random_idx]
            random_uncovered_coverable_point = self.coverable_pcd.points[random_uncovered_coverable_point_idx]

            closest_traversable_point = self.traversable_pcd.find_k_nearest(random_uncovered_coverable_point, 1)[0]
            if self.has_been_visited(closest_traversable_point, self.visited_threshold,  self.visited_waypoints):
                close_coverable_points_idx = self.tmp_coverable_pcd.points_idx_in_radius(random_uncovered_coverable_point, ROBOT_RADIUS)
                self.uncovered_coverable_points_idx = self.delete_values(self.uncovered_coverable_points_idx, close_coverable_points_idx)
                #self.print("Has been visited. Removing " + str(len(close_coverable_points_idx)))
                closest_not_visited = self.find_closest_traversable(closest_traversable_point, self.step_size, self.visited_threshold, self.visited_waypoints, self.step_size*10)
                if closest_not_visited is False:
                    #self.print("BFS could not find an unvisited close")
                    continue 
                return closest_not_visited
                
                continue
            
            if not self.accessible(random_uncovered_coverable_point, self.visited_waypoints):
                close_coverable_points_idx = self.tmp_coverable_pcd.points_idx_in_radius(random_uncovered_coverable_point, ROBOT_RADIUS)
                self.uncovered_coverable_points_idx = self.delete_values(self.uncovered_coverable_points_idx, close_coverable_points_idx)
                #self.print("Inaccessible. Removing " + str(len(close_coverable_points_idx)))
                continue
            
            break

        if len(uncovered_coverable_points_idx) and not self.time_limit_reached():
            return closest_traversable_point
        
        return False




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