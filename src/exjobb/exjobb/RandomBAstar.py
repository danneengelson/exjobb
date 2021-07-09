from exjobb.Parameters import ROBOT_SIZE
from exjobb.PointCloud import PointCloud
from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx
import timeit
from numba import njit
import operator
import copy 
from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE
from exjobb.MotionPlanner import MotionPlanner
from exjobb.RandomBAStarPath import PlanePath
from exjobb.Spiral import Spiral
from exjobb.RandomSpriralPath import SpiralPath
import pickle

CELL_STEP_SIZE = STEP_SIZE #1.25*ROBOT_RADIUS
VISITED_TRESHOLD = 0.5*STEP_SIZE #0.8*ROBOT_RADIUS
COVEREAGE_EFFICIENCY_GOAL = 0.95
RRT_STEP_SIZE = 0.5*STEP_SIZE
RRT_COVEREAGE_EFFICIENCY_GOAL = 0.3
MAX_ITERATIONS = 30
GOAL_CHECK_FREQUENCY = 50


NUMBER_OF_ANGLES = 8

POINTS_FOR_BACKTRACK = 50

DO_BASTAR_PLANNING = False

TRAPPED = 0
ADVANCED = 1
REACHED = 2


class RandomBAstar(CPPSolver):
    def __init__(self, logger, motion_planner):
            
        self.logger = logger
        super().__init__(logger, motion_planner)
        self.name = "Random Sample"

    def get_cpp_path(self, start_point):
        self.start_tracking()
        coverage = 0
        self.path = np.array([start_point])
        self.points_to_mark = np.array([start_point])
        self.move_to(start_point)

        current_position = start_point

        if DO_BASTAR_PLANNING:
            BAstar_paths = []
            total_points = len(self.pcd.points)
            unvisited_points = np.arange(total_points)

            i = -1
            visited_waypoints_part_I = np.empty((0,3))

            while len(unvisited_points)/total_points > 0.05 and i <= MAX_ITERATIONS:
                i += 1
                #self.print(i)
                random_point = self.get_random_unvisited_point(i, visited_waypoints_part_I)
                BAstar_paths_from_this_point = []

                for angle_idx in range(NUMBER_OF_ANGLES):
                    angle_offset = angle_idx * np.pi/NUMBER_OF_ANGLES
                    new_BAstar_path = PlanePath(self.print, self.motion_planner, random_point, angle_offset, visited_waypoints_part_I)
                    #new_BAstar_path = SpiralPath(self.print, self.motion_planner, random_point, visited_waypoints_part_I)
                    BAstar_paths_from_this_point.append(new_BAstar_path)
                    #self.print(str(i) + " - coverage for " + str(angle_idx) +": " + str(new_BAstar_path.coverage))
                    if new_BAstar_path.coverage == 0:
                        break

                best_path = max(BAstar_paths_from_this_point, key=operator.attrgetter("coverage"))
                self.print(str(i) + "- coverage: " + str(best_path.coverage))
                
                if best_path.coverage > 0.01:
                    BAstar_paths.append( best_path )
                    #self.print(best_path.path)
                    #self.print(visited_waypoints)
                    #visited_waypoints = np.append(visited_waypoints, best_path.path, axis=0)
                    visited_waypoints_part_I = np.append(visited_waypoints_part_I, best_path.path, axis=0)
                    unvisited_points = self.delete_values(unvisited_points, best_path.visited_points_idx)
                    self.print("total coverage left: " + str(len(unvisited_points)/ total_points))

            
            self.print("Number of found paths: " + str(len(BAstar_paths)))
            start_points, end_points, sorted_paths, visited_points_idx = self.get_greedy_sorted_paths(BAstar_paths)
            
            visited_waypoints = np.empty((0,3))
            for sorted_path in sorted_paths:
                #self.print(sorted_path.path)
                #self.print(visited_waypoints)
                visited_waypoints = np.append(visited_waypoints, sorted_path.path, axis=0)
                

            with open('cached_visited_points_idx.dictionary', 'wb') as cached_pcd_file:
                cache_data = {  "visited_points_idx": visited_points_idx, 
                                "visited_waypoints": visited_waypoints,
                                "start_points": start_points,
                                "end_points": end_points,
                                "sorted_paths": sorted_paths,
                                }
                pickle.dump(cache_data, cached_pcd_file)
        else:
            with open('cached_visited_points_idx.dictionary', 'rb') as cached_pcd_file:
                cache_data = pickle.load(cached_pcd_file)
                visited_points_idx = cache_data["visited_points_idx"]
                visited_waypoints = cache_data["visited_waypoints"]
                start_points = cache_data["start_points"]
                end_points = cache_data["end_points"]
                sorted_paths = cache_data["sorted_paths"]
                self.pcd.visited_points_idx = visited_points_idx
                #sorted_paths = []


        #random_points = self.get_random_sampled_points_simple(visited_points_idx)
        #self.print(len(random_points))
        
        before  = len(visited_waypoints)
        spiral_paths = []
        i = 3
        while len(visited_points_idx) < 0.98*len(self.pcd.points): 
            i += 33
            random_unvisited_point = self.get_random_unvisited_point(i, visited_waypoints)
            spiral_path = SpiralPath(self.print, self.motion_planner, random_unvisited_point, visited_waypoints)
            
            if len(spiral_path.path) < 3:
                continue
            sorted_paths.append(spiral_path)    
            start_points.append(spiral_path.start)
            end_points.append(spiral_path.end)
            visited_waypoints = np.append(visited_waypoints, spiral_path.path, axis=0)
            visited_points_idx = np.unique(np.append(visited_points_idx, spiral_path.visited_points_idx))
            self.print(len(visited_points_idx) / len(self.pcd.points))
        



        #for spiral in spiral_paths:
        #    self.path = np.append(self.path, spiral.path, axis=0)
        #spiral_cpp = Spiral(self.logger, self.motion_planner)
        
        #spiral_cpp.path = visited_waypoints
        
        #self.path = spiral_cpp.get_cpp_path(random_unvisited_point, visited_waypoints) 
        #self.path = self.path[before:]

        


        points_to_visit_in_order = self.traveling_salesman(sorted_paths)

        self.print("points_to_visit_in_order " + str(len(points_to_visit_in_order)))
        self.print("sorted_paths " + str(len(sorted_paths)))
        self.print("end_points " + str(len(end_points)))
        self.print("start_points " + str(len(start_points)))

        self.follow_given_path(current_position, points_to_visit_in_order, start_points, end_points, sorted_paths)
        #self.path = points_to_visit_in_order
        #self.follow_path(self.path)
        self.print_stats(self.path)

        return self.path

    def get_random_sampled_points_simple(self, visited_points_idx):
        pcd = PointCloud(self.print, points=self.motion_planner.pcd.points)
        pcd.visited_points_idx = visited_points_idx
        waypoints = np.empty((0,3))
        uncovered_points_idx = self.delete_values(np.arange(len(pcd.points)), visited_points_idx)
        nbr_of_points_to_cover = len(uncovered_points_idx)
        covered_points_idx = np.array([])
        while len(covered_points_idx) < 0.90*nbr_of_points_to_cover:
            if len(uncovered_points_idx):
                random_idx = np.random.choice(len(uncovered_points_idx), 1, replace=False)[0]
            else:
                break
            
            random_point =  pcd.points[uncovered_points_idx[random_idx]]
            points_nearby = pcd.points_idx_in_radius(random_point, ROBOT_RADIUS)

            nbr_of_points_nearby = 30

            if len(points_nearby) < nbr_of_points_nearby:
                random_points_nearby = points_nearby
            else:
                random_points_nearby = points_nearby[np.random.choice(len(points_nearby), nbr_of_points_nearby, replace=False)]
            #
            #self.print("random_points_nearby" + str(random_points_nearby))
            #self.print("unvisited points_nearby: " + str(len(np.intersect1d(points_nearby, uncovered_points_idx))))
            covered_points = []
            for point_idx in random_points_nearby:
                #self.print("point_idx: " + str(point_idx))
                point = pcd.points[point_idx]
                #self.print("point: " + str(point))
                points_nearby = pcd.points_idx_in_radius(point, ROBOT_RADIUS)
                #self.print("points_nearby: " + str(len(points_nearby)))
                unvisited_points = np.intersect1d(points_nearby, uncovered_points_idx)
                #self.print("unvisited_points: " + str(len(unvisited_points)))
                covered_points.append(len(unvisited_points))
                #covered_points_idx_list.append(unvisited_points)
                #self.print("covered_points: " + str(covered_points))
                #covered_points.append(len(points_nearby))

            best_point_idx = random_points_nearby[np.argmax(covered_points)]
            #self.print("best_point_idx: " + str(best_point_idx))
            #self.print("best: " + str(covered_points[np.argmax(covered_points)]))
            new_waypoint = pcd.points[best_point_idx]
            #self.print("new_waypoint: " + str(new_waypoint))
            waypoints = np.append(waypoints, [new_waypoint], axis=0)
            points_nearby = pcd.points_idx_in_radius(new_waypoint, ROBOT_RADIUS)
            unvisited_points = np.intersect1d(points_nearby, uncovered_points_idx)
            covered_points_idx = np.unique(np.append(covered_points_idx, unvisited_points))
            #self.print("covered_points_idx: " + str(len(covered_points_idx)))
            uncovered_points_idx = self.delete_values(uncovered_points_idx, covered_points_idx)
            #self.print("uncovered_points_idx: " + str(len(uncovered_points_idx)))
            self.print(len(covered_points_idx)  / nbr_of_points_to_cover)
        
        full_visited_points_idx = np.unique(np.append(visited_points_idx, covered_points_idx)) 
        self.print(len(full_visited_points_idx)  / len(pcd.points))
        return waypoints

    def get_random_sampled_points(self, visited_points_idx):
        pcd = PointCloud(self.print, points=self.motion_planner.pcd.points)
        pcd.visited_points_idx = visited_points_idx

        waypoints = np.empty((0,3))
        k = 1
        
        uncovered_points_idx = self.delete_values(np.arange(len(pcd.points)), visited_points_idx)
        nbr_of_points_to_k_cover = len(uncovered_points_idx)
        covered_points_idx = np.array([])
        k_covered_points_idx = np.array([])
        
        while len(k_covered_points_idx) < nbr_of_points_to_k_cover:
            #self.print(len(k_covered_points_idx) / len(self.pcd.points))
            if len(uncovered_points_idx):
                random_idx = np.random.choice(len(uncovered_points_idx), 1, replace=False)[0]
                new_waypoint = pcd.points[uncovered_points_idx[random_idx]]
            else:
                #self.print("ALL POINTS COVERED")
                random_idx = np.random.choice(len(covered_points_idx), 1, replace=False)[0]
                new_waypoint = pcd.points[covered_points_idx[random_idx]]


            

            waypoints = np.append(waypoints, [new_waypoint], axis=0)
            covered_points_idx = np.append(covered_points_idx, pcd.points_idx_in_radius(new_waypoint, ROBOT_RADIUS))
            #self.print("k_covered_points" + str(k_covered_points))
            #self.print("uncovered_points_idx" + str(uncovered_points_idx))
            #self.print("new_waypoint" + str(new_waypoint))
            #self.print("waypoints" + str(waypoints))
            #self.print("all_covered_points_idx" + str(all_covered_points_idx))
            uncovered_points_idx = self.delete_values(uncovered_points_idx, covered_points_idx)
            k_covered_points_idx = np.unique(np.append(k_covered_points_idx, self.get_k_covered_points_idx(covered_points_idx, k)))
            covered_points_idx = self.delete_values_not_unique(covered_points_idx, k_covered_points_idx)
            self.print(len(k_covered_points_idx) / nbr_of_points_to_k_cover)
        
        full_visited_points_idx = np.unique(np.append(visited_points_idx, k_covered_points_idx)) 
        self.print(len(full_visited_points_idx)  / len(pcd.points))
        return waypoints


    def traveling_salesman(self, paths):
        tree = Tree("BAstar paths")
        
        
        def get_weight(from_idx, to_idx):
            from_point = tree.nodes[from_idx]
            to_point = tree.nodes[to_idx]
            return 100 + np.linalg.norm( to_point[0:2] - from_point[0:2] ) + 10 * abs( to_point[2] - from_point[2] )

        for path in paths:
            start_point = path.start
            end_point = path.end
            start_point_node_idx = tree.add_node(start_point)
            for node_idx, point in enumerate(tree.nodes[:-1]):
                tree.add_edge(start_point_node_idx, node_idx, get_weight(start_point_node_idx, node_idx))
            
            end_point_node_idx = tree.add_node(end_point)
            for node_idx, point in enumerate(tree.nodes[:-2]):
                tree.add_edge(end_point_node_idx, node_idx, get_weight(end_point_node_idx, node_idx))

            tree.add_edge(start_point_node_idx, end_point_node_idx, 0)

        #for point in random_points:
        #    new_point_node_idx = tree.add_node(point)
        #    for node_idx, point in enumerate(tree.nodes[:-1]):
        #        tree.add_edge(new_point_node_idx, node_idx, get_weight(new_point_node_idx, node_idx))

        traveling_Salesman_path =  tree.get_traveling_salesman_path()
        self.print(traveling_Salesman_path)
        return [tree.nodes[i] for i in traveling_Salesman_path] 

    def get_greedy_sorted_paths(self, BAstar_paths):
        sorted_paths = []
        start_points = []
        end_points = []
        pcd = PointCloud(self.print, points=self.motion_planner.pcd.points)
        prev_coverage = 0
        BAstar_paths_unvisited_points_length = [0]*len(BAstar_paths)
        uncovered_paths = [i for i in BAstar_paths]
        visited_points = np.empty((0,3))

        while len(uncovered_paths):
            for idx, path in enumerate(BAstar_paths):
                BAstar_paths_unvisited_points_length[idx] = len(self.delete_values( path.visited_points_idx, pcd.visited_points_idx ))
            
            max_covering_path_idx = np.argmax(BAstar_paths_unvisited_points_length)
            max_path = BAstar_paths[max_covering_path_idx]
            
            if max_path in uncovered_paths:
                uncovered_paths.remove(max_path)
            else:
                break
            
            current_position = max_path.start
            pruned_path = np.empty((0,3))
            for point in max_path.path:
                #if self.is_blocked(current_position, point, visited_points):
                #    continue

                pruned_path = np.append(pruned_path, [point], axis=0)
                current_position = point

            if len(pruned_path) == 0:
                continue

            max_path.path = pruned_path
            max_path.start = pruned_path[0]
            max_path.end = pruned_path[-1]
            visited_points = np.append(visited_points, pruned_path, axis=0)


            
            pcd.visit_path(pruned_path, ROBOT_RADIUS)
            self.points_to_mark = np.append(self.points_to_mark, [max_path.start], axis=0)

            sorted_paths.append(max_path)
            start_points.append(max_path.start)
            end_points.append(max_path.end)

            coverage = pcd.get_coverage_efficiency()
            self.print("coverage" + str(coverage))
            if coverage - prev_coverage < 0.005:
            #if coverage - prev_coverage == 0:
                break
            prev_coverage = coverage
            #self.path = self.points_to_mark

        return start_points, end_points, sorted_paths, pcd.visited_points_idx

    def follow_given_path(self, start_position, points_to_visit_in_order, start_points, end_points, sorted_paths):
        current_position = start_position
        for idx, point in enumerate(points_to_visit_in_order):

            if np.array_equal(point, current_position):
                continue

            self.print("Moving to point " + str(idx) + " out of " + str(len(points_to_visit_in_order)))
            path_to_next_starting_point = self.motion_planner.Astar(current_position, point)
            self.follow_path(path_to_next_starting_point)

            #self.print("point" + str(point))
            start_point_idx = False
            end_point_idx = False

            try:
                start_point_idx = [np.array_equal(point, start_point) for start_point in start_points].index(True)
            except ValueError:
                start_point_idx = False
            
                try:
                    end_point_idx = [np.array_equal(point, end_point) for end_point in end_points].index(True)
                except ValueError:
                    end_point_idx = False

            if start_point_idx is not False:
                self.follow_path(sorted_paths[start_point_idx].path)
                current_position = sorted_paths[start_point_idx].end
            elif end_point_idx is not False:
                self.follow_path(np.flip(sorted_paths[end_point_idx].path, 0))
                current_position = sorted_paths[end_point_idx].start
            else:
                self.move_to(point)
                current_position = point


    def get_random_unvisited_point(self, i, visited_waypoints):
        all_points_idx = np.arange(len(self.pcd.points))
        np.random.seed(20*i)
        random_idx = np.random.choice(len(all_points_idx), 1, replace=False)[0]
        random_point = self.pcd.points[all_points_idx[random_idx]]
        while self.has_been_visited(random_point, visited_waypoints) or not self.accessible(random_point, visited_waypoints):
            random_idx = np.random.choice(len(all_points_idx), 1, replace=False)[0]
            random_point = self.pcd.points[all_points_idx[random_idx]]

        return random_point

    def accessible(self, point, visited_waypoints):
        if len(visited_waypoints) == 0:
            return True
        closest_point = visited_waypoints[np.argmin(np.linalg.norm(visited_waypoints - point, axis=1))]
        path_to_point = self.motion_planner.RRT(closest_point, point)
        return path_to_point is not False


    def delete_values(self, array, values):
        return array[ np.isin(array, values, assume_unique=True, invert=True) ]

    def delete_values_not_unique(self, array, values):
        return array[ np.isin(array, values, invert=True) ]

    def has_been_visited(self, point, path=None):
        if path is None:
            path = self.path
        distances = np.linalg.norm(path - point, axis=1)
        return np.any(distances <= VISITED_TRESHOLD) 

    def is_blocked(self, from_point, to_point, path = None):
        if path is None:
            path = self.path

        if self.has_been_visited(to_point, path):
            return True
        

        if not self.motion_planner.is_valid_step(from_point, to_point):
            return True
        
        return False


    def get_k_covered_points_idx(self, covered_points_idx, k):
        points_idx, counts = np.unique(covered_points_idx, return_counts=True)
        k_covered_points = points_idx[counts >= k]
        return k_covered_points


    def get_points_to_mark(self):
        #return self.backtrack_list
        return self.points_to_mark