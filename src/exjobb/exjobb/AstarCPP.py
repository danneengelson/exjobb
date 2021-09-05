from networkx.classes.function import neighbors
import numpy as np
from heapq import heappush, heappop
from itertools import count
import timeit
import csv

from numpy.lib.function_base import cov

from exjobb.CPPSolver import CPPSolver 
from exjobb.Parameters import BASTAR_STEP_SIZE, BASTAR_VISITED_TRESHOLD, COVEREAGE_EFFICIENCY_GOAL, ASTAR_STEP_SIZE, ROBOT_RADIUS
from exjobb.PointCloud import PointCloud

GOAL_COVERAGE = 0.97


class AstarCPP(CPPSolver):
    ''' Solving the Coverage Path Planning Problem with BAstar
    '''
    def __init__(self, print, motion_planner, coverable_pcd, time_limit=None):
        '''
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        '''
        super().__init__(print, motion_planner, coverable_pcd, time_limit)
        self.name = "Optimal"
        self.step_size = BASTAR_STEP_SIZE
        self.visited_threshold = BASTAR_VISITED_TRESHOLD
        


    def get_coverage(self, curr_node, parent=None):
        if parent is None:
            parent = self.parents[curr_node]
        current_path = self.get_full_path_idx(curr_node, parent)
        self.tmp_pcd.visit_path(self.astar_points[current_path])
        coverage = self.tmp_pcd.get_coverage_efficiency()
        self.tmp_pcd.covered_points_idx = np.array([])
        return coverage

    def goal_reached(self, curr_node, parent):
        #return False
        if curr_node == 0:
            return False
        coverage = self.get_coverage(curr_node, parent)
        #self.print(coverage)
        return coverage > GOAL_COVERAGE

    def get_full_path_idx(self, curr_node, parent=None):
        if parent is None:
            parent = self.parents[curr_node]
        #self.print(self.parents)
        path = [curr_node]
        node = parent
        while node is not None:
            #self.print(node)
            path.append(node)
            node = self.parents[node]
        path.reverse()
        return path

    def cost(self, from_node, to_node):

        

        def distance(a, b):
            a_point = self.astar_points[a]
            b_point = self.astar_points[b]
            return np.linalg.norm(a_point - b_point)

        def rotation(a,b):

            point_parent_a = self.astar_points[self.parents[a]]
            point_a = self.astar_points[a]
            point_b = self.astar_points[b]

            prev = (point_a - point_parent_a) / np.linalg.norm(point_a - point_parent_a)
            next = (point_b - point_a) / np.linalg.norm(point_b - point_a)
            dot_product = np.dot(prev, next)
            curr_rotation = np.arccos(dot_product)
            if not np.isnan(curr_rotation):
                return abs(curr_rotation)
            return 0

        def visited(a, b):
            current_path = self.get_full_path_idx(a, self.parents[a])
            if self.has_been_visited(b, path=self.astar_points[current_path]):
                return 1
            return 0

        if self.parents[from_node] is None:
            #return 10**6
            return distance(from_node, to_node)  

        #coverage_cost = 0
        #coverage = self.get_coverage(to_node, from_node)
        #if coverage < 0.95:
        #    
        #    coverage_cost = (1- coverage) * len(self.coverable_pcd.points) * 0.1
        #    self.print(coverage_cost)
        #    return coverage_cost

    
        #return rotation(from_node, to_node)
        return distance(from_node, to_node) + rotation(from_node, to_node) #+ coverage_cost

    def heuristic(self, curr_node, parent):
        #return 0
        coverage = self.get_coverage(curr_node, parent)
        if coverage < GOAL_COVERAGE:
            return (1- coverage) * len(self.coverable_pcd.points) * 0.1
        return 0

    def get_cpp_path(self, start_point):
        self.data = []
        self.start_tracking()
        coverage = 0
        self.move_to(start_point)

        #end_point = start_point + np.array([-9,4,0])
        self.astar_points = np.array([start_point])#, end_point])

        start_node = 0
        target_node = 1

        push = heappush
        pop = heappop

        c = count()
        queue = [(np.Inf, next(c), start_node, 0, None)]

        enqueued = {}
        self.parents = {}
        self.tmp_pcd = PointCloud(self.print, points=self.coverable_pcd.points)

        iter = 0
        lowest_f = np.Inf
        best_node = 0
        best_node_parent = None
        self.parents[0] = None
        start_printing = False

        while queue:
            # Pop the smallest item from queue.
            iter += 1
            #self.print("\n\queue: " + str(queue))
            #self.print(self.parents)
            f, __, curnode, g_cost, parent = pop(queue)
            
            if start_printing:
                self.print("\n\niter: " + str(iter))
                self.print("f: " + str(f))
                self.print("curnode: " + str(curnode))
                self.print("g_cost: " + str(g_cost))
                self.print("parent: " + str(parent))
                self.print("point: " + str(self.astar_points[curnode]))
        
            
            if f < lowest_f:
                #self.print("New lowest!")
                #path = self.get_full_path_idx(best_node, best_node_parent)
                #coverage = self.get_coverage(best_node, best_node_parent)
                #self.print("prev path: " + str(path))
                #self.print("prev coverage: " + str(coverage))
                #self.print("prev lowest_f: " + str(lowest_f))
                #self.print("new lowest_f: " + str(f))
                #self.print("\n\niter: " + str(iter))
                #self.print("f: " + str(f))
                #self.print("curnode: " + str(curnode))
                #self.print("g_cost: " + str(g_cost))
                #self.print("parent: " + str(parent))
                #self.print("point: " + str(self.astar_points[curnode]))
                lowest_f = f
                best_node = curnode
                best_node_parent = parent

                best_path = self.get_full_path_idx(best_node, best_node_parent)
                #coverage = self.get_coverage(best_node, best_node_parent)
                #self.print("new path: " + str(best_path))
                #self.print("new coverage: " + str(coverage))
                #if coverage > 0.7:
                #    start_printing = True
                
                


            if iter % 50 == 0:
                path = self.get_full_path_idx(best_node, best_node_parent)
                coverage = self.get_coverage(best_node, best_node_parent)
                self.print("coverage: " + str(coverage))
                self.save_results(self.astar_points[path], coverage, queue)
            
            if iter % 50 == 0:
                self.print("iter: " + str(iter) + " open_set: " + str(len(queue)) + ", visited: " + str(len(self.parents)))


            if self.goal_reached(curnode, parent) or iter > 20000:
                #path = self.get_full_path_idx(best_node, best_node_parent)
                #self.path = self.astar_points[path]
                self.print("GOAL REACHED")
                self.print(best_path)
                path = self.astar_points[best_path]
                self.print(path)
                self.follow_path(path)
                self.print_stats(self.path)
                return self.path

            #if curnode in self.parents:
                #self.print("curnode in parens")
                # Do not override the parent of starting node
                #if self.parents[curnode] is None:
                #    continue

                # Skip bad paths that were enqueued before finding a better one
                #g, h = enqueued[curnode]
                #self.print("Previously:")
                #self.print("g: " + str(g))
                #self.print("h: " + str(h))
                #path = self.get_full_path_idx(curnode, self.parents[curnode])
                #coverage = self.get_coverage(curnode, self.parents[curnode])
                #self.print("Previously path: " + str(path))
                #self.print("Previously coverage: " + str(coverage))

                #if g <= g_cost:
                    #self.print("nevermind " + str(g_cost))
                #    continue

                #self.print("Change of parent! to " + str(parent))
                #path = self.get_full_path_idx(curnode, parent)
                #coverage = self.get_coverage(curnode, parent)
                #self.print("Change path: " + str(path))
                #self.print("Change coverage: " + str(coverage))


            #self.parents[curnode] = parent
            #self.print(self.get_neighbours_for_astar(curnode).items())
            for neighbor, w in self.get_neighbours_for_astar(curnode).items():
                #self.print((neighbor, w))
                ncost = g_cost + w
                h = self.heuristic(neighbor, curnode)
                #if neighbor in enqueued:
                #    g, h_old = enqueued[neighbor]
                #    self.print("in enqued!")

                    #self.print("g, h: " + str((g, h)))
                    # if g <= ncost, a less costly path from the
                    # neighbor to the source was already determined.
                    # Therefore, we won't attempt to push this neighbor
                    # to the queue
                    #if g+h_old <= ncost + h :
                    #    #self.print("nevermind")
                    #    continue
                
                    

                #
                #Added:


                if neighbor != 0:
                    self.parents[neighbor] = curnode
                    #enqueued[neighbor] = ncost, h
                    push(queue, (ncost + h, next(c), neighbor, ncost, curnode))
        self.print("END REACHED")
        #path = self.get_full_path_idx(best_node, best_node_parent)
        self.print(best_path)
        self.follow_path(self.astar_points[best_path])
        self.print_stats(self.path)

        return self.path

    def get_neighbours_for_astar(self, curr_node):
        ''' Help function for Astar to find neighbours to a given node 
        Args:
            curr_node: Node as index in self.astar_points
        Returns:
            Valid neighbours to curr_node as a dictionary, with index in
            self.astar_points as key and distance to given node as value. 
        '''
        current_point = self.astar_points[curr_node]
        neighbours = {}
        nbr_of_neighbours = 8

        for direction in range(nbr_of_neighbours):
            angle = direction/nbr_of_neighbours*np.pi*2
            x = current_point[0] + np.cos(angle) * self.step_size
            y = current_point[1] + np.sin(angle) * self.step_size
            z = current_point[2]
            pos = np.array([x, y, z])
            
            nearest_point = self.traversable_pcd.find_k_nearest(pos, 1)[0]

            nbr_of_astar_points = len(self.astar_points)
            new_neighbours = []

            if any(np.equal(self.astar_points, nearest_point).all(1)):
                continue

            if self.motion_planner.is_valid_step(current_point, nearest_point):
            #if self.parents.get(curr_node):
            #    node_path = self.get_full_path_idx(curr_node)
            #    current_path = self.astar_points[node_path]
            #else:
            #    self.print("hej" + str(curr_node))
            #    current_path = np.array([current_point])

            #if not self.is_blocked(current_point, nearest_point, self.visited_threshold, current_path):
                
                node = self.get_node(nearest_point, nbr_of_astar_points)

                

                if node is False:
                    node = len(self.astar_points) + len(new_neighbours)
                    self.astar_points = np.append(self.astar_points, [nearest_point], axis=0)
                    #new_neighbours.append(nearest_point)

                if node == curr_node:
                    continue

                neighbours[node] = self.cost(curr_node, node)
        
        #for point in new_neighbours:
        #    self.astar_points = np.append(self.astar_points, [point], axis=0)
        #self.print(len(neighbours))
        return neighbours

    def get_long_neighbours_for_astar(self, curr_node):
        ''' Help function for Astar to find neighbours to a given node 
        Args:
            curr_node: Node as index in self.astar_points
        Returns:
            Valid neighbours to curr_node as a dictionary, with index in
            self.astar_points as key and distance to given node as value. 
        '''
        current_point = self.astar_points[curr_node]
        neighbours = {}
        nbr_of_neighbours = 8
        

        if self.parents.get(curr_node):
            node_path = self.get_full_path_idx(curr_node)
            current_path = self.astar_points[node_path]
        else:
            self.print("hej" + str(curr_node))
            current_path = np.array([current_point])

        self.print("current_point" + str(current_point))

        for direction in range(nbr_of_neighbours):
            
            angle = direction/nbr_of_neighbours*np.pi*2
            direction_vector = []
            x = np.cos(angle) * self.step_size
            y = np.sin(angle) * self.step_size
            z = 0
            direction_vector = np.array([x, y, z])
            self.print("direction_vector: " + str(direction_vector))
            pos = current_point + direction_vector
            new_neighbour_point = self.traversable_pcd.find_k_nearest(pos, 1)[0]
            

            valid_neighbour = False
            while not self.is_blocked(current_point, new_neighbour_point, self.visited_threshold, current_path):
                neighbour_point = new_neighbour_point
                valid_neighbour = True
                pos = pos + direction_vector
                new_neighbour_point = self.traversable_pcd.find_k_nearest(pos, 1)[0]
                if np.array_equal(neighbour_point, new_neighbour_point):
                    break
                
            

            if valid_neighbour:
                self.print("neighbour_point" + str(neighbour_point))
                if any(np.equal(self.astar_points, neighbour_point).all(1)):
                    continue
                
                node = len(self.astar_points)
                self.astar_points = np.append(self.astar_points, [neighbour_point], axis=0)
                neighbours[node] = self.cost(curr_node, node)
                self.print("is valid! " + str(node))

    
        #self.print("astar_points: " + str(self.astar_points))
        return neighbours


    def get_node(self, point, nbr_of_astar_points):
        ''' Finds the node (index in self.astar_points) that is close to a given point.
        Args:
            point: A [x,y,z] array with the position of the point
        Returns:
            Index of the closest point in self.astar_points or False if no point found nearby.
        '''
        return False
        distance_to_existing_nodes = np.linalg.norm(self.astar_points[0:nbr_of_astar_points] - point, axis=1)
        closest_node = np.argmin(distance_to_existing_nodes)
        distance = distance_to_existing_nodes[closest_node]
        if distance < self.visited_threshold*0.15:
            return closest_node

        return False
        
    def save_results(self, path, coverage, open_set):

        def get_length_of_path(path):
            ''' Calculates length of the path in meters
            '''
            length = 0
            for point_idx in range(len(path) - 1):
                length += np.linalg.norm( path[point_idx] - path[point_idx + 1] )
            return length

        def get_total_rotation(path):
            ''' Calculates the total rotation made by the robot while executing the path
            '''
            rotation = 0

            for point_idx in range(len(path) - 2):
                prev = (path[point_idx+1] - path[point_idx]) / np.linalg.norm( path[point_idx] - path[point_idx + 1])
                next = (path[point_idx+2] - path[point_idx+1]) / np.linalg.norm( path[point_idx+2] - path[point_idx + 1])
                dot_product = np.dot(prev, next)
                curr_rotation = np.arccos(dot_product)
                if not np.isnan(curr_rotation):
                    rotation += abs(curr_rotation)

            return rotation

        frontier = len(open_set)
        visited = len(self.parents)
        current_time = timeit.default_timer() - self.start_time
        current_path_length = len(self.path)
        self.data.append({
            "algorithm": "Optimal",
            "time": round(current_time, 1),
            "frontier": frontier,
            "visited": visited,
            "length": get_length_of_path(path),
            "rotation": get_total_rotation(path),
            "coverage": round(coverage*100, 2)
        })

        with open('optimal_path.csv', 'w', newline='') as csvfile:
            fieldnames = ['algorithm', 'time', "coverage", "length", "rotation", "frontier", "visited"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.data:
                writer.writerow(result)

    