from exjobb.Tree import Tree

import numpy as np
import open3d as o3d
import operator
import collections
import timeit
import networkx as nx


from heapq import heappush, heappop
from itertools import count

from networkx.algorithms.shortest_paths.weighted import _weight_function

STEP_SIZE = 0.1
RRT_STEP_SIZE = 3*STEP_SIZE
ASTAR_STEP_SIZE = 5*STEP_SIZE
UNTRAVERSABLE_THRESHHOLD = 8*STEP_SIZE

TRAPPED = 0
ADVANCED = 1
REACHED = 2

#Astar = https://networkx.org/documentation/stable/_modules/networkx/algorithms/shortest_paths/astar.html




class MotionPlanner():

    def __init__(self, logger, ground_pcd):
        self.logger = logger
        self.pcd = ground_pcd

    def RRT(self, start_point, end_point):
        tree_a = Tree(name="start")
        tree_a.add_node(start_point)
        tree_b = Tree(name="goal")
        tree_b.add_node(end_point)

        nbr_of_points_in_pcd = len(self.pcd.points)
        for i in range(10000):
            random_point = self.pcd.points[np.random.randint(nbr_of_points_in_pcd)]
            new_point_1, status = self.extend(tree_a, random_point)

            if status != TRAPPED:
                new_point_2, status = self.extend(tree_b, new_point_1)
                if status == REACHED:
                    #self.logger.info("Path found")
                    path = self.get_shortest_path(tree_a, tree_b, new_point_2)
                    return path
            
            tree_a, tree_b = tree_b, tree_a
        
        #self.logger.warn("Failed to find path using RRT")
        return False

    def extend(self, tree, extension_point):
        nearest_node_idx, nearest_point = tree.nearest_node(extension_point)
        new_point = self.new_point_towards(nearest_point, extension_point, RRT_STEP_SIZE)

        if self.is_valid_step(nearest_point, new_point):
            distance = np.linalg.norm(new_point - nearest_point)
            new_node_idx = tree.add_node(new_point)
            tree.add_edge(nearest_node_idx, new_node_idx, distance)
            if np.array_equal(new_point, extension_point):
                return new_point, REACHED
            else:
                return new_point, ADVANCED
        else:
            return new_point, TRAPPED


    def get_shortest_path(self, tree_1, tree_2, connection_point):
        total_graph = nx.disjoint_union(tree_1.tree, tree_2.tree)
        full_nodes = np.append(tree_1.nodes, tree_2.nodes, axis=0)
        connection_idx_tree_1 = len(tree_1.nodes) - 1
        connection_idx_tree_2, point = tree_2.nearest_node(connection_point)
        connection_idx_tree_2_total_graph = connection_idx_tree_1 + connection_idx_tree_2 + 1        
        total_graph.add_edge(connection_idx_tree_1, connection_idx_tree_2_total_graph, weight=0)

        start_idx = 0
        end_idx = len(tree_1.nodes)

        def dist(a, b):
            a_point = full_nodes[a]
            b_point = full_nodes[b]
            return np.linalg.norm(b_point - a_point)

        path = nx.astar_path(total_graph, start_idx, end_idx, heuristic=dist)
        return np.array([ full_nodes[idx] for idx in path ])
            

    def Astar(self, start_point, end_point):

        self.astar_points = np.array([start_point, end_point])
        start = 0
        target = 1

        #self.print("begining: " + str(self.astar_points))

        def distance(a, b):
            a_point = self.astar_points[a]
            b_point = self.astar_points[b]
            return np.linalg.norm(a_point - b_point)

        push = heappush
        pop = heappop

        c = count()
        queue = [(0, next(c), start, 0, None)]

        enqueued = {}
        explored = {}

        while queue:
            # Pop the smallest item from queue.
            _, __, curnode, dist, parent = pop(queue)

            if curnode == target:
                path = [curnode]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                #self.print(path)
                return self.astar_points[path]

            if curnode in explored:
                # Do not override the parent of starting node
                if explored[curnode] is None:
                    continue

                # Skip bad paths that were enqueued before finding a better one
                qcost, h = enqueued[curnode]
                if qcost < dist:
                    continue

            explored[curnode] = parent
            curpoint = self.astar_points[curnode]
            
            for neighbor, w in self.get_neighbours_for_astar(curnode).items():
                ncost = dist + w
                #self.print("ncost" + str(ncost))
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]

                    #self.print("qcost, h: " + str((qcost, h)))
                    # if qcost <= ncost, a less costly path from the
                    # neighbor to the source was already determined.
                    # Therefore, we won't attempt to push this neighbor
                    # to the queue
                    if qcost <= ncost:
                        continue
                else:
                    h = distance(neighbor, target)

                enqueued[neighbor] = ncost, h
                push(queue, (ncost + h, next(c), neighbor, ncost, curnode))
            
        self.print("No path found using Astar")
        return False

    def get_neighbours_for_astar(self, curnode):
        current_point = self.astar_points[curnode]
        neighbours = {}
        nbr_of_neighbours = 4

        for direction in range(nbr_of_neighbours):
            angle = direction/nbr_of_neighbours*np.pi*2
            x = current_point[0] + np.cos(angle) * ASTAR_STEP_SIZE
            y = current_point[1] + np.sin(angle) * ASTAR_STEP_SIZE
            z = current_point[2]
            pos = np.array([x, y, z])
            
            nearest_point = self.pcd.find_k_nearest(pos, 1)[0]
            if self.is_valid_step(current_point, nearest_point):
                
                node = self.get_node(nearest_point)

                if node is False:
                    node = len(self.astar_points)
                    self.astar_points = np.append(self.astar_points, [nearest_point], axis=0)

                if node == curnode:
                    continue

                neighbours[node] = np.linalg.norm(current_point - nearest_point)

        return neighbours


    def get_node(self, nearest_point):
        distance_to_existing_nodes = np.linalg.norm(self.astar_points - nearest_point, axis=1)
        #self.print("distances: " + str(distance_to_existing_nodes))
        closest_node = np.argmin(distance_to_existing_nodes)
        distance = distance_to_existing_nodes[closest_node]
        #self.print("closest: " + str(closest_node) + " at node " + str(distance))
        if distance < 0.8 * ASTAR_STEP_SIZE  :
            return closest_node
        return False
        
    def new_point_towards(self, start_point, end_point, step_size):
        if np.linalg.norm(end_point - start_point) < step_size:
            return end_point
        direction = self.get_direction_vector(start_point, end_point)
        new_pos = start_point + step_size * direction
        return self.pcd.find_k_nearest(new_pos, 1)[0]   

    def get_direction_vector(self, start, goal):
        line_of_sight = goal - start
        return line_of_sight / np.linalg.norm(line_of_sight)
    
    def is_valid_step(self, from_point, to_point):
        total_step_size = np.linalg.norm(to_point - from_point)
        if total_step_size <= STEP_SIZE:
            return True

        nbr_of_steps = int(np.floor(total_step_size / STEP_SIZE))

         
        prev_point = from_point
        for step in range(nbr_of_steps):
            end_pos = prev_point + self.get_direction_vector(prev_point, to_point) * STEP_SIZE
            new_point = self.new_point_towards(prev_point, end_pos, STEP_SIZE)

            if np.linalg.norm(new_point - prev_point) > UNTRAVERSABLE_THRESHHOLD:
                return False

            if np.array_equal(new_point, to_point):
                return True

            prev_point = new_point

        return True
    
    def print(self, object_to_print):
        self.logger.info(str(object_to_print))