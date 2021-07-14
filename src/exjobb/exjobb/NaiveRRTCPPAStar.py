from networkx.algorithms.shortest_paths.dense import reconstruct_path
from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx

from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE

RRT_STEP_SIZE = 3*STEP_SIZE
COVEREAGE_EFFICIENCY_GOAL = 0.4
MAX_ITERATIONS = 10000
GOAL_CHECK_FREQUENCY = 50

ASTAR_FLAG = -1

TRAPPED = 0
ADVANCED = 1
REACHED = 2

class NaiveRRTCPPAstar(CPPSolver):

    def __init__(self, logger, ground_pcd):
        super().__init__(logger, ground_pcd)
        self.name = "Naive RRT CPP with RRT Motion Planning Between Points"

    def get_cpp_path(self, start_point):
        self.start_tracking()
        tree = self.build_RRT_tree(start_point)
        self.print("hej")
        self.path = self.find_path_through_tree2(tree)
        self.print(self.path )
        self.points_to_mark = [start_point]
        self.print_stats(self.path)
        return self.path

    def find_path_through_tree(self, tree):
       
        nbr_of_nodes = len(tree.nodes)

        def neighbors(node):
            return [n for n in tree.tree.neighbors(node)]

        start_node = 0
        visited = [start_node]
        queue = deque([(start_node, neighbors(start_node))])
        last_node = False
        connected = True  
        path = np.array([start_node], dtype=int)

        while queue:

            if len(visited) ==  nbr_of_nodes:
                astar_expected = np.where(path == ASTAR_FLAG)
                path_points = np.zeros( (len(path), 300, 3) )
                path_points[:,0] =  tree.nodes[path]
                for unconnected_idx in astar_expected[0]:
                    astar_path = self.motion_planner.RRT(tree.nodes[path[unconnected_idx-1]], tree.nodes[path[unconnected_idx+1]])
                    if astar_path is False:
                        #self.print("path not found between " + str(tree.nodes[path[unconnected_idx-1]]) + " and " + str(tree.nodes[path[unconnected_idx+1]]))
                        astar_path = np.array([tree.nodes[path[unconnected_idx-1]], tree.nodes[path[unconnected_idx+1]]])
                    
                    path_points[unconnected_idx, 0:len(astar_path)] = astar_path
                path_points = np.reshape(path_points, (-1, 3))
                path_points = path_points[ ~np.all(path_points == 0, axis=1) ]

                return path_points

            parent, children = queue[-1]
            
            child_found = False
            for child in children:
                if child not in visited:
                    visited.append(child)
                    queue.append((child, neighbors(child)))
                    child_found = True

                    if not connected:
                        path = np.append(path, ASTAR_FLAG)
                        connected = True

                    path = np.append(path, child)
                    break

            if not child_found:                               
                queue.pop()
                connected = False

    def find_path_through_tree2(self, tree):
       
        def neighbors(node):
            return [n for n in tree.tree.neighbors(node)]

        start_node = 0
        visited = np.array([start_node])
        queue = np.array([start_node])
        path = np.empty((0,3))

        while len(queue) > 0:
            current, queue = queue[-1], queue[0:-1]
            #current, queue = queue[0], queue[1:]
            visited = np.append(visited, current)
            for neighbour in neighbors(current):
                if neighbour not in visited:
                    queue = np.append(queue, neighbour)

        prev_node = start_node
        self.print(visited)
        for idx, node in enumerate(visited[2:]):
            self.print("Visiting " + str(idx) + " out of " + str(len(visited)-2))
            path_to_node = self.motion_planner.Astar(tree.nodes[prev_node], tree.nodes[node])
            path = np.append(path, path_to_node, axis=0)
            #if node in neighbors(prev_node):
            #    path = np.append(path, [tree.nodes[node]], axis=0)
            #else: 
            #    path_to_node = self.motion_planner.Astar(tree.nodes[prev_node], tree.nodes[node])
            #    path = np.append(path, path_to_node, axis=0)

            prev_node = node

        return path


    def build_RRT_tree(self, start_point):
        tree = Tree()
        tree.add_node(start_point)

        nbr_of_points_in_pcd = len(self.pcd.points)
        for i in range(MAX_ITERATIONS):
            random_point = self.pcd.points[np.random.randint(nbr_of_points_in_pcd)]
            new_point_1, status = self.extend(tree, random_point)
            #self.logger.info("status: " + str(status))
            if status == TRAPPED:
                continue
            
            self.pcd.visit_only_point(new_point_1, ROBOT_RADIUS)

            if i % GOAL_CHECK_FREQUENCY == 0:
                coverage = self.pcd.get_coverage_efficiency()
                self.logger.info("Coverage: " + str(round(coverage*100, 2)) + "%")
                if coverage > COVEREAGE_EFFICIENCY_GOAL:
                    self.logger.info("Coverage reached")
                    
                    return tree
        
        self.logger.warn("Failed to cover")
        return tree

    def extend(self, tree, extension_point):
        nearest_node_idx, nearest_point = tree.nearest_node(extension_point)
        new_point = self.motion_planner.new_point_towards(nearest_point, extension_point, RRT_STEP_SIZE)
        #self.logger.info(str(np.linalg.norm(new_point - nearest_point)))

        if self.motion_planner.is_valid_step(nearest_point, new_point):
            distance = np.linalg.norm(new_point - nearest_point)
            new_node_idx = tree.add_node(new_point)
            tree.add_edge(nearest_node_idx, new_node_idx, distance)
            #self.logger.info("New: " + str(new_node_idx) + ": " + str(new_point) + ", dist: " + str(distance))
            #self.logger.info("New in tree: " + str(new_node_idx) + ": " + str(tree.nodes[new_node_idx]))
            return new_point, ADVANCED

        else:
            return new_point, TRAPPED


    
    def get_points_to_mark(self):
        #return self.backtrack_list
        return self.points_to_mark