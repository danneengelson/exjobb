from exjobb.Tree import Tree
import numpy as np
from collections import deque
import networkx as nx

from exjobb.CPPSolver import CPPSolver, ROBOT_RADIUS, STEP_SIZE

RRT_STEP_SIZE = 3*STEP_SIZE
COVEREAGE_EFFICIENCY_GOAL = 0.8
MAX_ITERATIONS = 10000
GOAL_CHECK_FREQUENCY = 50

TRAPPED = 0
ADVANCED = 1
REACHED = 2

class NaiveRRTCPPDFS(CPPSolver):

    def __init__(self, logger, ground_pcd):
        self.name = "Naive RRT CPP with DFS"
        super().__init__(logger, ground_pcd)

    def get_cpp_path(self, start_point):
        self.start_tracking()
        tree = self.build_RRT_tree(start_point)
        path = self.find_path_through_tree(tree)
        self.print_stats(path)
        return path

    def find_path_through_tree(self, tree):
       
        nbr_of_nodes = len(tree.nodes)

        def neighbors(node):
            return [n for n in tree.tree.neighbors(node)]

        start_node = 0
        visited = [start_node]
        queue = deque([(start_node, neighbors(start_node))])
        last_node = False
        path = np.array([], dtype=int)
        while queue:

            if len(visited) ==  nbr_of_nodes:
                return tree.nodes[path]

            parent, children = queue[-1]
            path = np.append(path, parent)     
            
            child_found = False
            for child in children:
                if child not in visited:
                    visited.append(child)
                    queue.append((child, neighbors(child)))
                    child_found = True
                    break

            if not child_found:                               
                queue.pop()

            

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
            
            self.pcd.visit_point(new_point_1, ROBOT_RADIUS)

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
        new_point = self.new_point_towards(nearest_point, extension_point, RRT_STEP_SIZE)
        #self.logger.info(str(np.linalg.norm(new_point - nearest_point)))

        if self.is_valid_step(nearest_point, new_point):
            distance = np.linalg.norm(new_point - nearest_point)
            new_node_idx = tree.add_node(new_point)
            tree.add_edge(nearest_node_idx, new_node_idx, distance)
            #self.logger.info("New: " + str(new_node_idx) + ": " + str(new_point) + ", dist: " + str(distance))
            #self.logger.info("New in tree: " + str(new_node_idx) + ": " + str(tree.nodes[new_node_idx]))
            return new_point, ADVANCED

        else:
            return new_point, TRAPPED