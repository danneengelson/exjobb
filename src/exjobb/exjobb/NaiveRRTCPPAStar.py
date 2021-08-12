import numpy as np

from exjobb.Tree import Tree
from exjobb.CPPSolver import CPPSolver
from exjobb.PointCloud import PointCloud
from exjobb.MotionPlanner import TRAPPED
from exjobb.Parameters import COVEREAGE_EFFICIENCY_GOAL, NAIVE_RRT_CPP_MAX_ITERATIONS, NAIVE_RRT_CPP_GOAL_CHECK_FREQUENCY


class NaiveRRTCPPAstar(CPPSolver):
    """ Implementation of the Naive RRT Coverage Path Planning Algorithm
    """

    def __init__(self, print, motion_planner, coverable_pcd):
        """
        Args:
            print: function for printing messages
            motion_planner: Motion Planner of the robot wihch also has the Point Cloud
        """
        super().__init__(print, motion_planner, coverable_pcd)
        self.name = "Naive RRT CPP with Deep First Search"

    def get_cpp_path(self, start_point):
        """Generates a path that covers the area using Naive RRT.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot

        Returns:
            Nx3 array with waypoints
        """

        self.start_tracking()
        self.move_to(start_point)
        tree = self.build_RRT_tree(start_point)
        path = self.find_path_through_tree(tree)
        self.print_stats(path)
        return path


    def build_RRT_tree(self, start_point):
        ''' Builds up a tree. Simply picks a random traversable point 
        and expands the tree towards the random point. This is done until 
        the desired coverage or a maximum amount of iterations has been reached.

        Args:
            start_point: Start position of the robot

        Returns:
            The generated Tree
        '''
        tree = Tree()
        tree.add_node(start_point)
        tmp_coverable_pcd = PointCloud(self.print, points=self.coverable_pcd.points)
        
        nbr_of_points_in_traversable_pcd = len(self.traversable_pcd.points)

        def get_random_point():
            return self.traversable_pcd.points[np.random.randint(nbr_of_points_in_traversable_pcd)]

        for i in range(NAIVE_RRT_CPP_MAX_ITERATIONS):
            random_point = get_random_point()
            new_point, status = self.motion_planner.extend(tree, random_point)

            if status == TRAPPED:
                continue
            
            tmp_coverable_pcd.visit_position(new_point, apply_unique=True)

            if i % NAIVE_RRT_CPP_GOAL_CHECK_FREQUENCY == 0:
                coverage = tmp_coverable_pcd.get_coverage_efficiency()
                self.print("Coverage: " + str(round(coverage*100, 2)) + "%")
                if coverage > COVEREAGE_EFFICIENCY_GOAL:
                    self.print("Coverage reached")
                    
                    return tree
        
        self.print("Failed to cover")
        return tree

    def find_path_through_tree(self, tree):
        """Generate a path through all nodes in tree using Deep First Search
        and then creates paths between points using Astar.

        Args:
            tree: Tree that should be covered
        """ 
       
        def neighbors(node):
            return [n for n in tree.tree.neighbors(node)]

        start_node = 0
        visited = np.array([start_node])
        queue = np.array([start_node])

        while len(queue) > 0:
            current, queue = queue[-1], queue[0:-1]
            visited = np.append(visited, current)
            for neighbour in neighbors(current):
                if neighbour not in visited:
                    queue = np.append(queue, neighbour)

        prev_node = start_node

        for idx, node in enumerate(visited[2:]):
            if idx % 1000 == 0:
                self.print("Visiting " + str(idx) + " out of " + str(len(visited)-2))
            path_to_node = self.motion_planner.Astar(tree.nodes[prev_node], tree.nodes[node])
            self.follow_path(path_to_node)
            prev_node = node
        
        return self.path
    
    