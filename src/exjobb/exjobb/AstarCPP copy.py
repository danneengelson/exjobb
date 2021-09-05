from networkx.classes.function import neighbors
import numpy as np
from heapq import heappush, heappop
from itertools import count

from exjobb.CPPSolver import CPPSolver 
from exjobb.Parameters import BASTAR_STEP_SIZE, BASTAR_VISITED_TRESHOLD, COVEREAGE_EFFICIENCY_GOAL, ASTAR_STEP_SIZE, ROBOT_RADIUS
from exjobb.PointCloud import PointCloud



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


    def goal_reached(self, curr_idx):
        return curr_idx == 1

    def cost(self, from_point_idx, to_point_idx)

    def get_cpp_path(self, start_point):
        self.start_tracking()
        coverage = 0
        self.move_to(start_point)

        end_point = start_point + np.array([-9,4,0])
        self.astar_points = np.array([start_point, end_point])

        start = 0
        target = 1

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

            if self.goal_reached(curnode):
                path = [curnode]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                #astar_spt_path = self.AstarSPT(self.astar_points[path])
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
            
            for neighbor, w in self.get_neighbours_for_astar(curnode).items():
                ncost = dist + w
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




    def get_cpp_path2(self, start_point):
        """Generates a path that covers the area using BAstar Algorithm.

        Args:
            start_point: A [x,y,z] np.array of the start position of the robot

        Returns:
            Nx3 array with waypoints
        """
        
        self.start_tracking()
        coverage = 0
        self.move_to(start_point)

        #self.move_to(start_point + np.array([3,0,0]))
        goal_point = start_point + np.array([-9,4,0])
        #return self.path

        push = heappush
        pop = heappop

        c = count()
        self.points = {}
        self.points[0] = np.array(start_point)
        self.points[1] = goal_point

        start_idx = 0
        # f, node_idx
        open_set = [(0, start_idx)]
        prev_point = {}
        g_cheapest_cost_to_start = {}
        f_best_final_cost_guess = {}

        prev_point[start_idx] = start_idx
        g_cheapest_cost_to_start[start_idx] = 0
        f_best_final_cost_guess[start_idx] = 0
        
        def get_rotation(a,b):
            prev = (self.points[a] - self.points[prev_point[a]]) / np.linalg.norm( self.points[prev_point[a]] - self.points[a])
            next = (self.points[b] - self.points[a]) / np.linalg.norm( self.points[b] - self.points[a])
            dot_product = np.dot(prev, next)
            curr_rotation = np.arccos(dot_product)
            if not np.isnan(curr_rotation):
                return abs(curr_rotation)
            return 0

        def get_distance(a,b):
            a_point = self.points[a]
            b_point = self.points[b]
            return np.linalg.norm(a_point - b_point)

        def cost(a, b):
            length = get_distance(a,b)
            path_until_now = self.reconstruct_path(prev_point, a)
            visited_cost = 0
            if self.has_been_visited(self.points[b], ROBOT_RADIUS, path=path_until_now):
                visited_cost = 100
            rotation = get_rotation(a,b)
            return length + visited_cost + 5*rotation

        def path_cost(path):
            prev_point = path[0]
            total_cost = 0
            for point in path[1:]:
                total_cost += cost(prev_point, point)
            return total_cost

        def goal_reached(curr_idx):
            return curr_idx == 1

        def heuristic(point_idx):
            current_path = self.reconstruct_path(prev_point, curr_idx)
            pcd = PointCloud(self.print, points=self.coverable_pcd.points)
            pcd.visit_path(current_path)
            coverage = pcd.get_coverage_efficiency()
            if not coverage:
                return 10**6
            return 1/coverage * 100
        #self.print(self.get_node(start_point + np.array([-8.8,4,0])))
        
        
        

        while open_set:
            #current := the node in openSet having the lowest fScore[] value
            f, curr_idx = pop(open_set)
            self.print("curr: " + str((curr_idx, f)))
            current_path = self.reconstruct_path(prev_point, curr_idx)
            pcd = PointCloud(self.print, points=self.coverable_pcd.points)
            pcd.visit_path(current_path)
            coverage = pcd.get_coverage_efficiency()
            self.print("coverage: " + str(coverage))
            self.print("path length: " + str(len(current_path)))
            if pcd.get_coverage_efficiency() > 0.03:
                return current_path
            
            for neighbor, w in self.get_neighbours_for_astar(curr_idx).items():
                possible_cost_to_start = g_cheapest_cost_to_start[curr_idx] + cost(curr_idx, neighbor)
                if not g_cheapest_cost_to_start.get(neighbor) or possible_cost_to_start < g_cheapest_cost_to_start[neighbor]:
                    prev_point[neighbor] = curr_idx
                    g_cheapest_cost_to_start[neighbor] = possible_cost_to_start
                    f_best_final_cost_guess[neighbor] = possible_cost_to_start + heuristic(neighbor)
                    if neighbor not in open_set:
                        push(open_set, (f_best_final_cost_guess[neighbor], neighbor))
                    else:
                        self.print("FOUND NEIGHBOUR!")
                    

            pass
        '''
        // A* finds a path from start to goal.
        // h is the heuristic function. h(n) estimates the cost to reach goal from node n.
        function A_Star(start, nextgoal, h)
            // The set of discovered nodes that may need to be (re-)expanded.
            // Initially, only the start node is known.
            // This is usually implemented as a min-heap or priority queue rather than a hash-set.
            openSet := {start}

            // For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start
            // to n currently known.
            cameFrom := an empty map

            // For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
            gScore := map with default value of Infinity
            gScore[start] := 0

            // For node n, fScore[n] := gScore[n] + h(n). fScore[n] represents our current best guess as to
            // how short a path from start to finish can be if it goes through n.
            fScore := map with default value of Infinity
            fScore[start] := h(start)

            while openSet is not empty
                // This operation can occur in O(1) time if openSet is a min-heap or a priority queue
                current := the node in openSet having the lowest fScore[] value
                if current = goal
                    return reconstruct_path(cameFrom, current)

                openSet.Remove(current)
                for each neighbor of current
                    // d(current,neighbor) is the weight of the edge from current to neighbor
                    // tentative_gScore is the distance from start to the neighbor through current
                    tentative_gScore := gScore[current] + d(current, neighbor)
                    if tentative_gScore < gScore[neighbor]
                        // This path to neighbor is better than any previous one. Record it!
                        cameFrom[neighbor] := current
                        gScore[neighbor] := tentative_gScore
                        fScore[neighbor] := gScore[neighbor] + h(neighbor)
                        if neighbor not in openSet
                            openSet.add(neighbor)

            // Open set is empty but goal was never reached
            return failure
        '''
        current_position = start_point

        
        
        self.print_stats(self.path)
        
        return self.path

    
    def reconstruct_path(self, cameFrom, current):
        #self.print("reconstruct_path")
        total_path = np.array([self.points[current]])
        while current != 0:
            current = cameFrom[current]
            total_path = np.append(total_path, [self.points[current]], axis=0)
        return total_path

    def get_neighbours_for_astar(self, curr_node):
        ''' Help function for Astar to find neighbours to a given node 
        Args:
            curr_node: Node as index in self.astar_points
        Returns:
            Valid neighbours to curr_node as a dictionary, with index in
            self.astar_points as key and distance to given node as value. 
        '''
        current_point = self.points[curr_node]
        neighbours = {}
        nbr_of_neighbours = 8

        for direction in range(nbr_of_neighbours):
            angle = direction/nbr_of_neighbours*np.pi*2
            x = current_point[0] + np.cos(angle) * self.step_size
            y = current_point[1] + np.sin(angle) * self.step_size
            z = current_point[2]
            pos = np.array([x, y, z])
            
            nearest_point = self.traversable_pcd.find_k_nearest(pos, 1)[0]

            if self.motion_planner.is_valid_step(current_point, nearest_point):
                
                node = self.get_node(nearest_point)

                if node is False:
                    node = max(self.points.keys())+1
                    self.points[node] = nearest_point

                if node == curr_node:
                    continue

                neighbours[node] = self.cost(current_point, nearest_point)

        return neighbours

    
    def get_node(self, point):
        ''' Finds the node (index in self.points) that is close to a given point.
        Args:
            point: A [x,y,z] array with the position of the point
        Returns:
            Index of the closest point in self.astar_points or False if no point found nearby.
        '''
        #distance_to_existing_nodes = np.linalg.norm(self.points.values() - point, axis=1)
        #closest_node = np.argmin(distance_to_existing_nodes)
        #distance = distance_to_existing_nodes[closest_node]
        closest_idx, closest_point = min(self.points.items(), key=lambda x: np.linalg.norm(np.array(x[1]) - np.array(point)))
        if np.linalg.norm(np.array(closest_point) - np.array(point)) < self.visited_threshold:
            return closest_idx

        return False
    