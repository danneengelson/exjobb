
from exjobb.Parameters import ROBOT_STEP_SIZE, GROUND_OFFSET
import numpy as np
import timeit
import operator

class TraversabilityDetector:
    def __init__(self, print):
        self.print = print

    def get_traversable_points_idx(self, full_pcd,floor):
        start = timeit.default_timer()
        ground_cells_pos_list = self.get_ground_cells_pos_list(floor)
        #self.print("ground_cells_pos_list: " + str(len(ground_cells_pos_list)))

        ground_cells_pos_list = self.assign_with_breadth_first(ground_cells_pos_list, floor)


        traversable_points = np.array([], int)
        for pos in ground_cells_pos_list:
            pos = tuple(pos)
            points_idx_in_cell = floor.cells_grid[pos].points_idx_in_full_pcd 
            
            ground_points_idx_in_cell = self.get_points_below_z(full_pcd, points_idx_in_cell, floor.cells_grid[pos].elevation)
            #ground_points_idx_in_cell = points_idx_in_cell
            traversable_points = np.append(traversable_points, ground_points_idx_in_cell )
        
        
        traversable_points = np.unique(traversable_points)
        self.print_result(start, len(traversable_points), len(ground_cells_pos_list))
        #self.print(traversable_points)
        return traversable_points

    
    def assign_with_breadth_first(self, ground_cells_pos_list, floor):
        
        all_islands = []
        ground_cell_pos_queue = [tuple(pair) for pair in ground_cells_pos_list]

        while len(ground_cell_pos_queue):
            start_cell = ground_cell_pos_queue.pop()
            visited_cells = [start_cell]
            traversable_queue = [start_cell]
            while len(traversable_queue):
                current_pos = traversable_queue.pop()
                for neigbour_pos in self.get_neighbour(current_pos, floor):
                    
                    if not self.is_traversable_from_pos(neigbour_pos, current_pos, floor):
                        continue

                    if neigbour_pos in visited_cells:
                        continue

                    if neigbour_pos in ground_cell_pos_queue:
                        ground_cell_pos_queue.remove(neigbour_pos)

                    visited_cells.append(neigbour_pos)
                    traversable_queue.append(neigbour_pos)
                
            all_islands.append(visited_cells)


        cells_in_biggest_island = max(all_islands, key=len)

        for cell in cells_in_biggest_island:
            floor.cells_grid[cell].is_traversable = True

        return cells_in_biggest_island


    def get_neighbour(self, cell, floor):
        #directions = [(1,0), (0,1), (-1, 0), (0, -1), (1, 1), (-1,1), (-1, -1), (1, -1)]
        directions = [(1,0), (0,1), (-1, 0), (0, -1)]
        neighbours = []
        for dir in directions:
            neighbour = tuple(map(operator.add, cell, dir))
            if floor.is_valid_cell(neighbour):
                neighbours.append(neighbour)

        return neighbours
    
    def is_traversable_from_pos(self, neigbour_pos, ground_pos, floor):
        #self.print(ground_pos)
        #self.print(neigbour_pos)
        ground_pos_elevation = floor.cells_grid[ground_pos]
        neigbour_pos_elevation = floor.cells_grid[neigbour_pos]
        #self.print("ground_pos_elevation" + str(ground_pos_elevation))
        #self.print("neigbour_pos_elevation" + str(neigbour_pos_elevation))
        return abs(neigbour_pos_elevation.elevation - ground_pos_elevation.elevation) <= ROBOT_STEP_SIZE

    def get_ground_cells_pos_list(self, floor):
        ground_level_cells_pos_list = []

        def is_ground_level_cell(cell):
            if cell is None:
                return False
            return cell.elevation < floor.z_ground + GROUND_OFFSET
        
        x_poses, y_poses = np.where( np.vectorize(is_ground_level_cell)(floor.cells_grid))
        ground_level_cells = np.vstack((x_poses, y_poses)).T

        return ground_level_cells.tolist()

    def get_points_below_z(self, pcd, points_idx, z):     
        lowest_z = np.min(  np.asarray(pcd.points)[points_idx][:,2])

        #if z - lowest_z >= ROBOT_STEP_SIZE:
        #    points_below_z = np.where(  np.asarray(pcd.points)[points_idx][:,2] <= lowest_z + ROBOT_STEP_SIZE  )
        #else:
        #    points_below_z = np.where(  np.asarray(pcd.points)[points_idx][:,2] <= z  )  

        points_below_z = np.where(  np.asarray(pcd.points)[points_idx][:,2] <= lowest_z + 2*ROBOT_STEP_SIZE  )[0]
        #points_below_z = np.where(  np.asarray(pcd.points)[points_idx][:,2] <= z  )[0]
        #points_over_robot_step = np.where(  np.asarray(pcd.points)[points_idx][:,2] > z - ROBOT_STEP_SIZE )[0]
        
        #idx_in_points_idx = np.array(np.intersect1d(points_below_z, points_over_robot_step), int)
        idx_in_points_idx = np.array(points_below_z, int)
        return np.take(points_idx, idx_in_points_idx)
 
    def print_result(self, start, traversable_points, traversable_cells):
        end = timeit.default_timer()
        self.print("-"*20)
        self.print("FIND TRAVERSABLE POINTS")
        self.print("Computational time: " + str(round(end - start, 1)) + " sec")
        self.print("Number of traversable points: " + str(traversable_points))
        self.print("Number of traversable cells: " + str(traversable_cells))
