
import numpy as np
import timeit
import operator

from exjobb.Parameters import ROBOT_STEP_SIZE, GROUND_OFFSET


class TraversabilityDetector:
    ''' Class for detecting obstacles and unaccessible areas from the DEM, Discrete
    Elevation Model to find coverable areas.
    '''
    def __init__(self, print):
        ''' 
        Args:
            print: function for printing messages
        '''
        self.print = print

    def get_coverable_points_idx(self, full_pcd, floor):
        ''' Finds points that are accessable. No robot size properties are taken into concern.
        Args:
            full_pcd: Point Cloud of the environment including other floors.
            floor: Floor instance with the DEM.
        '''
        start = timeit.default_timer()
        ground_cells_poses = self.get_ground_cells_poses(floor)

        accessible_cells_poses = self.get_accessible_cells_poses(ground_cells_poses, floor)


        coverable_points_idx = np.array([], int)
        for pos in accessible_cells_poses:
            points_idx_in_cell = floor.cells_grid[pos].points_idx_in_full_pcd 
            #####
            elevation = floor.cells_grid[pos].elevation 
            ###
            points_in_cell = np.asarray(full_pcd.points)[points_idx_in_cell]
            new_coverable_points_idx = self.get_coverable_points_idx_in_cell(points_in_cell, points_idx_in_cell, elevation)
            coverable_points_idx = np.append(coverable_points_idx, new_coverable_points_idx )
        
        coverable_points_idx = np.unique(coverable_points_idx)
        self.print_result(start, coverable_points_idx, accessible_cells_poses)

        return coverable_points_idx

    
    def get_accessible_cells_poses(self, ground_cells_poses, floor):
        ''' Using breadth first to find all accessable poses. Uses ground cells as
        starting points, which creates mulitple islands of connected cells.
        The cells of the island with most cells are chosen as accessable.
        Args:
            ground_cells_poses: positions of all ground level cells
            floor: Floor instance with the DEM.
        '''
        
        all_islands = []
        ground_cells_queue = [tuple(pair) for pair in ground_cells_poses]

        while ground_cells_queue:
            start_cell = ground_cells_queue.pop()
            visited_cells = [start_cell]
            traversable_cells_queue = [start_cell]
            while traversable_cells_queue:
                current_pos = traversable_cells_queue.pop()
                for neigbour_pos in self.get_neighbours_poses(current_pos):

                    if not floor.is_valid_cell(neigbour_pos):
                        continue   

                    if neigbour_pos in visited_cells:
                        continue 
                    
                    if not self.is_valid_step(current_pos, neigbour_pos, floor):
                        continue                    

                    if neigbour_pos in ground_cells_queue:
                        ground_cells_queue.remove(neigbour_pos)

                    visited_cells.append(neigbour_pos)
                    traversable_cells_queue.append(neigbour_pos)
                
            all_islands.append(visited_cells)


        cells_in_biggest_island = max(all_islands, key=len)

        for cell in cells_in_biggest_island:
            floor.cells_grid[cell].is_traversable = True

        return cells_in_biggest_island


    def get_neighbours_poses(self, cell):
        ''' Returns the position of the north, south, west an east neighbour of a given cell.
        Args:
            cell: tuple with an arbitary position
        '''

        directions = [(1,0), (0,1), (-1, 0), (0, -1)]
        neighbours = []
        for dir in directions:
            neighbour = tuple(map(operator.add, cell, dir))
            neighbours.append(neighbour)

        return neighbours
    
    def is_valid_step(self, from_pos, to_pos, floor):
        ''' Checks whether a step from one cell to another is possible by
        looking at the height difference between the cells.
        Args:
            from_pos: tuple (x,y) representing the position of the start cell
            to_pos: tuple (x,y) representing the position of the goal cell
            floor: Floor instance with the DEM.
        '''
        ground_pos_elevation = floor.cells_grid[from_pos].elevation
        neigbour_pos_elevation = floor.cells_grid[to_pos].elevation

        return abs(neigbour_pos_elevation - ground_pos_elevation) <= ROBOT_STEP_SIZE

    def get_ground_cells_poses(self, floor):
        ''' Finds all ground level cells in the floor.
        Args:
            floor: Floor instance with the DEM. 
        Returns:
            A list of ground cell positions in a tuple format (x,y)
        '''

        def is_ground_level_cell(cell):
            if cell is None:
                return False
            return cell.elevation < floor.z_ground + GROUND_OFFSET
        
        x_poses, y_poses = np.where( np.vectorize(is_ground_level_cell)(floor.cells_grid))
        ground_level_cells = np.vstack((x_poses, y_poses)).T

        return ground_level_cells.tolist()

    def get_coverable_points_idx_in_cell(self, points, points_idx, elevation):  
        ''' Finds points in the cell that are coverable by taking the points close
        to the elvation level of the cell.
        Args:
            points: Nx3 array with all points in the cell
            points_idx: indicies of these points in the full point cloud.        
        '''
        z_values = points[:,2]        
        coverable = np.where(  abs(z_values - elevation) < ROBOT_STEP_SIZE  )[0]
        coverable = np.array(coverable, int)

        return np.take(points_idx, coverable)
 
    def print_result(self, start, coverable_points, coverable_cells):
        ''' Prints result data of the floor segmentation.
        '''
        end = timeit.default_timer()
        self.print("-"*20)
        self.print("FIND COVERABLE POINTS")
        self.print("Computational time: " + str(round(end - start, 1)) + " sec")
        self.print("Number of coverable points: " + str(len(coverable_points)))
        self.print("Number of coverable cells: " + str(len(coverable_cells)))
