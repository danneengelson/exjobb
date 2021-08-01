import numpy as np
import open3d as o3d
import timeit

from exjobb.Parameters import CELL_SIZE, Z_RESOLUTION, MIN_FLOOR_HEIGHT, FLOOR_THICKNESS, MIN_POINTS_IN_CELL

class ElevationDetector:
    ''' Class for creating a Discrete Elevation Model of the Point Cloud.
    '''

    def __init__(self, print):
        ''' 
        Args:
            print: function for printing messages
        '''
        self.print = print

    def find_elevation(self, full_pcd, floor):
        ''' Creating a Discrete Elevation Model of the Point Cloud by 
        assigning elevation heights to every valid cell in a 2D grid
        of the given floor. 
        Args:
            full_pcd: Point Cloud of the environment including other floors
            floor: Floor Class object of an arbitary floor in the environment.
        '''
        start = timeit.default_timer()

        nbr_of_cells = floor.cells_grid.shape[0]*floor.cells_grid.shape[1]
        cells_with_elevation = 0
        kdtree = o3d.geometry.KDTreeFlann(full_pcd)

        for x_idx, y_idx in np.ndindex(floor.cells_grid.shape):
            
            x = x_idx * CELL_SIZE + floor.min_x
            y = y_idx * CELL_SIZE + floor.min_y  

            if self.approximate_number_of_points_in_cell_is_too_low(kdtree, x, y, floor):
                continue

            cell_bounding_box = o3d.geometry.AxisAlignedBoundingBox(
                    [x, y, floor.z_ground],
                    [x + CELL_SIZE, y + CELL_SIZE, floor.z_ceiling]
                )
            
            points_idx_in_cell = cell_bounding_box.get_point_indices_within_bounding_box(full_pcd.points)


            z_values_of_points_in_cell = np.asarray(full_pcd.points)[points_idx_in_cell , 2]
            z_values_of_points_in_cell = np.append(z_values_of_points_in_cell, floor.z_ceiling)

            if len(z_values_of_points_in_cell) < MIN_POINTS_IN_CELL:
                continue

            cells_with_elevation += 1

            elevation = self.get_elevation_of_cell(z_values_of_points_in_cell)

            nbr_of_floor_points = self.get_nbr_of_floor_points(elevation, z_values_of_points_in_cell)
            
            if nbr_of_floor_points < MIN_POINTS_IN_CELL:
                continue
            
            floor.add_cell(x_idx, y_idx, elevation, points_idx_in_cell)

        self.print_result(start, cells_with_elevation, nbr_of_cells)  


    def get_elevation_of_cell(self, z_values):    
        ''' Starting from the lowest point, it looks for an empty space in the cell
        with a height of at least the MIN_FLOOR_HEIGHT, where there are no points.
        Args:
            z_values: List of the z-values of all points in the cell
        '''

        sorted_z_values = np.sort(z_values)

        for idx, z in enumerate(sorted_z_values[1:]):
            prev_z = sorted_z_values[idx]
            
            if abs(z - prev_z) > MIN_FLOOR_HEIGHT:
                break
                
        return prev_z

    def approximate_number_of_points_in_cell_is_too_low(self, kdtree, x, y, floor):
        ''' Quick approximate check to see if a cell is valid. On different heights,
        it checks whether the number of points in a radius is bigger than the minimum
        amount of MIN_POINTS_IN_CELL. The purpose is to lower the computational time.
        Args:
            kdtree: a Open3D KDTreeFlann representation of the point lcoud
            x: x-value of the cell
            y: y-value of the cell
            floor: The Floor where the cell is included. 
        
        '''
        
        x_center = x + CELL_SIZE/2
        y_center = y + CELL_SIZE/2
        z_values = np.arange( floor.z_ground, floor.z_ceiling, Z_RESOLUTION)
        
        radius = CELL_SIZE / np.sqrt(2)
        for z in z_values:
            center_point = np.asarray([x_center, y_center, z])
            [k, idx, _] = kdtree.search_radius_vector_3d(center_point, radius)
            
            if len(idx) > MIN_POINTS_IN_CELL:
                return False
        
        return True
            
    def get_nbr_of_floor_points(self, elevation, z_values):
        ''' Calculates points that represent the floor in a cell. Looks for
        points that are at a hegiht of FLOOR_THICKNESS below the elevation.
        Args:
            elevation: Elevation height of the cell
            z_values: list of heights of all points in the cell
        '''
        is_close_to_elevation = np.logical_and(z_values >=  elevation - FLOOR_THICKNESS, z_values <= elevation)
        return len(np.where(is_close_to_elevation)[0])


    def print_result(self, start, cells_with_elevation, nbr_of_cells):
        ''' Prints result data of the floor segmentation.
        '''
        end = timeit.default_timer()
        self.print("-"*20)
        self.print("FIND ELEVATION")
        self.print("Computational time: " + str(round(end - start, 1)) + " sec")
        self.print("Valid cells with elevation: " + str(cells_with_elevation) + " out of " + str(nbr_of_cells))            