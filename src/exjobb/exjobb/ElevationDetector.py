import numpy as np
import open3d as o3d
import timeit

from exjobb.Parameters import CELL_SIZE
from exjobb.Parameters import Z_RESOLUTION

#Density to count as valid: a cluster with at least 1 point per 5cm² -> 400/m^2
CELL_AREA = CELL_SIZE * CELL_SIZE
MIN_POINTS_IN_CELL = CELL_AREA * 100
ROBOT_HEIGHT = 1.5
UNKNOWN_ELEV_Z = -100 

class ElevationDetector:
    def __init__(self, print):
        self.print = print

    def find_elevation(self, full_pcd, floor):
        start = timeit.default_timer()
        cells_in_grid = floor.cells_grid.shape[0]*floor.cells_grid.shape[1]
        cells_with_elevation = 0
        total_part = 0
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

            #if len(z_values_of_points_in_cell) < MIN_POINTS_IN_CELL:
            #    continue
            if len(z_values_of_points_in_cell) < MIN_POINTS_IN_CELL:
                #self.print("Too few in cell")
                continue

            cells_with_elevation += 1
            
            elevation = self.get_elevation_of_cell(z_values_of_points_in_cell)

            
            #self.print(z_values_of_points_in_cell)
            #self.print(elevation)
            is_close_to_elevation = np.logical_and(z_values_of_points_in_cell >=  elevation-5*Z_RESOLUTION, z_values_of_points_in_cell <= elevation)
            #self.print(is_close_to_elevation)
            nbr_of_cells_close_to_elevation = len(np.where(is_close_to_elevation)[0])
            #self.print(nbr_of_cells_close_to_elevation)
            
            if nbr_of_cells_close_to_elevation < MIN_POINTS_IN_CELL:
                #self.print("Too few at the elevation point")
                continue
            
            floor.add_cell(x_idx, y_idx, elevation, points_idx_in_cell)
            #start_part = timeit.default_timer()
            #end_part = timeit.default_timer()
            #total_part += end_part-start_part


        
        self.print_result(start, cells_with_elevation, cells_in_grid)
        self.print("total_part" + str(total_part))

    

    def get_elevation_of_cell(self, z_values_of_points_in_cell):    

        sorted_z_values = np.sort(z_values_of_points_in_cell)

        for idx, z in enumerate(sorted_z_values[1:]):
            prev_z = sorted_z_values[idx]
            
            if abs(z - prev_z) > ROBOT_HEIGHT:
                break
                
        return prev_z

    def approximate_number_of_points_in_cell_is_too_low(self, kdtree, x, y, floor):
        x_center = x + CELL_SIZE/2
        y_center = y + CELL_SIZE/2
        z_values = np.arange( floor.z_ground, floor.z_ceiling, Z_RESOLUTION)
        
        cell_points_idx = np.asarray([], dtype=np.int)
        radius = CELL_SIZE / np.sqrt(2)
        for z in z_values:
            center_point = np.asarray([x_center, y_center, z])
            [k, idx, _] = kdtree.search_radius_vector_3d(center_point, radius)
            
            #test:
            if len(idx) > MIN_POINTS_IN_CELL:
                return False


            #cell_points_idx = np.append(cell_points_idx, idx)

            #if len(np.unique(cell_points_idx)) > MIN_POINTS_IN_CELL:
            #    return False
        
        return True
            


    def print_result(self, start, cells_with_elevation, cells_in_grid):
        end = timeit.default_timer()
        self.print("-"*20)
        self.print("FIND ELEVATION")
        self.print("Computational time: " + str(round(end - start, 1)) + " sec")
        self.print("Valid cells with elevation: " + str(cells_with_elevation) + " out of " + str(cells_in_grid))            