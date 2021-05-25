
import numpy as np
import open3d as o3d
from exjobb.Parameters import CELL_SIZE

NO_CELL = -1

class Cell:
    def __init__(self, elevation, points_idx_in_full_pcd):
        self.points_idx_in_full_pcd = points_idx_in_full_pcd
        self.elevation = elevation
        self.is_traversable = False
    #self.neighbours#???


class Floor:
    def __init__(self, name, full_pcd, points_idx=np.array([])):
        self.name = name
        self.points_idx_in_full_pcd = points_idx

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.asarray(full_pcd.points)[points_idx.astype(int)])
        
        bounding_box = self.pcd.get_axis_aligned_bounding_box()
        min_bounds = np.min(np.asarray(bounding_box.get_box_points()), axis=0)
        max_bounds = np.max(np.asarray(bounding_box.get_box_points()), axis=0)

        self.min_x = min_bounds[0]
        self.min_y = min_bounds[1]
        self.min_z = min_bounds[2]
        self.max_x = max_bounds[0]
        self.max_y = max_bounds[1]
        self.max_z = max_bounds[2] 
        self.z_ground = self.min_z
        self.z_ceiling = self.max_z

        self.cells_grid = self.make_2D_grid()

    def make_2D_grid(self):
        nbr_of_x = int(np.ceil((self.max_x-self.min_x) / CELL_SIZE)) 
        nbr_of_y = int(np.ceil((self.max_y-self.min_y) / CELL_SIZE)) 
        grid = np.empty((nbr_of_x, nbr_of_y), dtype=object)
        return grid

    def add_cell(self, x_idx, y_idx, elevation, points_idx_in_full_pcd):
        new_cell = Cell(elevation, points_idx_in_full_pcd)
        self.cells_grid[x_idx, y_idx] = new_cell

    def get_elevation(self, cell_idx):
        return self.cells[int(cell_idx)].elevation

    def is_valid_cell(self, pos):
        try:
            cell = self.cells_grid[pos]
            if cell is None:
                return False
                
            return True
        except:
            return False