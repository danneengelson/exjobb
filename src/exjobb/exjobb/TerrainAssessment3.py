import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import operator
import collections
import timeit

# TUNING VALUES:
STEP_RESOLUTION = 3
NUMBER_OF_LAYERS = 50
GROUND_THRESHOLD = 200000
ROBOT_SIZE = 1
MIN_POINTS_IN_CELL = 40
ROBOT_HEIGHT = 1.5
STEP_SIZE = 0.25 * ROBOT_SIZE
#0.3 - bra floor 1, 0.2 - bra floor 2, 0.25 - nästan bra floor 1 för size = 1
#0.15 med size 0,5 blev bra. ev. lite lägre, någon enstaka fel
FLOOR_NR = 0

UNKNOWN_ELEV_Z = -100

UNKNOWN = 0
GROUND = 1
WALL = 2
INCLINED = 3



class Pose:
    k_nearest_points = None
    center_point = None
    normal = None
    traversable = True 
    voxel_idx = None

class Position:
    def __init__(self, position):
        self.position = position

class TerrainAssessment3():

    def __init__(self, logger, pcd):
        self.logger = logger
        self.pcd = pcd.raw_pcd
        self.points = pcd.points
        self.pcd_kdtree = pcd.kdtree

    def analyse_terrain(self, start_pos):
        start_total = timeit.default_timer()

        bbox = self.pcd.get_axis_aligned_bounding_box()
        min_bounds = np.min(np.asarray(bbox.get_box_points()), axis=0)
        max_bounds = np.max(np.asarray(bbox.get_box_points()), axis=0)
        center_point = bbox.get_center()
        min_x = min_bounds[0]
        min_y = min_bounds[1]
        min_z = min_bounds[2]
        max_x = max_bounds[0]
        max_y = max_bounds[1]
        max_z = max_bounds[2] 


        ######################
        # FLOOR SEGMENTATION #
        ######################
        start_floor_segmentation = timeit.default_timer()

        segmentized_floors = self.get_segmentized_floors(min_x, max_x, min_y, max_y, min_z, max_z)

        self.full_pcd = []

        for floor in range(len(segmentized_floors)):

            pcd = segmentized_floors[floor]["pcd"]
            ground_z = segmentized_floors[floor]["ground_z"]
            ceiling_z = segmentized_floors[floor]["ceiling_z"]

            end_floor_segmentation = timeit.default_timer()
            self.logger.info("Floor segmentation: " + str(end_floor_segmentation - start_floor_segmentation))


            ##################
            # FIND ELEVATION #
            ##################

            start_find_elevation = timeit.default_timer()

            self.nbr_of_x = int(np.ceil((max_x-min_x) / ROBOT_SIZE)) 
            self.nbr_of_y = int(np.ceil((max_y-min_y) / ROBOT_SIZE)) 
            x_values = np.linspace(min_x, max_x, self.nbr_of_x) 
            y_values = np.linspace(min_y, max_y, self.nbr_of_y) 

            # Coordinates of all evaluated cells
            self.all_cells_pos_list = []

            # PCD of evaluated cells
            self.all_cells_pcd_list = []

            # Grid with classes
            self.cell_classes_grid = np.empty((self.nbr_of_x, self.nbr_of_y))
            self.cell_classes_grid.fill(UNKNOWN)

            # Grid with respective index in list
            self.cell_list_idx_grid = np.empty((self.nbr_of_x, self.nbr_of_y))

            # Grid with elev levels
            self.cell_elev_z_grid = np.empty((self.nbr_of_x, self.nbr_of_y))

            total_geom = 0
            total_loop = 0
            total_assign = 0
            total_downsample = 0

            for x_pos, x in enumerate(x_values[1:]):
                for y_pos, y in enumerate(y_values[1:]):
                                
                    cell_bbox = o3d.geometry.AxisAlignedBoundingBox( [x-ROBOT_SIZE, y-ROBOT_SIZE, ground_z], [x, y, ceiling_z])
                    
                    #Sample down PCD
                    #start_downsample = timeit.default_timer()
                    #self.sample_down_pcd(self.pcd)
                    #end_downsample  = timeit.default_timer()
                    #total_downsample += (end_downsample - start_downsample)
                    
                    start_geom = timeit.default_timer()               
                    cell_pos = (x_pos, y_pos)
                    cell_pcd = self.pcd.crop(cell_bbox)
                    cell_z_values = self.get_z_values(cell_pcd)
                    nbr_of_cell_z_values = len(cell_z_values)

                    
        
                    end_geom = timeit.default_timer()
                    total_geom += (end_geom - start_geom)

                    cell_elev_z = UNKNOWN_ELEV_Z

                    if nbr_of_cell_z_values < MIN_POINTS_IN_CELL*4:
                        continue               
                    
                    

                    start_loop = timeit.default_timer()

                    cell_z_values = np.append(cell_z_values, ceiling_z)
                    cell_elev_z = self.get_elev_z(cell_z_values)

                    end_loop = timeit.default_timer()
                    total_loop += (end_loop - start_loop)

                    start_assign = timeit.default_timer()

                    if self.is_ground_cell(cell_elev_z, ground_z):
                        self.update_cell_in_grid(cell_pos, GROUND, cell_elev_z, len(self.all_cells_pos_list))
                    else:
                        self.update_cell_in_grid(cell_pos, UNKNOWN, cell_elev_z, len(self.all_cells_pos_list))

                    self.add_cell_to_list(cell_pos, cell_pcd)   

                    end_assign = timeit.default_timer()
                    total_assign += (end_assign - start_assign)
            
            

            self.logger.info("total_downsample: " + str(total_downsample)) 
            self.logger.info("total_geom: " + str(total_geom))
            self.logger.info("total_loop: " + str(total_loop))
            self.logger.info("total_assign: " + str(total_assign)) 
        
            end_find_elevation = timeit.default_timer()
            self.logger.info("Find elevation: " + str(end_find_elevation - start_find_elevation))
            
            #################
            # Breadth First #
            #################

            start_breadth_first_search = timeit.default_timer()

            ground_cells_pos_list = self.get_ground_cells_pos_list()
            self.assign_with_breadth_first(ground_cells_pos_list)
            
            end_breadth_first_search = timeit.default_timer()
            self.logger.info("Breadth first: " + str(end_breadth_first_search - start_breadth_first_search))


            #############
            # VISUALIZE #
            #############
            start_visualize = timeit.default_timer()

            for cell_pos in self.all_cells_pos_list:
                cell_idx = int(self.cell_list_idx_grid[cell_pos])

                if self.cell_classes_grid[cell_pos] == GROUND or self.cell_classes_grid[cell_pos] == INCLINED:

                    self.paint_cell(cell_idx, [0,0,0])

                    if not self.cell_elev_z_grid[cell_pos] == UNKNOWN_ELEV_Z:
                        ground_points_pcd = self.get_painted_ground_points_pcd(cell_idx, cell_pos)
                        if not ground_points_pcd:
                            continue
                        
                        self.all_cells_pcd_list.append(ground_points_pcd)               
                    
                elif self.cell_classes_grid[cell_pos] == WALL:
                    self.paint_cell(cell_idx, [1,0,0])

                elif self.cell_classes_grid[cell_pos] == UNKNOWN:
                    self.paint_cell(cell_idx, [1,0,1])
            
            end_visualize = timeit.default_timer() 
            self.logger.info("Visualize: " + str(end_visualize - start_visualize))
            
            self.full_pcd.extend(self.all_cells_pcd_list)


        end_total = timeit.default_timer()        
        
        self.logger.info("Total: " + str(end_total - start_total))

        o3d.visualization.draw_geometries(self.full_pcd,
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
        
        
        return []


    def get_neighbour(self, cell):
        directions = [(1,0), (0,1), (-1, 0), (0, -1), (1, 1), (-1,1), (-1, -1), (1, -1)]
        neighbours = []
        for dir in directions:
            neighbour = tuple(map(operator.add,cell,dir))
            if self.is_inside_grid(neighbour):
                neighbours.append(neighbour)

        return neighbours

    def get_segmentized_floors(self, min_x, max_x, min_y, max_y, min_z, max_z):
        z_values = np.linspace(min_z, max_z, NUMBER_OF_LAYERS)
        number_of_points_in_layers = []
        potential_maximas = []

        for idx in range(NUMBER_OF_LAYERS - 1):
            layer = o3d.geometry.AxisAlignedBoundingBox( np.array([min_x, min_y, z_values[idx]]), [max_x, max_y, z_values[idx + 1]])
            cell_pcd = self.pcd.crop(layer)
            number_of_points_in_layer = len(cell_pcd.points)
            number_of_points_in_layers.append(number_of_points_in_layer)
            
            if self.is_local_maximum(idx, number_of_points_in_layers):
                potential_maximas.append(idx-1)

        #self.logger.info(str(z_values[potential_maximas]))
        #plt.plot(z_values[1:], number_of_points_in_layers)
        #plt.ylabel('some numbers')
        #plt.show()
        floor_levels = z_values[potential_maximas]
        floor_1 = self.pcd.crop(o3d.geometry.AxisAlignedBoundingBox( np.array([min_x, min_y, floor_levels[0] ]), [max_x, max_y, floor_levels[1]-0.5]))
        floor_2 = self.pcd.crop(o3d.geometry.AxisAlignedBoundingBox( np.array([min_x, min_y, floor_levels[1]-0.5 ]), [max_x, max_y, max_z]))

        segmentized_floors = [{
            "pcd": floor_1,
            "ground_z": floor_levels[0],
            "ceiling_z": floor_levels[1]-0.5
        },
        {
            "pcd": floor_2,
            "ground_z": floor_levels[1]-0.5,
            "ceiling_z": max_z
        }]
        return segmentized_floors

    def is_local_maximum(self, idx, number_of_points_in_layers):
        if idx < 3:
            return False
            
        number_of_points_in_prev_layer = number_of_points_in_layers[idx-1]
        number_of_points_in_prev_prev_layer = number_of_points_in_layers[idx-2]
        prev_is_local_maximum = number_of_points_in_prev_layer > number_of_points_in_layers[idx] and number_of_points_in_prev_layer > number_of_points_in_prev_prev_layer

        if prev_is_local_maximum and number_of_points_in_prev_layer > GROUND_THRESHOLD: 
            return True
        
        return False

    def sample_down_pcd(pcd):
        down_sampled_pcd = pcd
        #down_sampled_pcd = self.pcd.uniform_down_sample(10)
        #down_sampled_pcd = self.pcd.voxel_down_sample(0.5)
        return down_sampled_pcd

    def get_z_values(self, pcd):
        return np.asarray(pcd.points)[:,2]

    def get_elev_z(self, z_values):
        sorted_z_values = np.sort(z_values)
        for idx, z in enumerate(sorted_z_values[1:]):
            prev_z = sorted_z_values[idx]
            
            if abs(z - prev_z) > ROBOT_HEIGHT:
                break
                
        return prev_z
    
    def is_inside_grid(self, pos):
        x_pos = pos[0]
        y_pos = pos[1]
        if x_pos < 0:
            return False
        if y_pos < 0:
            return False
        if x_pos >= self.nbr_of_x:
            return False
        if y_pos >= self.nbr_of_y:
            return False
        
        return True

    def update_cell_in_grid(self,cell_pos, new_class, elev_z, list_idx):
        self.cell_classes_grid[cell_pos] = new_class
        self.cell_elev_z_grid[cell_pos] = elev_z
        self.cell_list_idx_grid[cell_pos] = list_idx

    def add_cell_to_list(self, cell_pos, cell_pcd):        
        self.all_cells_pos_list.append(cell_pos)
        self.all_cells_pcd_list.append( cell_pcd )

    def is_ground_cell(self, cell_elev_z, ground_z):
        return cell_elev_z < ground_z + STEP_SIZE

    def assign_with_breadth_first(self, ground_cell_pos_queue):
        while len(ground_cell_pos_queue):
            #self.logger.info(str(len(q)))
            ground_pos = ground_cell_pos_queue.pop()
            for neigbour_pos in self.get_neighbour(ground_pos):
                
                if self.cell_classes_grid[neigbour_pos] == UNKNOWN and self.is_traversable_from_pos(neigbour_pos, ground_pos):
                    self.cell_classes_grid[neigbour_pos] = INCLINED
                    ground_cell_pos_queue.append(neigbour_pos)

                elif self.cell_classes_grid[neigbour_pos] == UNKNOWN:
                    self.cell_classes_grid[neigbour_pos] = WALL

    def is_traversable_from_pos(self, neigbour_pos, ground_pos):
        return abs(self.cell_elev_z_grid[neigbour_pos] - self.cell_elev_z_grid[ground_pos]) <= STEP_SIZE

    def paint_cell(self, cell_idx, color):
        self.all_cells_pcd_list[cell_idx] = self.all_cells_pcd_list[cell_idx].paint_uniform_color(color)
    
    def get_painted_ground_points_pcd(self, cell_idx, cell_pos):
        x = self.all_cells_pcd_list[cell_idx].get_center()[0]
        y = self.all_cells_pcd_list[cell_idx].get_center()[1]
        z = self.cell_elev_z_grid[cell_pos]

        ground_level_bbox = o3d.geometry.AxisAlignedBoundingBox([x-ROBOT_SIZE/2, y-ROBOT_SIZE/2, z - 0.5], [x+ROBOT_SIZE/2, y+ROBOT_SIZE/2, z + 0.5])
        ground_level_points = self.all_cells_pcd_list[cell_idx].crop(ground_level_bbox)
        ground_level_points.translate([0,0,0.03])

        if len(ground_level_points.points) > MIN_POINTS_IN_CELL:
            if self.cell_classes_grid[cell_pos] == GROUND:
                return ground_level_points.paint_uniform_color(np.array([0,1,0]))
            elif self.cell_classes_grid[cell_pos] == INCLINED:
                return ground_level_points.paint_uniform_color(np.array([0,1,0]))

        return False

    def get_ground_cells_pos_list(self):
        ground_cells_pos_list = []
        x_poses, y_poses = np.where(self.cell_classes_grid == GROUND)
        for idx in range(len(x_poses)):
            cell_pos = (x_poses[idx], y_poses[idx])
            ground_cells_pos_list.append( cell_pos )
        return ground_cells_pos_list


            

                       