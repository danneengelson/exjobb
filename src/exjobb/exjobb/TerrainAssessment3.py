import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import operator
import collections
import timeit

# TUNING VALUES:
NUMBER_OF_LAYERS = 50
GROUND_THRESHOLD = 200000
ROBOT_SIZE = 0.995
MIN_POINTS_IN_CELL = ROBOT_SIZE*40
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

CROP_TUNNEL = False
TUNNELOFFSET_X = 35
TUNNELOFFSET_Y = 20

class TerrainAssessment3():

    def __init__(self, logger, pcd):
        self.logger = logger
        self.pcd = pcd.raw_pcd
        self.points = pcd.points
        self.pcd_kdtree = pcd.kdtree
        self.traversable_points_idx = np.array([])
        self.nbr_of_ground_level_points = 0
        self.nbr_of_covered_ground_level_points = 0
        self.nbr_of_covered_inclined_ground_level_cells = 0
        

    def analyse_terrain(self):
        start_total = timeit.default_timer()

        bbox = self.pcd.get_axis_aligned_bounding_box()
        self.min_bounds = np.min(np.asarray(bbox.get_box_points()), axis=0)
        max_bounds = np.max(np.asarray(bbox.get_box_points()), axis=0)
        #center_point = bbox.get_center()
        min_x = self.min_bounds[0]
        min_y = self.min_bounds[1]
        min_z = self.min_bounds[2]
        max_x = max_bounds[0]
        max_y = max_bounds[1]
        max_z = max_bounds[2] 

        ######################
        # FLOOR SEGMENTATION #
        ######################
        start_floor_segmentation = timeit.default_timer()

        segmentized_floors = self.get_segmentized_floors(min_x, max_x, min_y, max_y, min_z, max_z)

        self.full_pcd = []
        self.ground_points = np.empty((1,3))

        for floor in [1]: #range(len(segmentized_floors)):

            pcd = segmentized_floors[floor]["pcd"]
            ground_z = segmentized_floors[floor]["ground_z"]
            ceiling_z = segmentized_floors[floor]["ceiling_z"]#-0.5

            end_floor_segmentation = timeit.default_timer()
            self.logger.info("Floor segmentation: " + str(end_floor_segmentation - start_floor_segmentation))


            ##################
            # FIND ELEVATION #
            ##################
            if CROP_TUNNEL:
                max_x = max_bounds[0] - TUNNELOFFSET_X
                max_y = max_bounds[1] - TUNNELOFFSET_Y

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
                                
                    #cell_bbox = o3d.geometry.AxisAlignedBoundingBox( [x-ROBOT_SIZE, y-ROBOT_SIZE, ground_z], [x, y, ceiling_z])
                    
                    #Sample down PCD
                    #start_downsample = timeit.default_timer()
                    #self.sample_down_pcd(self.pcd)
                    #end_downsample  = timeit.default_timer()
                    #total_downsample += (end_downsample - start_downsample)
                    
                    start_geom = timeit.default_timer()               
                    cell_pos = (x_pos, y_pos)
                    #cell_pcd = self.pcd.crop(cell_bbox)
                    cell_pcd = self.get_cell_pcd(x, y, ground_z, ceiling_z)

                    if cell_pcd is False:
                        continue

                    cell_z_values = self.get_z_values(cell_pcd)
                    nbr_of_cell_z_values = len(cell_z_values)

                    
        
                    end_geom = timeit.default_timer()
                    total_geom += (end_geom - start_geom)

                    cell_elev_z = UNKNOWN_ELEV_Z

                    if nbr_of_cell_z_values < MIN_POINTS_IN_CELL:
                        continue               
                    
                    

                    start_loop = timeit.default_timer()

                    cell_z_values = np.append(cell_z_values, ceiling_z)
                    cell_elev_z = self.get_elev_z(cell_z_values)

                    end_loop = timeit.default_timer()
                    total_loop += (end_loop - start_loop)

                    start_assign = timeit.default_timer()

                    #if self.is_ground_cell(cell_elev_z, ground_z):
                    #    self.update_cell_in_grid(cell_pos, GROUND, cell_elev_z, len(self.all_cells_pos_list))
                    #else:
                    #    self.update_cell_in_grid(cell_pos, UNKNOWN, cell_elev_z, len(self.all_cells_pos_list))

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

            ground_cells_pos_list = self.get_ground_cells_pos_list(ground_z)
            self.assign_with_breadth_first(ground_cells_pos_list)
            
            end_breadth_first_search = timeit.default_timer()
            self.logger.info("Breadth first: " + str(end_breadth_first_search - start_breadth_first_search))


            #############
            # VISUALIZE #
            #############
            start_visualize = timeit.default_timer()

            #self.ground_points = np.empty((1,3))

            for cell_pos in self.all_cells_pos_list:
                cell_idx = int(self.cell_list_idx_grid[cell_pos])

                if self.cell_classes_grid[cell_pos] == GROUND or self.cell_classes_grid[cell_pos] == INCLINED:

                    self.paint_cell(cell_idx, [0.0,0.0,0.0])

                    if not self.cell_elev_z_grid[cell_pos] == UNKNOWN_ELEV_Z:
                        self.assign_traversable_points(cell_idx, cell_pos)

                        #ground_points_pcd = self.get_painted_ground_points_pcd(cell_idx, cell_pos)
                        #if not ground_points_pcd:
                        #    continue
                        #
                        #self.all_cells_pcd_list.append(ground_points_pcd)               
                    
                elif self.cell_classes_grid[cell_pos] == WALL:
                    self.paint_cell(cell_idx, [0,0,0])

                elif self.cell_classes_grid[cell_pos] == UNKNOWN:
                    self.paint_cell(cell_idx, [0,0,0])
            
            end_visualize = timeit.default_timer() 
            self.logger.info("Visualize: " + str(end_visualize - start_visualize))
            
            self.full_pcd.extend(self.all_cells_pcd_list)
            #self.get_results()
            #self.print_class_results()
            
            

             




        end_total = timeit.default_timer()        
        
        self.logger.info("Total: " + str(end_total - start_total))

       

        #self.ground_pcd = o3d.geometry.PointCloud()
        #self.logger.info(str(self.ground_points))
        #self.logger.info(str(self.ground_points.shape))
        #self.ground_pcd.points = o3d.utility.Vector3dVector(self.ground_points)
        #self.ground_pcd.paint_uniform_color(np.array([0,0,1]))
        #self.ground_pcd.translate([0,0,0.06])
        
        
        #self.full_pcd.append(self.ground_pcd)

        #o3d.visualization.draw_geometries(self.full_pcd,
        #                          zoom=0.3412,
        #                          front=[0.4257, -0.2125, -0.8795],
        #                          lookat=[2.6172, 2.0475, 1.532],
        #                          up=[-0.0694, -0.9768, 0.2024],
        #                          point_show_normal=True)
        
        
        #return self.ground_points


    def get_neighbour(self, cell):
        directions = [(1,0), (0,1), (-1, 0), (0, -1), (1, 1), (-1,1), (-1, -1), (1, -1)]
        neighbours = []
        for dir in directions:
            neighbour = tuple(map(operator.add, cell, dir))
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

        self.logger.info(str(z_values[potential_maximas]))
        #plt.plot(z_values[1:], number_of_points_in_layers)
        #plt.ylabel('Number of points')
        #plt.xlabel('z-value [m]')
        #plt.title('Number of Points on Different Heights')
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
        
        all_islands = []

        while len(ground_cell_pos_queue):
            start_cell = ground_cell_pos_queue[0]
            visited_cells = []
            traversable_queue = [start_cell]
            self.logger.info("Started with " + str( start_cell ))
            while len(traversable_queue):
                self.logger.info(str(len(traversable_queue)))
                ground_pos = traversable_queue.pop()
                self.logger.info(str(ground_pos))
                for neigbour_pos in self.get_neighbour(ground_pos):
                    
                    if not self.is_traversable_from_pos(neigbour_pos, ground_pos):
                        continue

                    if neigbour_pos in visited_cells:
                        continue

                    visited_cells.append(neigbour_pos)

                    if self.cell_classes_grid[neigbour_pos] == GROUND and self.is_traversable_from_pos(neigbour_pos, ground_pos):
                        found_ground_cells.append(neigbour_pos)
                        traversable_queue.append(neigbour_pos)

                    elif self.cell_classes_grid[neigbour_pos] == UNKNOWN and self.is_traversable_from_pos(neigbour_pos, ground_pos):
                        self.cell_classes_grid[neigbour_pos] = INCLINED
                        traversable_queue.append(neigbour_pos)

                    elif self.cell_classes_grid[neigbour_pos] == UNKNOWN:
                        self.cell_classes_grid[neigbour_pos] = WALL
                
            all_islands.append(visited_cells)

            self.logger.info("Found " + str( len(found_ground_cells) ) + " ground cells")
            for cell in found_ground_cells:
                try:
                    ground_cell_pos_queue.remove(cell)
                except:
                    self.logger.info("Did not found " + str(cell))
            self.logger.info("Now " + str( len(ground_cell_pos_queue) ) + " cells left in ground_cell_pos_queue")
        
        for island in all_islands:
            self.logger.info(str(len(island)))
            

    def is_traversable_from_pos(self, neigbour_pos, ground_pos):
        return abs(self.cell_elev_z_grid[neigbour_pos] - self.cell_elev_z_grid[ground_pos]) <= STEP_SIZE

    def paint_cell(self, cell_idx, color):
        self.all_cells_pcd_list[cell_idx] = self.all_cells_pcd_list[cell_idx].paint_uniform_color(color)
    
    def assign_traversable_points(self, cell_idx, cell_pos):
        x_idx = cell_pos[0]
        y_idx = cell_pos[1]
        min_x = self.min_bounds[0]
        min_y = self.min_bounds[1]
        x = min_x + x_idx * ROBOT_SIZE + ROBOT_SIZE/2
        y = min_y + y_idx * ROBOT_SIZE + ROBOT_SIZE/2
        z = self.cell_elev_z_grid[cell_pos]

        ground_level_bbox = o3d.geometry.AxisAlignedBoundingBox([x-ROBOT_SIZE/2, y-ROBOT_SIZE/2, z - STEP_SIZE], [x+ROBOT_SIZE/2, y+ROBOT_SIZE/2, z])

        points_in_cell = self.all_cells_pcd_list[cell_idx].points
        new_traversable_points_idx = ground_level_bbox.get_point_indices_within_bounding_box(self.pcd.points)
        if len(new_traversable_points_idx) > MIN_POINTS_IN_CELL:
            self.traversable_points_idx = np.append(self.traversable_points_idx, new_traversable_points_idx)
            self.nbr_of_ground_level_points += len(new_traversable_points_idx)

    def get_painted_ground_points_pcd(self, cell_idx, cell_pos):
        #x = self.all_cells_pcd_list[cell_idx].get_center()[0]
        #y = self.all_cells_pcd_list[cell_idx].get_center()[1]
        x_idx = cell_pos[0]
        y_idx = cell_pos[1]
        min_x = self.min_bounds[0]
        min_y = self.min_bounds[1]
        x = min_x + x_idx * ROBOT_SIZE + ROBOT_SIZE/2
        y = min_y + y_idx * ROBOT_SIZE + ROBOT_SIZE/2
        z = self.cell_elev_z_grid[cell_pos]

        ground_level_bbox = o3d.geometry.AxisAlignedBoundingBox([x-ROBOT_SIZE/2, y-ROBOT_SIZE/2, z - STEP_SIZE], [x+ROBOT_SIZE/2, y+ROBOT_SIZE/2, z + STEP_SIZE])
        ground_level_points = self.all_cells_pcd_list[cell_idx].crop(ground_level_bbox)
        ground_level_points.translate([0,0,0.03])

        points_in_cell = self.all_cells_pcd_list[cell_idx].points
        new_traversable_points_idx = ground_level_bbox.get_point_indices_within_bounding_box(self.pcd.points)
        self.traversable_points_idx = np.append(self.traversable_points_idx, new_traversable_points_idx)

        if x < 29.96999931-TUNNELOFFSET_X and y < 29.70000076-TUNNELOFFSET_Y:
            self.nbr_of_ground_level_points += np.asarray(ground_level_points.points).shape[0]

        if len(ground_level_points.points) > MIN_POINTS_IN_CELL:
            if self.cell_classes_grid[cell_pos] == GROUND:
                self.ground_points = np.append(self.ground_points, np.asarray(ground_level_points.points), axis=0)
                return ground_level_points.paint_uniform_color(np.array([1,0,0]))
            elif self.cell_classes_grid[cell_pos] == INCLINED:
                self.ground_points = np.append(self.ground_points, np.asarray(ground_level_points.points), axis=0)
                return ground_level_points.paint_uniform_color(np.array([1,1,0]))

        return False

    def get_ground_cells_pos_list(self, ground_z):
        ground_cells_pos_list = []
        x_poses, y_poses = np.where( self.is_ground_cell(self.cell_elev_z_grid, ground_z))
        for idx in range(len(x_poses)):
            cell_pos = (x_poses[idx], y_poses[idx])
            ground_cells_pos_list.append( cell_pos )

        x_poses, y_poses = np.where(self.cell_classes_grid == INCLINED)
        for idx in range(len(x_poses)):
            cell_pos = (x_poses[idx], y_poses[idx])
            ground_cells_pos_list.append( cell_pos )
        return ground_cells_pos_list

    def get_cell_pcd(self, x, y, min_z, max_z):
        start_get_cell_pcd = timeit.default_timer()
        z_values = np.linspace(min_z, max_z, 20)
        
        x_center = x-ROBOT_SIZE/2
        y_center = y-ROBOT_SIZE/2
        cell_points_idx = np.asarray([], dtype=np.int)
        total_search = 0
    
        for z in z_values:
            center_point = np.asarray([x_center, y_center, z])
            radius = ROBOT_SIZE
            [k, idx, _] = self.pcd_kdtree.search_radius_vector_3d(center_point, radius)
            cell_points_idx = np.append(cell_points_idx, idx)
        
        if len(cell_points_idx):
            cell_points_idx = np.unique(cell_points_idx)
            cell_points = np.asarray(self.pcd.points)[cell_points_idx[1:], :]

            cell_pcd = o3d.geometry.PointCloud()
            cell_pcd.points = o3d.utility.Vector3dVector(cell_points)
            return cell_pcd
  
        return False
    
    def print_class_results(self):
        total = 0
        x_poses, y_poses = np.where(self.cell_classes_grid == GROUND)
        total+=len(x_poses)
        self.logger.info("GROUND: " + str(len(x_poses)))

        x_poses, y_poses = np.where(self.cell_classes_grid == INCLINED)
        self.logger.info("INCLINED: " + str(len(x_poses)))
        total+=len(x_poses)

        x_poses, y_poses = np.where(self.cell_classes_grid == UNKNOWN)
        total+=len(x_poses)
        self.logger.info("UNKNOWN: " + str(len(x_poses)))
        x_poses, y_poses = np.where(self.cell_classes_grid == WALL)
        total+=len(x_poses)

        self.logger.info("UNTRAVERSABLE: " + str(len(x_poses)))
        self.logger.info("TOTAL: " + str(total))

    def get_results(self):        
        self.nbr_of_covered_inclined_ground_level_cells = 0
        ground_cells_pos_list = self.get_ground_cells_pos_list() #THIS NEEDS TO BE CHANGED
        angles = []
        angles_pos = []
        for ground_cell_pos in ground_cells_pos_list:
            elev_level = self.cell_elev_z_grid[ground_cell_pos]    
            


            '''
            color = [0.0, 0.0, 1.0]
            for neigbour_pos in self.get_neighbour(ground_cell_pos):
                if self.cell_classes_grid[neigbour_pos] == GROUND or self.cell_classes_grid[neigbour_pos] == INCLINED: 
                    if abs(self.cell_elev_z_grid[neigbour_pos] - elev_level) > 0.24:
                        color = [0.0, 1.0, 1.0]
                        break
            '''
            #self.logger.info(str(ground_cell_pos))
            idx_in_list = int(self.cell_list_idx_grid[ground_cell_pos])
            #self.logger.info(str(idx_in_list))
            pcd = self.all_cells_pcd_list[idx_in_list]
            #self.all_cells_pcd_list[idx_in_list].paint_uniform_color(np.array([0.0,0.0,1.0]))
            
            xy_center = pcd.get_center()[0:2]

            x_idx = ground_cell_pos[0]
            y_idx = ground_cell_pos[1]
            min_x = self.min_bounds[0]
            min_y = self.min_bounds[1]
            xy_center = (min_x + x_idx * ROBOT_SIZE + ROBOT_SIZE/2, min_y + y_idx * ROBOT_SIZE + ROBOT_SIZE/2) 
            
            center = np.asarray([xy_center[0], xy_center[1], elev_level])

            polygon = []
            x1 = center[0] + 0.50 * ROBOT_SIZE
            x2 = center[0] - 0.50 * ROBOT_SIZE
            #x2 = center[0] + 0.5 * ROBOT_SIZE
            #x1 = center[0] - 0.5 * ROBOT_SIZE
            #z1 = center[2] + 0.5 * ROBOT_SIZE
            #z2 = center[2] - 0.5 * ROBOT_SIZE
            for point in range(16):
                angle = point/16*np.pi*2
                y = center[1] + np.cos(angle) * ROBOT_SIZE * 0.50
                z = center[2] + np.sin(angle) * ROBOT_SIZE * 0.50
                
                polygon.append(
                    [0, y, z]
                )
            #polygon.append([x1, 0, z2])
            #polygon.append([x2, 0, z2])
            #polygon.append([x1, 0, z2])
            #polygon.append([x1, 0, z1])
            #polygon.append([x2, 0, z1])
            #polygon.append([x1, 0, z1])
            #polygon.append([x2, 0, z1])
            #polygon.append([x2, 0, z2])

            polygon_bbox = o3d.visualization.SelectionPolygonVolume()
            polygon_bbox.orthogonal_axis = "X"
            polygon_bbox.axis_max = float(x1)
            polygon_bbox.axis_min = float(x2)
            polygon_bbox.bounding_polygon = o3d.utility.Vector3dVector(np.asarray(polygon).astype("float64"))

            #self.logger.info("my axis_max:" + str(y2))
            #self.logger.info("my axis_min:" + str(y1))
            #self.logger.info("my bounding_polygon:" + str(np.asarray(polygon_bbox.bounding_polygon)))
            polygon_pcd = polygon_bbox.crop_point_cloud(pcd)


            #x = center[0]
            #y = center[1]
            #z = center[2]
            #ground_level_bbox = o3d.geometry.AxisAlignedBoundingBox([x-ROBOT_SIZE/2, y-ROBOT_SIZE/2, z - 0.5], [x+ROBOT_SIZE/2, y+ROBOT_SIZE/2, z + 0.5])
            #ground_level_points = pcd.crop(ground_level_bbox)

            
            mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=ROBOT_SIZE/2,
                                                          height=ROBOT_SIZE)
            mesh_cylinder.compute_vertex_normals()
            mesh_cylinder.paint_uniform_color([0.3, 0.7, 0.3])
            mesh_cylinder.translate(center + [0.0, 0.0, 0.00])
            R = o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, np.pi/2, 0])
            mesh_cylinder.rotate(R, center=center)
            #self.logger.info("my pcd: " + str(np.asarray(polygon_pcd.points).shape))
            #self.logger.info("bbox:" + str(np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points())))
            #self.logger.info("polygon:" + str(np.asarray(polygon_bbox.bounding_polygon)))
            #Fakse polygon:
            #bbox_points = np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points())
            #polygon_bbox.axis_max = np.max(bbox_points[:,1])
            #polygon_bbox.axis_min = np.min(bbox_points[:,1])
            #bbox_points[:,1] = 0 
            #polygon_bbox.bounding_polygon = o3d.utility.Vector3dVector(bbox_points)

            #self.logger.info("fake axis_max:" + str(polygon_bbox.axis_max))
            #self.logger.info("fake axis_min:" + str(polygon_bbox.axis_min))
            #self.logger.info("fake bounding_polygon:" + str(np.asarray(polygon_bbox.bounding_polygon)))
            
            #polygon_pcd = polygon_bbox.crop_point_cloud(pcd)

            polygon_pcd.paint_uniform_color([0.0,1.0,0.0])
            polygon_pcd.translate([0.0, 0.0, 0.06], True)
            #self.logger.info("fake pcd: " + str(np.asarray(polygon_pcd.points).shape))
            #mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
            #                                    height=1.0,
            #                                    depth=0.05)
            if x1 < 29.96999931-TUNNELOFFSET_X and y < 29.70000076-TUNNELOFFSET_Y:
                self.nbr_of_covered_ground_level_points += np.asarray(polygon_pcd.points).shape[0]
                if self.cell_classes_grid[ground_cell_pos] == INCLINED:
                    self.nbr_of_covered_inclined_ground_level_cells += 1
            #self.logger.info(str(center))
            #mesh_box.compute_vertex_normals()
            #mesh_box.paint_uniform_color(color)
            #mesh_box.translate(center)
            #self.full_pcd.append(polygon_pcd)
            #self.full_pcd.append(mesh_cylinder)

            #pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=1000))
            #pcd = pcd.normalize_normals()
            #kdtree = o3d.geometry.KDTreeFlann(pcd)
            #[k, idx, _] = kdtree.search_knn_vector_3d(center, 1)
            #idx_in_pcd = idx[0]
            #normal = np.asarray(pcd.normals)[idx_in_pcd]
            #angle = np.arccos(normal[2])
            #if angle > np.pi/2:
            #    angle = np.pi - angle
            #angles.append(angle)
         
            
            #self.logger.info(str(np.asarray(pcd.normals)[idx_in_pcd]))
            #self.logger.info(str(angle))
            #closest_point = 
            #Get pcd of cellground_cells_pos_list
            #get center
            #get elevation level
            #get closest point
            #get normal of that point
            #get maximum inclination
        
        
        part = float(self.nbr_of_covered_ground_level_points) / float(self.nbr_of_ground_level_points)
        self.logger.info("Ground level: " + str(self.nbr_of_ground_level_points))
        self.logger.info("Covered ground level: " + str(self.nbr_of_covered_ground_level_points))
        self.logger.info("Percent: " + str(part))
        self.logger.info("Inclined cells: " + str(self.nbr_of_covered_inclined_ground_level_cells))
        '''
        self.logger.info(str(self.ground_points[0:5]))
        self.logger.info(str(len(self.ground_points)))
        
        ground_pcd_idx = np.where(self.ground_points == [0.5,0.5,0.5])
        
        ground_cells_pos_list = self.get_ground_cells_pos_list()
        self.logger.info(str(np.asarray(self.pcd.colors)[0:10]))
        ground_pcd_idx = np.where(np.asarray(self.pcd.colors) == [0.5,0.5,0.5])
        
        self.logger.info(str(ground_pcd_idx[0:10]))
        self.logger.info(str(len(ground_cells_pos_list)))
        self.logger.info(str(len(ground_pcd_idx)))
        '''



            

                       