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
FLOOR_NR = 1

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

        start_floor_segmentation = timeit.default_timer()
        z_values = np.linspace(min_z, max_z, NUMBER_OF_LAYERS)
        number_of_points_in_layers = []
        potential_maximas = []
        for idx in range(NUMBER_OF_LAYERS - 1):
            layer = o3d.geometry.AxisAlignedBoundingBox( np.array([min_x, min_y, z_values[idx]]), [max_x, max_y, z_values[idx + 1]])
            cropped = self.pcd.crop(layer)
            number_of_points_in_layer = len(cropped.points)
            number_of_points_in_layers.append(number_of_points_in_layer)
            
            if idx > 2:
                number_of_points_in_prev_layer = number_of_points_in_layers[idx-1]
                number_of_points_in_prev_prev_layer = number_of_points_in_layers[idx-2]
                prev_is_local_maximum = number_of_points_in_prev_layer > number_of_points_in_layer and number_of_points_in_prev_layer > number_of_points_in_prev_prev_layer

                if prev_is_local_maximum and number_of_points_in_prev_layer > GROUND_THRESHOLD: 
                    potential_maximas.append(idx-1)
        
        self.logger.info(str(z_values[potential_maximas]))
        #plt.plot(z_values[1:], number_of_points_in_layers)
        #plt.ylabel('some numbers')
        #plt.show()
        floor_levels = z_values[potential_maximas]
        floor_1 = self.pcd.crop(o3d.geometry.AxisAlignedBoundingBox( np.array([min_x, min_y, floor_levels[0] ]), [max_x, max_y, floor_levels[1]]))
        floor_2 = self.pcd.crop(o3d.geometry.AxisAlignedBoundingBox( np.array([min_x, min_y, floor_levels[1] ]), [max_x, max_y, max_z]))
        
        


        nbr_of_x = int(np.ceil((max_x-min_x) / ROBOT_SIZE)) 
        nbr_of_y = int(np.ceil((max_y-min_y) / ROBOT_SIZE)) 
        x_values = np.linspace(min_x, max_x, nbr_of_x) 
        y_values = np.linspace(min_y, max_y, nbr_of_y) 


        if FLOOR_NR == 2:
            ground_floor_z = floor_levels[1]
            top_floor_z =  max_z
        else:
            ground_floor_z = floor_levels[0]
            top_floor_z = floor_levels[1]

        end_floor_segmentation = timeit.default_timer()
        self.logger.info("Floor segmentation: " + str(end_floor_segmentation - start_floor_segmentation))
        start_find_elevation = timeit.default_timer()

        ground_cells = []
        ground_cells_idx = []
        ground_cells_first_elev_level = []
        ground_cells_second_elev_level = []
        all_cells_coordinates = []

        wall_cells = []
        cells = []

        #grid_pcl = np.empty((nbr_of_x, nbr_of_y))
        grid_class = np.empty((nbr_of_x, nbr_of_y))
        grid_pcd_idx = np.empty((nbr_of_x, nbr_of_y))
        grid_class.fill(UNKNOWN)
        grid_first_elev_level = np.empty((nbr_of_x, nbr_of_y))
        grid_second_elev_level = np.empty((nbr_of_x, nbr_of_y))

        total_geom = 0
        total_loop = 0
        total_assign = 0
        total_downsample = 0

        for x_idx, x in enumerate(x_values[1:]):
            for y_idx, y in enumerate(y_values[1:]):
                
                
                
                cell = o3d.geometry.AxisAlignedBoundingBox( np.array([x-ROBOT_SIZE, y-ROBOT_SIZE, ground_floor_z]), [x, y, top_floor_z])
                start_downsample = timeit.default_timer()
                #down_sampled_pcd = self.pcd.uniform_down_sample(10)
                #down_sampled_pcd = self.pcd.voxel_down_sample(0.5)
                end_downsample  = timeit.default_timer()
                total_downsample += (end_downsample - start_downsample)
                
                start_geom = timeit.default_timer()
                #cropped = down_sampled_pcd.crop(cell)
                cropped = self.pcd.crop(cell)
                #grid_pcl[x, y] = cropped 
                
    
                z_values_in_cell = np.asarray(cropped.points)[:,2]
                end_geom = timeit.default_timer()

                
                total_geom += (end_geom - start_geom)

                nbr_of_z_values = len(z_values_in_cell)
                first_elev_level = False
                first_level_free_space = False
                second_elev_level = False
                second_level_free_space = False

                if nbr_of_z_values > MIN_POINTS_IN_CELL:
                    #z_values_in_cell = np.append(z_values_in_cell, ground_floor_z)
                    z_values_in_cell = np.append(z_values_in_cell, top_floor_z)
                    first_level_found = False
                    second_level_found = False

                    sorted_z_values = np.sort(z_values_in_cell)
                    #self.logger.info("nbr_of_z_values" + str(nbr_of_z_values))
                    # Find first level free space:
                    start_loop = timeit.default_timer()


                    for idx, z in enumerate(sorted_z_values[1:]):
                        #self.logger.info("idx" + str(idx))
                        #self.logger.info("z" + str(z))
                        prev_z = sorted_z_values[idx]
                        
                        if abs(z - prev_z) > ROBOT_HEIGHT:
                            #first_level_free_space = z
                            first_level_found = True
                            break
                    
                    
                    first_elev_level = prev_z

                    end_loop = timeit.default_timer()

                    total_loop += (end_loop - start_loop)

                    '''
                    if first_level_found:
                        reversed_sorted_z_values = np.flip(sorted_z_values[idx+1:])
                        
                        for new_idx, z in enumerate(reversed_sorted_z_values[1:]):
                            #self.logger.info("new_idx" + str(new_idx))
                            #self.logger.info("z" + str(z))
                            prev_z = reversed_sorted_z_values[new_idx]
                            if abs(z - prev_z) > ROBOT_HEIGHT:
                                second_elev_level = z
                                second_level_free_space = prev_z
                                second_level_found = True
                                break
                    '''
                        
                    #if second_elev_level:
                    #    self.logger.info("sorted_z_values" + str(sorted_z_values))
                    #    self.logger.info("idx" + str(idx))
                    #    self.logger.info("new_idx " + str(new_idx))
                    #    # Find second level free space:
                    #    self.logger.info("reversed_sorted_z_values" + str(reversed_sorted_z_values))
                    #    self.logger.info(str([first_elev_level, first_level_free_space, second_elev_level, second_level_free_space]))
                    #self.logger.info(str(ground_floor_z + STEP_SIZE))

                    start_assign = timeit.default_timer()

                    if first_elev_level < ground_floor_z + STEP_SIZE:
                        #cropped.paint_uniform_color(np.array([0,1,0]))
                        #ground_cells.append( (x,y) )
                        #ground_cells_idx
                        #ground_cells_first_elev_level.appen(first_elev_level)
                        #ground_cells_second_elev_level.appen(second_elev_level)
                        #self.logger.info(str(grid_class))
                        ground_cells.append((x_idx, y_idx))
                        grid_class[x_idx][y_idx] = GROUND
                        #if second_level_found:
                        #    grid_first_elev_level[x_idx][y_idx] = first_elev_level
                    
                    grid_first_elev_level[x_idx][y_idx] = first_elev_level
                    
                    
                    

                    #elif not first_level_found and not second_level_found:
                    #    #cropped.paint_uniform_color(np.array([1,0,0]))
                    #    grid_class[x_idx][y_idx] = WALL 
                    #else:
                    #    pass
                        #cropped.paint_uniform_color(np.array([0,0,0]))
                        #wall_cells.append(cropped)
                    all_cells_coordinates.append((x_idx, y_idx))
                    grid_pcd_idx[x_idx][y_idx] = len(cells)
                    cells.append( cropped )

                    end_assign = timeit.default_timer()

                    total_assign += (end_assign - start_assign)
                    
                    #plt.title("x,y: " + str(round(x)) + "," + str(round(y)) + " tracersable: " + str(potentionally_traversable_cell))
                    #plt.plot([1] * len(z_values_in_cell), z_values_in_cell, 'o')
                    #plt.show()

        end_find_elevation = timeit.default_timer()
        self.logger.info("total_downsample: " + str(total_downsample)) 
        self.logger.info("total_geom: " + str(total_geom))
        self.logger.info("total_loop: " + str(total_loop))
        self.logger.info("total_assign: " + str(total_assign)) 
        
        

        self.logger.info("Find elevation: " + str(end_find_elevation - start_find_elevation))
        start_breadth_first_search = timeit.default_timer()

        #for floor_idx in range(len(floor_levels)-1):
        #    floor_box = o3d.geometry.AxisAlignedBoundingBox( np.array([min_x, min_y, min_z ]), [max_x, max_y, floor_levels[floor_idx+1]])
        #    cropped = self.pcd.crop(floor_box)

        #self.logger.info(str(grid_class))
        '''
        elif first_elev_level:
                        for neighbour in self.get_neighbour((x_idx, y_idx), nbr_of_x, nbr_of_y):
                            if grid_class[neighbour] == GROUND and abs(first_elev_level - grid_first_elev_level[neighbour]) < STEP_SIZE:
                                ground_cells.append((x_idx, y_idx))
                                grid_class[x_idx][y_idx] = INCLINED
                                grid_first_elev_level[x_idx][y_idx] = first_elev_level
                                grid_second_elev_level[x_idx][y_idx] = second_elev_level
                                break
        '''
        #ground_cells_idx = np.where(grid_class == GROUND)
        #self.logger.info(str(len(ground_cells)))
        #self.logger.info(str(ground_cells))
        #border_cells_coordinates = []
        #for cell in ground_cells:
        #    for neigbour in self.get_neighbour(cell, nbr_of_x, nbr_of_y):
        #        if grid_class[neighbour] != GROUND
        
        q = ground_cells #collections.deque(ground_cells)
        while len(q):
            #self.logger.info(str(len(q)))
            cell = q.pop()
            for neigbour in self.get_neighbour(cell, nbr_of_x, nbr_of_y):
                
                #self.logger.info(str(neigbour))
                #self.logger.info(str(grid_first_elev_level[neigbour]))
                #self.logger.info(str(grid_first_elev_level[cell]))
                #self.logger.info(str(grid_class[neigbour]))
                if grid_class[neigbour] == UNKNOWN and abs(grid_first_elev_level[neigbour] - grid_first_elev_level[cell]) <= STEP_SIZE:
                    grid_class[neigbour] = INCLINED
                    #self.logger.info("INCLINED!")
                    q.append(neigbour)
                elif grid_class[neigbour] == UNKNOWN:
                    grid_class[neigbour] = WALL
                
        for cell in all_cells_coordinates:
            idx = int(grid_pcd_idx[cell])

            if grid_class[cell] == GROUND or grid_class[cell] == INCLINED:

                cells[idx] = cells[idx].paint_uniform_color(np.array([0,0,0]))
                if abs(float(grid_first_elev_level[cell]))>0.01 and abs(float(grid_first_elev_level[cell]))<15:
                    x = cells[idx].get_center()[0]
                    y = cells[idx].get_center()[1]
                    z = grid_first_elev_level[cell]
                    #self.logger.info(str([x-ROBOT_SIZE/2, y-ROBOT_SIZE/2, z-0.25]))
                    #self.logger.info(str([x+ROBOT_SIZE/2, y+ROBOT_SIZE/2, z+0.25]))
                    floor_level_bbox = o3d.geometry.AxisAlignedBoundingBox([x-ROBOT_SIZE, y-ROBOT_SIZE/2, z-1], [x+ROBOT_SIZE, y+ROBOT_SIZE/2, z+1])
                    floor_cells = cells[idx].crop(floor_level_bbox)

                    if len(floor_cells.points) > MIN_POINTS_IN_CELL:
                        if grid_class[cell] == GROUND:
                            floor_cells.paint_uniform_color(np.array([0,1,0]))
                        if grid_class[cell] == INCLINED:
                            floor_cells.paint_uniform_color(np.array([0,0,1]))

                        floor_cells.translate([0,0,0.03])
                        #cell_bbox =  o3d.geometry.OrientedBoundingBox()
                        #cell_bbox.create_from_axis_aligned_bounding_box(cells[idx].get_axis_aligned_bounding_box())
                        #rest_cells = cells[idx].crop(floor_level_bbox)#, cell_bbox)
                        #rest_cells.paint_uniform_color(np.array([0,0,0]))
                        #cells[idx] = rest_cells
                        cells.append(floor_cells)
                    #cells.append(floor_level_bbox)
                #cells[idx] = cells[idx].paint_uniform_color(np.array([0,1,0]))
                
                
            elif grid_class[cell] == WALL:
                #self.logger.info(str(grid_first_elev_level[cell]))
                cells[idx] = cells[idx].paint_uniform_color(np.array([1,0,0]))
                
                #cells[idx] = cells[idx]
            elif grid_class[cell] == UNKNOWN:
                cells[idx] = cells[idx].paint_uniform_color(np.array([1,0,1]))
            else:
                cells[idx] = cells[idx].paint_uniform_color(np.array([0,0,0]))
            
            '''
            if grid_class[cell] == GROUND or grid_class[cell] == UNKNOWN:
                #self.logger.info("first: " + str(grid_first_elev_level[cell]))
                #self.logger.info("sec: " + str(grid_second_elev_level[cell]))
                if abs(float(grid_first_elev_level[cell]))>0.01 and abs(float(grid_first_elev_level[cell]))<15:
                    #self.logger.info("2first: " + str(grid_first_elev_level[cell]))
                    #self.logger.info("2sec: " + str(grid_second_elev_level[cell]))
                    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.1)
                    mesh_box.compute_vertex_normals()
                    mesh_box.paint_uniform_color([1.0, 1.0, 0.0])
                    x = cells[idx].get_center()[0]
                    y = cells[idx].get_center()[1]
                    z = grid_first_elev_level[cell] - 0.5
                    mesh_box.translate([ x,y,z ])
                    cells.append(mesh_box)

                
                if abs(float(grid_second_elev_level[cell]))>0.01 and abs(float(grid_second_elev_level[cell]))<15:
                        mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.1)
                        mesh_box.compute_vertex_normals()
                        mesh_box.paint_uniform_color([0.0, 1.0, 1.0])
                        x = cells[idx].get_center()[0]
                        y = cells[idx].get_center()[1]
                        z = grid_second_elev_level[cell]
                        mesh_box.translate([ x,y,z ])
                        cells.append(mesh_box)
            '''
        end_breadth_first_search = timeit.default_timer()
        self.logger.info("Breadth first: " + str(end_breadth_first_search - start_breadth_first_search))
        end_total = timeit.default_timer()        
        self.logger.info("Total: " + str(end_total - start_total))

        o3d.visualization.draw_geometries(cells,
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
        
        
        return []


    def get_neighbour(self, cell, nbr_of_x, nbr_of_y):
        directions = [(1,0), (0,1), (-1, 0), (0, -1), (1, 1), (-1,1), (-1, -1), (1, -1)]
        neighbours = []
        for dir in directions:
            neighbour = tuple(map(operator.add,cell,dir))
            if neighbour[0] >= 0 and neighbour[0] < nbr_of_x and neighbour[1] >= 0 and neighbour[1] < nbr_of_y:
                neighbours.append(neighbour)

        return neighbours


    def get_random_poses(self):
        poses = []

        while len(poses) < 100:
            random_idx = np.random.randint(0, len(self.points))
            pos = self.points[random_idx]
            pose = self.get_pose_in_pos(pos)
            poses.append(pose)

        return poses


    def get_neighbour_pos(self, pos):
        #neighbours = np.empty((1,3))
        resolution = STEP_RESOLUTION
        #self.logger.info(str(pos))
        nearest_point = self.find_k_nearest(pos, 1)[0]
        #OPTIMISE THIS
        while np.linalg.norm(pos - nearest_point) < 0.1:
            pos += np.array([0.0, 0.0, 0.05])
            nearest_point = self.find_k_nearest(pos, 1)[0]
            
        #self.logger.info(str(np.linalg.norm(pos - nearest_point)))
        #self.logger.info("nearest: " + str(nearest_point))
        #if np.linalg.norm(pos[0:2] - nearest_point[0:2]) > 0.5:
        #    return []
        
        positive_x = np.array([resolution, 0.0, 0.0])
        negative_x = np.array([-resolution, 0.0, 0.0])
        positive_y = np.array([0.0, resolution, 0.0])
        negative_y = np.array([0.0, -resolution, 0.0])
        neighbours = np.array([nearest_point + negative_y])
        for direction in [negative_x, positive_y, positive_x]:
            #self.logger.info(str(np.array([nearest_point + direction])))
            #self.logger.info(str(neighbours))
            #self.logger.info(str(np.array([nearest_point + direction]).shape))
            #self.logger.info(str(neighbours.shape))
            neighbours = np.append(neighbours, np.array([nearest_point + direction]), axis=0)
        #self.logger.info("neighbours: " + str(neighbours))
        #self.logger.info("neighbours shape;" + str(neighbours.shape))
        return neighbours
    
    def not_close_to_visited(self, pos, visited):
        for pose in visited:
            if np.linalg.norm(pos - pose) < 0.9*STEP_RESOLUTION:
                return False
        return True
    
    def get_pose_in_pos(self, start_pos):
        k = 500
        k_nearest = self.find_k_nearest(start_pos, k)
        center_point = np.average(k_nearest, axis=0)
        covariance_matrix = np.cov( k_nearest-center_point, rowvar=False )
        eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
        smallest_eig_val_idx = np.argmin(eig_vals)
        normal_vector = eig_vecs[:, smallest_eig_val_idx]
        if normal_vector[2] < 0:
            normal_vector = -normal_vector
        #self.logger.info(str(np.average(k_nearest, axis=0)))
        #self.logger.info(str(np.corrcoef( k_nearest-center_point, rowvar=False )))
        #self.logger.info("eig_vals " + str(eig_vals))
        #self.logger.info("eig_vecs " + str(eig_vecs))
        #self.logger.info("smallest_eig_val_idx " + str(smallest_eig_val_idx))
        #self.logger.info("normal_vector " + str(normal_vector))

        new_pose = Pose()
        new_pose.k_nearest_points = np.array(k_nearest)
        new_pose.center_point = center_point
        new_pose.normal = normal_vector/np.linalg.norm(normal_vector)
        pitch = np.math.asin(new_pose.normal[2])
        self.logger.info(str(pitch))
        if pitch < np.pi/4:
            new_pose.traversable = False

        return new_pose


    def find_k_nearest(self, point, k):
        #distances = np.linalg.norm(self.points - point, axis=1)
        #nearest_point_idx = np.argmin(distances)
        #sorted_points = self.points[np.argsort(distances)]
        [k, idx, _] = self.pcd_kdtree.search_knn_vector_3d(point, k)
        #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        return self.points[idx, :]

