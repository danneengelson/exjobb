
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import timeit
from exjobb.Floor import Floor
from exjobb.Parameters import Z_RESOLUTION, GROUND_OFFSET

MIN_FLOOR_HEIGHT = 2
GROUND_FLOOR_THRESSHOLD = 100000 
VISUALIZE = True
SHOW_HISTOGRAM = False 

class FloorSegmentation:

    def __init__(self, print):
        self.print = print

    def get_segmentized_floors(self, pcd):
        '''
        Divides the point cloud into floors. 
        Returns a list of instances of class Floor.
        '''
        start = timeit.default_timer()
        bounding_box = self.get_bounding_box(pcd)
        layers = self.get_layers(pcd, bounding_box)

        points_idx_in_layers = np.array([layer["points_idx"] for layer in layers])
        nbr_of_points_in_layers = np.array([layer["nbr_of_points"] for layer in layers])
        start_z_of_layers = np.array([layer["start_z"] for layer in layers])

        potential_ground_layers_idx = self.get_local_maximas_idx(nbr_of_points_in_layers)  
        ground_layers_idx = self.get_ground_layers_idx(potential_ground_layers_idx, start_z_of_layers)

        segmentised_floors = []
        ground_layer_idx = ground_layers_idx[0]
        last_layer_idx = len(layers)
        ground_layers_idx.append(last_layer_idx)

        for floor_nr, next_ground_layer_idx in enumerate(ground_layers_idx[1:]):
            segmentised_floor = {}
            points_idx = np.array([]) 
            for idx in range(ground_layer_idx, next_ground_layer_idx):
                points_idx = np.append(points_idx, points_idx_in_layers[idx])
            
            floor = Floor("Floor " + str(floor_nr+1), pcd, points_idx)
            
            segmentised_floors.append(floor)

            ground_layer_idx = next_ground_layer_idx

        self.print_result(segmentised_floors, start)

        if VISUALIZE:
            self.visualze_segmentized_floors(segmentised_floors)

        return segmentised_floors



    def get_layers(self, pcd, bounding_box):
        layers = []
        layers_start_z = np.arange(bounding_box["min_z"], bounding_box["max_z"], Z_RESOLUTION)
        for z in layers_start_z:
            layer_bounding_box = o3d.geometry.AxisAlignedBoundingBox(
                [bounding_box["min_x"], bounding_box["min_y"], z],
                [bounding_box["max_x"], bounding_box["max_y"], z + Z_RESOLUTION]
            )
            points_idx_in_layer = layer_bounding_box.get_point_indices_within_bounding_box(pcd.points)
            layers.append({
                "points_idx": points_idx_in_layer,
                "start_z": z,
                "nbr_of_points": len(points_idx_in_layer)
            })
        return layers

    def get_local_maximas_idx(self, values):
        
        potential_maximas = np.array(argrelextrema(values, np.greater))
        potential_maximas = potential_maximas[values[potential_maximas] > GROUND_FLOOR_THRESSHOLD] 
        potential_maximas = potential_maximas - int(GROUND_OFFSET/Z_RESOLUTION) # offset to make sure we dont't miss any points.
        self.print(potential_maximas)
        if SHOW_HISTOGRAM:
            #For tuning GROUND_FLOOR_THRESSHOLD:
            plt.plot(np.arange(0, len(values))/10, values)
            plt.plot(np.arange(0, len(values))/10, 100000*np.ones((len(values), 1)))
            plt.ylabel('Number of points')
            plt.xlabel('Height [m]')
            plt.title('Number of Points on Different Heights')
            plt.show()

        return potential_maximas


    def get_ground_layers_idx(self, potential_ground_layers_idx, start_z_of_layers):
        ground_layers_idx = []
        potential_ground_layer_idx = potential_ground_layers_idx[0]
        potential_grounds_z = start_z_of_layers[potential_ground_layer_idx]
        for idx in potential_ground_layers_idx[1:]:
            if start_z_of_layers[idx] - potential_grounds_z > MIN_FLOOR_HEIGHT:
                ground_layers_idx.append(potential_ground_layer_idx)
            potential_ground_layer_idx = idx
            potential_grounds_z = start_z_of_layers[idx]
        
        ground_layers_idx.append(potential_ground_layer_idx)
        
        return ground_layers_idx
        

    def get_bounding_box(self, pcd):
        bounding_box = pcd.get_axis_aligned_bounding_box()
        min_bounds = np.min(np.asarray(bounding_box.get_box_points()), axis=0)
        max_bounds = np.max(np.asarray(bounding_box.get_box_points()), axis=0)
        bounding_box_info = {}
        bounding_box_info["min_x"] = min_bounds[0]
        bounding_box_info["min_y"] = min_bounds[1]
        bounding_box_info["min_z"] = min_bounds[2]
        bounding_box_info["max_x"] = max_bounds[0]
        bounding_box_info["max_y"] = max_bounds[1]
        bounding_box_info["max_z"] = max_bounds[2] 
        return bounding_box_info

    def print_result(self, segmentised_floors, start):
        end = timeit.default_timer()
        self.print("="*20)
        self.print("FLOOR SEGMENTATION")
        self.print("Number of floors: " + str(len(segmentised_floors)))
        self.print("Computational time: " + str(round(end - start, 1)) + " sec")
        for floor, segmentised_floor in enumerate(segmentised_floors):
            self.print("-"*20)
            self.print("Floor " + str(floor + 1))
            self.print("Ground: " + str(np.round(segmentised_floor.z_ground,2)))
            self.print("Ceiling: " + str(np.round(segmentised_floor.z_ceiling,2)))
            self.print("Number of points: " + str(len(segmentised_floor.points_idx_in_full_pcd)))
            
        self.print("="*20)

    def visualze_segmentized_floors(self, segmentized_floors):
        '''
        Not working properly right now.
        '''
        draw_elements = []
        for nr, floor in enumerate(segmentized_floors[0:2]):
            z = floor.min_z
            x = 0 #(floor.max_x - floor.min_x) / 2
            y = 0 #(floor.max_y - floor.min_y) / 2
            x_size = floor.max_x - floor.min_x
            y_size = floor.max_y - floor.min_y
            center = [x, y, z]
            #mesh_box = o3d.geometry.TriangleMesh.create_box(width= x_size, height=y_size, depth=0.05)
            #mesh_box.paint_uniform_color(np.array([1,0,0]))
            #mesh_box.translate(center)
            #draw_elements.append(mesh_box)
            pcd = floor.pcd
            pcd.paint_uniform_color(np.array([nr,0,1]))
            draw_elements.append(pcd)

        o3d.visualization.draw_geometries(draw_elements)
