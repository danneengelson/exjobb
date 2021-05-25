import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import operator
import collections
import timeit

from exjobb.FloorSegmentation import FloorSegmentation
from exjobb.ElevationDetector import ElevationDetector
from exjobb.TraversabilityDetector import TraversabilityDetector
from exjobb.Parameters import CELL_SIZE, ROBOT_STEP_SIZE
import pickle
import os

VISUALIZE = False
DO_TERRAIN_ASSESSMENT = True       
CALCULATE_COVERAGE_ON_RAMP = False

class TerrainAssessment():

    def __init__(self, logger, pcd, print):
        self.logger = logger
        self.pcd = pcd.raw_pcd
        self.points = pcd.points
        self.pcd_kdtree = pcd.kdtree
        self.traversable_points_idx = np.array([])
        self.print = print

        self.floor_segmentation = FloorSegmentation(print)
        self.elevation_detector = ElevationDetector(print)
        self.traversability_detector = TraversabilityDetector(print)
        

    def analyse_terrain(self):
        start_total = timeit.default_timer()
        self.full_pcd = []
        self.traversable_points_idx = np.array([], int)

        if DO_TERRAIN_ASSESSMENT:       
            segmentized_floors = self.floor_segmentation.get_segmentized_floors(self.pcd)
            for floor in segmentized_floors[0:2]:

                self.print("="*20)
                self.print(floor.name)                
                self.elevation_detector.find_elevation(self.pcd, floor)
                self.traversable_points_idx = np.append(self.traversable_points_idx, self.traversability_detector.get_traversable_points_idx(self.pcd, floor))

            with open('cached_segmentized_floors.dictionary', 'wb') as cached_pcd_file:
                floors = []
                for floor in segmentized_floors:
                    floors.append({
                        "name": floor.name,
                        "points_idx": floor.points_idx_in_full_pcd,
                        "cell_grid": floor.cells_grid,
                        "min_x": floor.min_x,
                        "min_y": floor.min_y,
                        "min_z": floor.min_z,
                        "max_z": floor.max_z
                    })
                cache_data = {
                    "segmentized_floors": floors,
                    "traversable_points_idx": self.traversable_points_idx,
                }
                pickle.dump(cache_data, cached_pcd_file)

        else:              


            with open('cached_segmentized_floors.dictionary', 'rb') as cached_pcd_file:
                cache_data = pickle.load(cached_pcd_file)
                segmentized_floors = cache_data["segmentized_floors"]
                self.traversable_points_idx = cache_data["traversable_points_idx"]
        
        covered_ground_pcd = False
        if CALCULATE_COVERAGE_ON_RAMP:



            covered_points, ground_points, draw_cylinders = self.get_covered_points(segmentized_floors)
            covered_ground_pcd_points = np.unique(covered_points, axis=0)
            covered_ground_pcd = o3d.geometry.PointCloud()
            covered_ground_pcd.points = o3d.utility.Vector3dVector(covered_ground_pcd_points)
            covered_ground_pcd.paint_uniform_color(np.array([0,0,1]))
            covered_ground_pcd.translate([0,0,0.03])
            self.print("Covered: " + str(len(covered_points)/ground_points))
            
        if VISUALIZE:
            draw_elements = []
            draw_elements.append( self.not_ground_pcd() )
            draw_elements.append( self.ground_pcd() )
            draw_elements.extend( draw_cylinders )
            if covered_ground_pcd:
                draw_elements.append( covered_ground_pcd )
            o3d.visualization.draw_geometries(draw_elements, zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024], point_show_normal=True)
  
            '''
            for x_idx, y_idx in np.ndindex(floor.cells_grid.shape):

                if floor.cells_grid[x_idx, y_idx] is None:
                    continue

                #if not floor.cells_grid[x_idx, y_idx].is_traversable:
                #    continue
                x = x_idx * CELL_SIZE + floor.min_x + CELL_SIZE/2
                y = y_idx * CELL_SIZE + floor.min_y + CELL_SIZE/2
                z = floor.cells_grid[x_idx, y_idx].elevation 
                z = floor.min_z
                center = [x,y,z]
                mesh_box = o3d.geometry.TriangleMesh.create_box(width= CELL_SIZE, height=CELL_SIZE, depth=0.05)
                #mesh_box.compute_vertex_normals()
                mesh_box.paint_uniform_color(np.array([1,0,0]))
                mesh_box.translate(center)
                #self.full_pcd.append(polygon_pcd)
                self.full_pcd.append(mesh_box)
            '''
            #self.pcd.paint_uniform_color([0.0,0.0,0.0])
            #removed_ground = np.asarray(self.pcd.points)
            #removed_ground_1 = np.delete(removed_ground, self.traversable_points_idx)
            #self.pcd.points = o3d.utility.Vector3dVector(removed_ground_1)
            #self.full_pcd.append( self.pcd )

            


            



            
            
            

            
            ##ground_pcd.translate([0,0,0.03])
            #self.full_pcd.append(ground_pcd)
            #self.ground_pcd = ground_pcd
            #self.print_results(segmentized_floors)

            
            #self.full_pcd.append(covered_ground_pcd)
            
            #nbr_of_ground_points = len(np.unique(np.asarray(ground_pcd.points), axis=0))
            #nbr_of_covered_ground_points = len(np.unique(np.asarray(covered_ground_pcd.points), axis=0))
            #self.print("ground_pcd: " + str(nbr_of_ground_points))
            #self.print("covered_ground_pcd: " + str(nbr_of_covered_ground_points))
            #self.print("percentage: " + str(nbr_of_covered_ground_points / nbr_of_ground_points))


            
        

 
        

        return 

    

    def not_ground_pcd(self):
        not_ground_pcd_points = np.asarray(self.pcd.points)
        not_ground_pcd_points = np.delete(not_ground_pcd_points, self.traversable_points_idx, axis=0)
        not_ground_pcd = o3d.geometry.PointCloud()
        not_ground_pcd.points = o3d.utility.Vector3dVector(not_ground_pcd_points)
        not_ground_pcd.paint_uniform_color(np.array([0.2,0.2,0.2]))
        not_ground_pcd.translate([0,0,-0.05])
        return not_ground_pcd

    def ground_pcd(self):
        ground_pcd_points = np.asarray(self.pcd.points)[self.traversable_points_idx]
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_pcd_points)
        ground_pcd.paint_uniform_color(np.array([0,1,0]))
        return ground_pcd

    def print_results(self, floors):
        self.print("="*5 + "RESULTS" + "="*5 )
        self.points_inside_cylinder = np.empty((1,3))
        for floor in floors:
            
            self.print("Name: " + floor["name"])
            total = floor["cell_grid"].shape[0] *  floor["cell_grid"].shape[1]

            def is_traversable(cell):
                if cell is None:
                    return False
                return cell.is_traversable
            def is_unknown(cell):
                if cell is None:
                    return True
                return False

            CELL_AREA =  CELL_SIZE * CELL_SIZE 
            traversable = len(np.where( np.vectorize(is_traversable)(floor["cell_grid"]))[0])
            unknown = len(np.where( np.vectorize(is_unknown)(floor["cell_grid"]))[0])
            self.print("Traversable area: " + str(traversable*CELL_AREA))
            self.print("Untraversable area: " + str((total - traversable - unknown)*CELL_AREA))
            self.print("Unknown area: " + str(unknown*CELL_AREA))
            self.print("Total area: " + str(total*CELL_AREA))

            for x_idx, y_idx in np.ndindex(floor["cell_grid"].shape):
                cell = floor["cell_grid"][x_idx, y_idx]
                if cell is None:
                    continue
                if not cell.is_traversable:
                    continue
                
                lowest_z = np.min(  np.asarray(self.pcd.points)[cell.points_idx_in_full_pcd][:,2])
                z = cell.elevation
                if z - lowest_z > 2*ROBOT_STEP_SIZE:
                    points_below_z = np.where(  np.asarray(self.pcd.points)[cell.points_idx_in_full_pcd][:,2] < lowest_z + ROBOT_STEP_SIZE  )
                else:
                    points_below_z = np.where(  np.asarray(self.pcd.points)[cell.points_idx_in_full_pcd][:,2] < z  )  

                points_below_z_idx = np.array(points_below_z[0], int)
                mean_z = np.mean(  np.asarray(self.pcd.points)[np.take(cell.points_idx_in_full_pcd, points_below_z_idx)][:,2])
                #self.print("mean_z " + str(mean_z))
                #self.print("elev " + str(cell.elevation))

                x = x_idx * CELL_SIZE + floor["min_x"] + CELL_SIZE/2
                y = y_idx * CELL_SIZE + floor["min_y"] + CELL_SIZE/2
                #z = floor["cell_grid"][x_idx, y_idx].elevation 
                z = mean_z
                center = [x,y,z]
                mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=CELL_SIZE/2,
                                                          height=CELL_SIZE)
                mesh_cylinder.compute_vertex_normals()
                mesh_cylinder.paint_uniform_color([0.3, 0.7, 0.3])
                mesh_cylinder.translate(center)
                R = o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, 0, 0])
                mesh_cylinder.rotate(R, center=center)
                #self.full_pcd.append(mesh_cylinder)
                #continue

                x_direction = False

                polygon = []
                if x_direction:
                    start = x + CELL_SIZE/2
                    end = x - CELL_SIZE/2
                else:
                    start = y + CELL_SIZE/2
                    end = y - CELL_SIZE/2

                for point in range(16):
                    angle = point/16*np.pi*2
                    if x_direction:
                        x = 0
                        y = center[1] + np.cos(angle) * CELL_SIZE * 0.501
                    else:
                        y = 0
                        x = center[0] + np.cos(angle) * CELL_SIZE * 0.501
                    
                    z = center[2] + np.sin(angle) * CELL_SIZE * 0.501

    def is_inside_ramp_area(self, point):
        point_good_format = o3d.utility.Vector3dVector([point])
        bounding_box = self.pcd.get_axis_aligned_bounding_box()
        min_bounds = np.min(np.asarray(bounding_box.get_box_points()), axis=0)
        max_bounds = np.max(np.asarray(bounding_box.get_box_points()), axis=0)
        self.ramp_bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bounds,
            [max_bounds[0] - 37, max_bounds[1] - 23, max_bounds[2]]
        )
        ramp_ground_points_idx = self.ramp_bounding_box.get_point_indices_within_bounding_box(point_good_format)

        return len(ramp_ground_points_idx) > 0


    def get_cell_grid(self, floor):
        try:
            return floor.cell_grid
        except:
            return floor["cell_grid"]

    def get_ground_points(self, cell):
        lowest_z = np.min(  np.asarray(self.pcd.points)[cell.points_idx_in_full_pcd][:,2])
        points_below_z = np.where(  np.asarray(self.pcd.points)[cell.points_idx_in_full_pcd][:,2] <= lowest_z + 2*ROBOT_STEP_SIZE  )[0]
        idx_in_points_idx = np.array(points_below_z, int)
        return np.take(cell.points_idx_in_full_pcd, idx_in_points_idx)

        z = cell.elevation
        if z - lowest_z > 2*ROBOT_STEP_SIZE:
            points_below_z = np.where(  np.asarray(self.pcd.points)[cell.points_idx_in_full_pcd][:,2] < lowest_z + ROBOT_STEP_SIZE  )
        else:
            points_below_z = np.where(  np.asarray(self.pcd.points)[cell.points_idx_in_full_pcd][:,2] < z  )  
            
        points_below_z_idx = np.array(points_below_z[0], int)
        return points_below_z_idx



    def get_covered_points(self, floors):
        ground_pcd = self.ground_pcd()
        total_ground_points = 0
        points_inside_cylinder = np.empty((1,3))
        draw_cylinders = []
        for floor in floors:

            def is_traversable(cell):
                if cell is None:
                    return False
                return cell.is_traversable

            for x_idx, y_idx in np.ndindex(self.get_cell_grid(floor).shape):
                cell = self.get_cell_grid(floor)[x_idx, y_idx]
                
                if not is_traversable(cell):
                    continue
                
                ground_points = self.get_ground_points(cell)
                mean_z = np.mean(  np.asarray(self.pcd.points)[ground_points][:,2])

                x = x_idx * CELL_SIZE + floor["min_x"] + CELL_SIZE/2
                y = y_idx * CELL_SIZE + floor["min_y"] + CELL_SIZE/2
                z = mean_z
                center = [x,y,z]
                #self.print(center)
                if not self.is_inside_ramp_area(center):
                    #self.print("nope")
                    continue

                total_ground_points += len(ground_points)

                x_direction = True

                polygon = []
                if x_direction:
                    start = x + CELL_SIZE/2
                    end = x - CELL_SIZE/2
                else:
                    start = y + CELL_SIZE/2
                    end = y - CELL_SIZE/2

                for point in range(16):
                    angle = point/16*np.pi*2
                    if x_direction:
                        x = 0
                        y = center[1] + np.cos(angle) * CELL_SIZE * 0.5 + 0.03
                    else:
                        y = 0
                        x = center[0] + np.cos(angle) * CELL_SIZE * 0.5 + 0.03
                    
                    z = center[2] + np.sin(angle) * CELL_SIZE * 0.5

                    polygon.append(
                            [x, y, z]
                        )
                #self.print(polygon)
                #self.print(polygon.reverse)
                polygon_bbox = o3d.visualization.SelectionPolygonVolume()
                if x_direction:
                    polygon_bbox.orthogonal_axis = "X"
                else:
                    polygon_bbox.orthogonal_axis = "Y"
                polygon_bbox.axis_max = float(start)
                polygon_bbox.axis_min = float(end)
                polygon_bbox.bounding_polygon = o3d.utility.Vector3dVector(np.asarray(polygon).astype("float64"))
                #self.print( polygon_bbox.axis_max)
                #self.print( polygon_bbox.axis_min)
                polygon_pcd = polygon_bbox.crop_point_cloud(ground_pcd)
                #self.print( len(polygon_pcd.points))
                points_inside_cylinder = np.append(points_inside_cylinder, polygon_pcd.points, axis=0)
                self.print(points_inside_cylinder.shape)

                if False:# VISUALIZE:
                    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=CELL_SIZE* 0.51, height=CELL_SIZE)
                    mesh_cylinder.compute_vertex_normals()
                    mesh_cylinder.paint_uniform_color([0.3, 0.3, 0.7])
                    mesh_cylinder.translate(center)
                    if x_direction:
                        R = o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, np.pi/2, 0])
                    else:
                        R = o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, 0, 0])
                    mesh_cylinder.rotate(R, center=center)
                    draw_cylinders.append(mesh_cylinder)

        return points_inside_cylinder, total_ground_points, draw_cylinders

            #mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=CELL_SIZE/2,
            #                                          height=CELL_SIZE)
            #mesh_cylinder.compute_vertex_normals()
            #mesh_cylinder.paint_uniform_color([0.3, 0.7, 0.3])
            #mesh_cylinder.translate(center)
            #R = o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, 0, 0])
            #mesh_cylinder.rotate(R, center=center)
            ##self.full_pcd.append(mesh_cylinder)
            #continue
































































































                    
            
                