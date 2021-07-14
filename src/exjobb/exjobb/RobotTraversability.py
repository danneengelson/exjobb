import numpy as np
import open3d as o3d
import timeit
from exjobb.PointCloud import PointCloud
import collections
from exjobb.Parameters import CELL_SIZE, ROBOT_STEP_SIZE, ROBOT_SIZE
import pickle
import os

class RobotTraversability():

    def __init__(self, print, pcd):
        self.pcd = pcd     

        self.print = print        

    def get_traversable_points_for_robot_idx(self, ground_points_idx):
        ##only for rviz
        self.ground_points_idx =  ground_points_idx
        ##

        start_total = timeit.default_timer()
        is_traversable = 0
        self.loop = 0
        
        pcd_points_idx = range(len(self.pcd.points))
        not_ground_points_idx = np.delete(pcd_points_idx, ground_points_idx)
        
        self.not_ground_point_cloud = PointCloud(self.print, points= self.pcd.points[not_ground_points_idx])
        self.ground_point_cloud = PointCloud(self.print, points= self.pcd.points[ground_points_idx])

        traversable_points = ground_points_idx
        not_ground_points_idx_queue = not_ground_points_idx
        
        while len(not_ground_points_idx_queue) > 0: 
            self.print(len(not_ground_points_idx_queue))
            
            not_ground_point_idx, not_ground_points_idx_queue = not_ground_points_idx_queue[0], not_ground_points_idx_queue[1:]
            
            point = self.pcd.points[not_ground_point_idx]
            distance_to_nearest = self.ground_point_cloud.distance_to_nearest(point)
            

            if distance_to_nearest > 0.25 and distance_to_nearest < ROBOT_SIZE:                
                points_nearby = self.pcd.points_idx_in_radius(point, 1.5*ROBOT_SIZE)
                traversable_points = self.delete_values(traversable_points, points_nearby)
                points_close_nearby = self.pcd.points_idx_in_radius(point, ROBOT_SIZE/2)
            elif distance_to_nearest <= 0.25:
                points_close_nearby = self.pcd.points_idx_in_radius(point, 0.25)
            else:
                points_close_nearby = self.pcd.points_idx_in_radius(point, distance_to_nearest-ROBOT_SIZE)

            not_ground_points_idx_queue = self.delete_values(not_ground_points_idx_queue, points_close_nearby)

            

        self.print("total: " + str(timeit.default_timer() - start_total))

        return self.pcd.points[traversable_points]

        

    def delete_values(self, array, values):
        return array[ np.isin(array, values, assume_unique=True, invert=True) ]


    def get_info(self, point):
        self.print("distance_to_nearest" + str(self.ground_point_cloud.distance_to_nearest(point)))
