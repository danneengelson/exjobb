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
        #ground_points_idx = ground_points_idx[0:100000]
        self.ground_points_idx =  ground_points_idx
        start_total = timeit.default_timer()
        is_traversable = 0
        self.loop = 0
        
        pcd_points_idx = range(len(self.pcd.points))
        not_ground_points_idx = np.delete(pcd_points_idx, ground_points_idx)
        
        self.not_ground_point_cloud = PointCloud(self.print, points= self.pcd.points[not_ground_points_idx])
        self.ground_point_cloud = PointCloud(self.print, points= self.pcd.points[ground_points_idx])

        traversable_points = np.array([], dtype=int)

        traversable_points = ground_points_idx
        not_ground_points_idx_queue = not_ground_points_idx
        
        while len(not_ground_points_idx_queue) > 0 and len(traversable_points) > 700000: 
            
            
            #if len(traversable_points) % 1000000 == 0:
            self.print("traversable_points: " + str(len(traversable_points)))
            self.print("not_ground_points_idx" + str(len(not_ground_points_idx_queue)))
            
            #is_traversable_start = timeit.default_timer()
            not_ground_point_idx, not_ground_points_idx_queue = not_ground_points_idx_queue[0], not_ground_points_idx_queue[1:]
            #self.print(ground_point_idx)
            
            point = self.pcd.points[not_ground_point_idx]
            distance_to_nearest = self.ground_point_cloud.distance_to_nearest(point)
            
            #self.print("distance_to_nearest" + str(distance_to_nearest))
            if distance_to_nearest > 0.25 and distance_to_nearest < 0.75:
                
                points_nearby = self.pcd.points_idx_in_radius(point, 0.75)
                #traversable_points = np.delete(traversable_points, np.where(traversable_points in points_nearby))
                traversable_points = self.delete_values(traversable_points, points_nearby)
                points_close_nearby = self.pcd.points_idx_in_radius(point, 0.25)
                #self.print(point)
                #self.print(len(not_ground_points_idx))
                #self.print(len(not_ground_points_idx_queue))
                #self.print(len(np.intersect1d(not_ground_points_idx, not_ground_points_idx_queue)))
                #left = np.delete(not_ground_points_idx, np.where(not_ground_points_idx in not_ground_points_idx_queue) )
                #self.print(len(left))
                #return self.pcd.points[self.delete_values(not_ground_points_idx, not_ground_points_idx_queue)]
            elif distance_to_nearest <= 0.25:
                points_close_nearby = self.pcd.points_idx_in_radius(point, distance_to_nearest)
            else:
                points_close_nearby = self.pcd.points_idx_in_radius(point, distance_to_nearest-0.5)

            not_ground_points_idx_queue = self.delete_values(not_ground_points_idx_queue, points_close_nearby)

            

        self.print("total: " + str(timeit.default_timer() - start_total))

        return self.pcd.points[traversable_points]


        ground_points_idx_queue = ground_points_idx


        

    def delete_values(self, array, values):
        return array[ np.isin(array, values, assume_unique=True, invert=True) ]

    def is_traversable_by_robot(self, point, ground_points_idx):

        distance_to_nearest = self.not_ground_point_cloud.distance_to_nearest(point)
        return distance_to_nearest > 0.1
        total_points = len(self.pcd.points_idx_in_radius(point, 1))
        not_ground_points = len(self.not_ground_point_cloud.points_idx_in_radius(point, 1))
        return  not_ground_points / total_points <= 0.725

        points_inside_robot_collision_box = self.pcd.points_idx_in_radius(point, 0.5)
        nbr_of_ground_points_inside_robot_collision_box = np.sum(np.in1d(points_inside_robot_collision_box, ground_points_idx, assume_unique=True))
        return nbr_of_ground_points_inside_robot_collision_box == len(points_inside_robot_collision_box)
        #self.print(len(points_inside_robot_collision_box))
        #self.print(hej)
        #loop_start = timeit.default_timer()
        #for point in points_inside_robot_collision_box:
        #    if point not in ground_points_idx:
        #        self.loop += timeit.default_timer() - loop_start
        #        return False
        #self.loop += timeit.default_timer() - loop_start
        #return True

    def get_info(self, point):
        self.print("distance_to_nearest" + str(self.ground_point_cloud.distance_to_nearest(point)))
        self.print("is_traversable_by_robot" + str(self.is_traversable_by_robot(point, self.ground_points_idx)))