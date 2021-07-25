import numpy as np
import timeit
from exjobb.PointCloud import PointCloud
from exjobb.Parameters import ROBOT_SIZE, MARGIN



class RobotTraversability():
    ''' Class for calculating points where a robot could go without collision.
    '''

    def __init__(self, print, pcd):
        ''' 
        Args:
            print: function for printing messages
            pcd: Point Cloud of the environment.
        '''
        self.pcd = pcd     
        self.print = print        

    def get_traversable_points_idx(self, ground_points_idx):
        ''' Given some points, find those which are traversable by the robot,
        given robot size specifics. go through every point in the point cloud 
        that is classified as obstacle and filter out ground points that are 
        close.
        Args:
            ground_points_idx:  indexes of points in the point cloud that are 
                                classified as obstacle free.
        '''
        start_total = timeit.default_timer()
                
        all_points_idx = range(len(self.pcd.points))
        obstacle_points_idx = np.delete(all_points_idx, ground_points_idx)
        
        self.obstacle_point_cloud = PointCloud(self.print, points= self.pcd.points[obstacle_points_idx])
        self.ground_point_cloud = PointCloud(self.print, points= self.pcd.points[ground_points_idx])

        traversable_points = ground_points_idx
        obstacle_points_idx_queue = obstacle_points_idx
        
        while len(obstacle_points_idx_queue) > 0: 
            
            obstacle_point_idx, obstacle_points_idx_queue = obstacle_points_idx_queue[0], obstacle_points_idx_queue[1:]
            
            point = self.pcd.points[obstacle_point_idx]
            distance_to_nearest_ground_point = self.ground_point_cloud.distance_to_nearest(point)
            

            if distance_to_nearest_ground_point > MARGIN and distance_to_nearest_ground_point < ROBOT_SIZE:                
                collision_risk_points = self.pcd.points_idx_in_radius(point, 1.5*ROBOT_SIZE)
                traversable_points = self.delete_values(traversable_points, collision_risk_points)
                points_nearby = self.pcd.points_idx_in_radius(point, ROBOT_SIZE/2)
            elif distance_to_nearest_ground_point <= MARGIN:
                points_nearby = self.pcd.points_idx_in_radius(point, MARGIN)
            else:
                points_nearby = self.pcd.points_idx_in_radius(point, distance_to_nearest_ground_point-ROBOT_SIZE)

            obstacle_points_idx_queue = self.delete_values(obstacle_points_idx_queue, points_nearby)

        return self.pcd.points[traversable_points]


    def delete_values(self, array, values):
        ''' Removes specific values from an array
        Args:
            array: NumPy array to remove values from
            values: NumPy array with values that should be removed.
        '''
        return array[ np.isin(array, values, assume_unique=True, invert=True) ]