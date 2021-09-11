import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
from exjobb.PointCloud import PointCloud
from exjobb.Parameters import ROBOT_RADIUS
from scipy.spatial import distance_matrix
BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
COVERABLE = 1
OBSTACLE = 0


class Map():
    def __init__(self, print, map_file, resolution):
        self.resolution = resolution
        self.print = print
        image = Image.open(map_file) 
        color_map = np.array(image)

        if color_map.ndim == 2:
            self.map = np.where(color_map>250, COVERABLE, OBSTACLE).astype(int)

        else:
            self.print(color_map.shape)
            color_map_reshape = color_map.reshape((color_map.shape[1]*color_map.shape[0]), color_map.shape[2])
            coverable = np.where(np.all(color_map_reshape[:,:] == WHITE,  axis=1))[0]
            self.map = np.zeros(color_map_reshape.shape[0]).astype(int)
            self.print(self.map.shape)
            self.map[~coverable] = OBSTACLE
            self.map[coverable] = COVERABLE
            self.print(len(coverable))
            self.print(self.map)
            self.map = self.map.reshape((color_map.shape[0], color_map.shape[1]))
            self.print(self.map.shape)

        rows, cols = np.where(self.map != -100)
        self.points = np.vstack([rows, cols, np.zeros(len(cols))]).T*self.resolution


    def map_to_traversable_and_coverable_pcd(self):
        traversable_points, coverable_points = self.get_traversable_and_coverable_points()
        return PointCloud(print, points=traversable_points), PointCloud(print, points=coverable_points)

    def map_to_full_pcd(self):
        return PointCloud(print, points=self.points)

    def get_points_that_are(self, value):
        rows, cols = np.where(self.map == value)
        return np.vstack([rows, cols, np.zeros(len(cols))]).T*self.resolution
    


    def get_traversable_and_coverable_points(self):
        desize = int(len(self.points) / 70000)
        obstacle_points = self.get_points_that_are(OBSTACLE)[::desize,:]
        coverable_points = self.get_points_that_are(COVERABLE)[::desize,:]

        self.print("obst" + str(obstacle_points.shape))
        self.print("coverable_points" + str(coverable_points.shape))

        ground_points_idx = np.where(self.map.flatten() == COVERABLE)[0]
        traversable_points_idx = ground_points_idx
        full_pcd = self.map_to_full_pcd()

        dist_matrix = distance_matrix(obstacle_points, coverable_points)

        rows, cols = np.where(dist_matrix < 10*self.resolution)

        border_points = obstacle_points[rows]


        for i, untraversable_point in enumerate(border_points):
            #if i % 100 == 0:
            #    self.self.print("Working on border pos " + str(i) + " out of " + str(len(uncoverable_border_points))) 
            collision_risk_points = full_pcd.points_idx_in_radius(untraversable_point, ROBOT_RADIUS)
            traversable_points_idx = self.delete_values(traversable_points_idx, collision_risk_points) 

        traversable_pcd = PointCloud(print, points= full_pcd.points[traversable_points_idx.astype(int)])

        coverable_points_idx_queue = ground_points_idx
        coverable_points_idx_queue = self.delete_values(coverable_points_idx_queue, traversable_points_idx)
        false_coverable_points_idx = np.array([])
        while len(coverable_points_idx_queue):
            if len(coverable_points_idx_queue) % 1000 == 0:
                self.print("coverable_points_idx_queue: " + str(len(coverable_points_idx_queue)))
            point_idx, coverable_points_idx_queue = coverable_points_idx_queue[0], coverable_points_idx_queue[1:]
            distance_to_nearest_traversable_point = traversable_pcd.distance_to_nearest(full_pcd.points[point_idx]) 
            if distance_to_nearest_traversable_point > ROBOT_RADIUS:
                false_coverable_points_idx = np.append(false_coverable_points_idx, point_idx)
        
        real_coverable_points_idx = self.delete_values(ground_points_idx, false_coverable_points_idx)

        #self.print((len(real_coverable_points_idx), len(traversable_points_idx), len(false_coverable_points_idx), len(full_pcd.points)))  

        return full_pcd.points[traversable_points_idx], full_pcd.points[real_coverable_points_idx]

    def delete_values(self, array, values):
        ''' Removes specific values from an array
        Args:
            array: NumPy array to remove values from
            values: NumPy array with values that should be removed.
        '''
        return array[ np.isin(array, values, assume_unique=True, invert=True) ]

def main():
    #resolution: m / pixel
    #map = Map("src/exjobb/maps/map_simple_open.png", 0.015)
    map = Map("src/exjobb/maps/map-ipa-apartment.png", 0.05) #0.015
    map.get_traversable_and_coverable_points()