import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import visualization_msgs.msg as visualization_msgs
import geometry_msgs.msg as geometry_msgs

import open3d as o3d
import numpy as np

import os

#Code from https://github.com/SebastianGrans/ROS2-Point-Cloud-Demo/blob/master/pcd_demo/pcd_publisher/pcd_publisher_node.py

class PointCloud:

    def __init__(self, print, file = None, points = None ):
        self.print = print
        #self.filter_pointcloud2()

        #pcd_path = "smallpointcloud.pcd"
        if file is not None:
            self.print("Reading point cloud file...")
            self.raw_pcd = o3d.io.read_point_cloud(os.getcwd() + "/src/exjobb/exjobb/" + file)
            self.points = np.asarray(self.raw_pcd.points)

        elif points is not None:
            self.raw_pcd = o3d.geometry.PointCloud()
            self.raw_pcd.points = o3d.utility.Vector3dVector(points)
            self.points = points
        else:
            self.print("Missing pointcloud argument")
            return
        #self.points = np.unique(np.asarray(pcd.points), axis=0)

        
        #self.print(str(self.points.shape))
        #self.print(str(self.points))
        self.pcd = self.point_cloud(self.points, 'my_frame')
        self.kdtree = o3d.geometry.KDTreeFlann(self.raw_pcd)
        self.visited_points_idx = np.array([])
        self.traversable_points_idx = np.array([])
        #self.pcd_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd', 10)
        #timer_period = 5

        #self.print("Start publishing point cloud")
        #self.timer = self.create_timer(timer_period, self.marker_publisher)
        #self.pcd_pub = self.create_timer(7, self.point_cloud_publisher)
    
    def visit_point(self, point, robot_radius):
        [k, idx, _] = self.kdtree.search_radius_vector_3d(point, robot_radius)
        self.visited_points_idx = np.append(self.visited_points_idx, idx)
        
        #self.print("Coverage: " + str(self.get_coverage_efficiency()))

    def visit_path(self, path, robot_radius):
        for point in path:
            [k, idx, _] = self.kdtree.search_radius_vector_3d(point, robot_radius)
            self.visited_points_idx = np.append(self.visited_points_idx, idx)
    
    def detect_visited_points_from_path(self, visited_positions, robot_radius):
        self.print("Visiting " + str(len(visited_positions)) + " points...")
        for position in visited_positions:
            [k, idx, _] = self.kdtree.search_radius_vector_3d(position, robot_radius)
            self.visited_points_idx = np.append(self.visited_points_idx, idx)

    def get_visiting_rate_in_area(self, point, radius):
        [k, idx, _] = self.kdtree.search_radius_vector_3d(point, radius)
        total_points = len(idx)
        nbr_of_visited = len(np.intersect1d(self.visited_points_idx, np.asarray(idx)))
        return nbr_of_visited/total_points

    def get_pcd_from_visited_points(self):
        #self.print(str(self.visited_points_idx))
        self.print("Creating point cloud from " + str(len(self.points[self.visited_points_idx.astype(int), :])) + " visited points...")
        #visited_pcd = o3d.geometry.PointCloud()
        #visited_pcd.points = o3d.utility.Vector3dVector(self.points[self.visited_points_idx.astype(int), :])
        return self.point_cloud(self.points[self.visited_points_idx.astype(int), :], 'my_frame')

    def get_coverage_efficiency(self):
        #self.print("Counting Coverage efficiency...")
        self.visited_points_idx = np.unique(self.visited_points_idx, axis=0)
        return len(self.visited_points_idx) / len(self.points)

    def get_coverage_efficiency_increase_from_path(self, path, robot_radius):
        current_coverage_efficiency = self.get_coverage_efficiency()
        new_visited_points = np.empty((0,3))
        for point in path:
            [k, idx, _] = self.kdtree.search_radius_vector_3d(point, robot_radius)
            new_visited_points = np.append(new_visited_points, idx)
        new_coverage_efficiency = len(  np.unique(np.append(self.visited_points_idx, new_visited_points))) / len(self.points)
        
        return new_coverage_efficiency - current_coverage_efficiency

    def get_multiple_coverage(self, path, robot_radius):
        
        all_idx = np.array([], dtype=np.int)
        for point in path:
            [k, idx, _] = self.kdtree.search_radius_vector_3d(point, robot_radius)
            all_idx = np.append(all_idx, idx)

        nodes, inv, counts = np.unique(all_idx, return_inverse=True, return_counts=True)
        return np.mean(counts)
        #return len(self.visited_points_idx) / len(self.points)

    def point_cloud_publisher(self):
        #self.print("hej2")
        # For visualization purposes, I rotate the point cloud with self.R 
        # to make it spin. 
        #self.points = self.points @ self.R
        # Here I use the point_cloud() function to convert the numpy array 
        # into a sensor_msgs.PointCloud2 object. The second argument is the 
        # name of the frame the point cloud will be represented in. The default
        # (fixed) frame in RViz is called 'map'
        self.pcd = self.point_cloud(self.points, 'my_frame')
        #self.print("hej6")   
        # Then I publish the PointCloud2 object 
        #self.pcd_publisher.publish(self.pcd)
        #self.print('Publishing: "%s"' % self.pcd)

    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        Code source:
            https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
        References:
            http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
            http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
            http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html
        """
        # In a PointCloud2 message, the point cloud is stored as an byte 
        # array. In order to unpack it, we also include some parameters 
        # which desribes the size of each individual point.
        #self.print("hej3")    
        
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes() 

        #self.print("hej4")
        # The fields specify what the bytes represents. The first 4 bytes 
        # represents the x-coordinate, the next 4 the y-coordinate, etc.
        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]

        # The PointCloud2 message also has a header which specifies which 
        # coordinate frame it is represented in. 
        header = std_msgs.Header(frame_id=parent_frame)

        #self.print("hej5")

        return sensor_msgs.PointCloud2(
            header=header,
            height=1, 
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3), # Every point consists of three float32s.
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )

    def find_k_nearest(self, point, k):
        [k, idx, _] = self.kdtree.search_knn_vector_3d(point, k)
        return self.points[idx, :]

    def distance_to_nearest(self, point):
        nearest_point = self.find_k_nearest(point, 1)
        return np.linalg.norm(point[0:2] - nearest_point[0,0:2])

    def points_idx_in_radius(self, point, radius):
        [k, idx, _] = self.kdtree.search_radius_vector_3d(point, radius)
        return np.asarray(idx, dtype=int)
    
    def filter_pointcloud(self):
        x1 = 351463
        x2 = 351517
        y1 = 4022867
        y2 = 4022914
        z0 = 58

        pcd_path = "out.pcd"
    
        self.print("Reading point cloud file...")
        pcd = o3d.io.read_point_cloud(os.getcwd() + "/src/exjobb/exjobb/" + pcd_path, print_progress=True)
        self.print("Filtering point cloud...")

        origin = np.array([[x1,y1,z0]]) 
        self.points = np.array([[0,0,0]]) 

        all_points = np.asarray(pcd.points)
        
        c = np.zeros((1,2))
        c[0,0] = (x2-x1)/2 + x1
        c[0,1] = (y2-y1)/2 + y1
        temp = all_points[:,:2] - np.repeat(c, all_points.shape[0], axis=0)
        valid_idx = np.where(np.linalg.norm(temp, axis=1) < 30)
        self.points = all_points[valid_idx] - np.array([c[0,0], c[0,1], z0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        o3d.io.write_point_cloud(os.getcwd() + "/src/exjobb/exjobb/pointcloud.pcd", pcd)
    
    def filter_pointcloud2(self):
        x1 = 0
        x2 = 10
        y1 = 10
        y2 = 20
        z0 = 0

        pcd_path = os.getcwd() + "/src/exjobb/exjobb/pointcloud.pcd"
    
        self.print("Reading point cloud file...")
        pcd = o3d.io.read_point_cloud(pcd_path, print_progress=True)
        self.print("Filtering point cloud...")

        origin = np.array([[x1,y1,z0]]) 
        self.points = np.array([[0,0,0]]) 

        all_points = np.asarray(pcd.points)
        
        c = np.zeros((1,2))
        c[0,0] = (x2-x1)/2 + x1
        c[0,1] = (y2-y1)/2 + y1
        temp = all_points[:,:2] - np.repeat(c, all_points.shape[0], axis=0)
        valid_idx = np.where(np.linalg.norm(temp, axis=1) < 5)
        self.points = all_points[valid_idx] - np.array([c[0,0], c[0,1], z0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        o3d.io.write_point_cloud(os.getcwd() + "/src/exjobb/exjobb/smallpointcloud.pcd", pcd)

