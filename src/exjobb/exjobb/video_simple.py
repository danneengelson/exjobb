
from copy import deepcopy
from exjobb.BAstar import BAstar
from exjobb.Environments import MapEnvironment, PointCloudEnvironment
from exjobb.MotionPlanner import MotionPlanner
from exjobb.PointCloud import PointCloud
import exjobb.ROSMessage as ROSMessage
from exjobb.RandomBAstar3 import RandomBAstar3
from exjobb.Spiral import Spiral
import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import visualization_msgs.msg as visualization_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import numpy as np
import pickle
import open3d as o3d
import time


class MainNode(Node):

    def __init__(self):
        super().__init__('MainNode')

        #Publishers:
        self.markers_pub = self.create_publisher(visualization_msgs.MarkerArray, 'marker', 3000)
        self.markers_path_pub = self.create_publisher(visualization_msgs.Marker, 'path_markers', 3000)

        self.pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'pcd', 10)
        self.coverable_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'coverable_pcd', 100)
        self.visited_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'visited_pcd', 100)
        self.visited_ground_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'visited_ground_pcd', 100)
        self.traversable_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'traversable_pcd', 100)
        self.inaccessible_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'inaccessible_pcd', 100)
        self.path_pub = self.create_publisher(nav_msgs.Path, 'path', 10)
            

        #Create Environment
        environment = MapEnvironment(self.print, "simple_open.dictionary", "src/exjobb/maps/map_simple_open.png", 0.015)        
        point_cloud  = environment.full_pcd
        traversable_point_cloud = environment.traversable_pcd
        coverable_point_cloud = environment.coverable_pcd
        motion_planner = MotionPlanner(self.print, traversable_point_cloud)
        

        TIME_LIMIT = 400
        #start_point = np.array([5,0.55,0]) #spiral
        start_point = np.array([0.55,0.7,0])
        goal_coverage = 1
        paths_markers = []
        #Get CPP path
        current = "BA*"
        current_param = {  'angle_offset': 0,
                        'step_size': 0.5,
                        'visited_threshold': 0.375} 
        
        if current == "BA*":
            cpp = BAstar(self.print, motion_planner, coverable_point_cloud, time_limit=TIME_LIMIT, parameters = current_param)
        if current == "Inward Spiral":
            cpp = Spiral(self.print, motion_planner, coverable_point_cloud, time_limit=TIME_LIMIT, parameters = current_param)
            
        self.pcd_pub.publish(point_cloud.point_cloud(point_cloud.points, 'my_frame'))
        self.coverable_pcd_pub.publish(coverable_point_cloud.point_cloud(coverable_point_cloud.points, 'my_frame'))
        self.traversable_pcd_pub.publish(traversable_point_cloud.point_cloud(traversable_point_cloud.points, 'my_frame'))

        path = cpp.get_cpp_path(start_point, goal_coverage=goal_coverage)
        self.print(cpp.print_stats(path))
        current_max = 0
        coverable_point_cloud.covered_points_idx = np.array([])
        while current_max < len(path):
            self.markers_msg = visualization_msgs.MarkerArray()
            stamp = self.get_clock().now().to_msg()
            
            msg = ROSMessage.line_marker(0, stamp, path[0:current_max], [0.1,1.0,1.0], "path")
            self.markers_msg.markers = [msg]
            
            
            current_max += 2
            new_path = path[current_max-2:current_max]

            
            self.markers_pub.publish(self.markers_msg)

            #if current_max > 2:
            coverable_point_cloud.visit_path(new_path)
            visited_ground_points_pcd = coverable_point_cloud.get_covered_points_as_pcd()
            self.visited_ground_pcd_pub.publish(visited_ground_points_pcd)

    def print(self, object_to_print):
        self.get_logger().info(str(object_to_print))

def main(args=None):
    rclpy.init(args=args)
    ros_node = MainNode()
    rclpy.spin(ros_node)
    ros_node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()