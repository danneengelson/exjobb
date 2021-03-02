import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import visualization_msgs.msg as visualization_msgs
import geometry_msgs.msg as geometry_msgs

import numpy as np

import os

from exjobb.PointCloud import PointCloud
from exjobb.TerrainAssessment import TerrainAssessment
from exjobb.TerrainAssessment2 import TerrainAssessment2
from exjobb.TerrainAssessment3 import TerrainAssessment3
from exjobb.ROSMessage import RED, GREEN, BLUE
import exjobb.ROSMessage as ROSMessage


class MainNode(Node):

    def __init__(self):
        super().__init__('MainNode')

        #Publishers:
        self.markers_publisher = self.create_publisher(visualization_msgs.MarkerArray, 'marker', 3000)
        self.pcd_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd', 10)
        timer_period = 5

        start_pos = [ 0.2,  2.7, -5.3]

        self.point_cloud = PointCloud(self.get_logger(), "pointcloud.pcd")

        terrain_assessment = TerrainAssessment3(self.get_logger(), self.point_cloud)
        
        self.traversable_positions = terrain_assessment.analyse_terrain(start_pos)
        
        self.get_logger().info("nbr of poses: " + str(len(self.traversable_positions)))
        #Timers:
        self.get_logger().info("Start publishing point cloud")
        #self.timer = self.create_timer(timer_period, self.marker_publisher)
        #pcd_pub = self.create_timer(timer_period, self.point_cloud_publisher)
        self.marker_publisher()
        self.point_cloud_publisher()
    
    def point_cloud_publisher(self):
        self.pcd_publisher.publish(self.point_cloud.pcd)

    def marker_publisher(self):
        self.markers_msg = visualization_msgs.MarkerArray()
        self.last_id = 0
        for count, pose in enumerate(self.traversable_positions):
            #self.get_logger().info("New point: " + str(pose.center_point))
            #self.get_logger().info("In array now: " + str(len(self.markers_msg.markers)))
        # Draw k_nearest_points
            if pose.traversable:
                color = BLUE
            else:
                color = RED

            for id, point in enumerate(pose.k_nearest_points):
                
                #self.get_logger().info(str(point))
                stamp = self.get_clock().now().to_msg()
                msg = ROSMessage.point_marker(self.last_id, stamp, point, color)
                self.markers_msg.markers.append(msg)
                self.last_id += 1
            
            #Draw normal
            stamp = self.get_clock().now().to_msg()
            normal_arrow = ROSMessage.arrow(self.last_id, stamp, pose.center_point, pose.normal, color)
            self.markers_msg.markers.append(normal_arrow)
            self.last_id += 1
        #self.get_logger().info("In TOTAL: " + str(len(self.markers_msg.markers)))
            self.get_logger().info("Publishing pose nr: " + str(count) + " out of " + str(len(self.traversable_positions)))
            #self.markers_publisher.publish(self.markers_msg)
        

def main(args=None):
    rclpy.init(args=args)
    ros_node = MainNode()
    rclpy.spin(ros_node)
    ros_node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()