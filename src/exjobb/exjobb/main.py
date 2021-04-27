import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import visualization_msgs.msg as visualization_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs

import numpy as np

import os

import pickle

from exjobb.PointCloud import PointCloud
from exjobb.TerrainAssessment import TerrainAssessment
from exjobb.TerrainAssessment2 import TerrainAssessment2
from exjobb.TerrainAssessment3 import TerrainAssessment3
from exjobb.MotionPlanner import MotionPlanner
from exjobb.NaiveRRTCPPDFS import NaiveRRTCPPDFS
from exjobb.NaiveRRTCPPAStar import NaiveRRTCPPAstar
from exjobb.ROSMessage import RED, GREEN, BLUE
import exjobb.ROSMessage as ROSMessage

DO_TERRAIN_ASSESSMENT = True
PUBLISH_FULL_PCD = True
PUBLISH_GROUND_PCD = True
PUBLISH_MARKERS = True
PUBLISH_PATH = False
PUBLISH_PATH_ANIMATION = False
PUBLISH_VISITED_PCD = False 
PUBLISH_VISITED_GROUND_PCD = False

MOTION_PLANNER_TEST = False
CPP_TEST = False

class MainNode(Node):

    def __init__(self):
        super().__init__('MainNode')



        #Publishers:
        self.markers_pub = self.create_publisher(visualization_msgs.MarkerArray, 'marker', 3000)
        self.pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'pcd', 10)
        self.ground_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'ground_pcd', 100)
        self.visited_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'visited_pcd', 100)
        self.visited_ground_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'visited_ground_pcd', 100)
        self.path_pub = self.create_publisher(nav_msgs.Path, 'path', 10)
        self.last_id = 0
        timer_period = 5
        animation_time_period = 0.5

        self.animation_iteration = 0
        

        #start_pos = [ 0.2,  2.7, -5.3]
        start_pos = [ -5.2,  -12.7, -10.3]
        end_pos = [ 5.2,  2.7, -5.3]
        #start_pos = [ 1.69000006, 19. ,        -5.26468706]
        #end_pos = [ 1.15999997, 20.29999924 , -5.28468704]

        self.point_cloud = PointCloud(self.get_logger(), file="pointcloud.pcd")
        
        ######### FOR NOW:
        #pcd_pub = self.create_timer(timer_period, self.point_cloud_publisher)
        #return

        motion_planner_pcd = self.point_cloud

        if DO_TERRAIN_ASSESSMENT:       
            terrain_assessment = TerrainAssessment3(self.get_logger(), self.point_cloud)
            terrain_assessment.analyse_terrain()

            traversable_points = self.point_cloud.points[np.unique(terrain_assessment.traversable_points_idx).astype(int)]

            with open('cached_traversable_points.dictionary', 'wb') as cached_pcd_file:
                cache_data = {"traversable_points": traversable_points}
                pickle.dump(cache_data, cached_pcd_file)

        else:
            with open('cached_traversable_points.dictionary', 'rb') as cached_pcd_file:
                cache_data = pickle.load(cached_pcd_file)
                traversable_points = cache_data["traversable_points"]


        self.ground_point_cloud = PointCloud(self.get_logger(), points= traversable_points)
        motion_planner_pcd = self.ground_point_cloud
        
        if PUBLISH_GROUND_PCD:
            ground_pcd_pub = self.create_timer(timer_period, self.ground_point_cloud_publisher)           
           

        start_point = self.point_cloud.find_k_nearest(start_pos, 1)[0]
        end_point = self.point_cloud.find_k_nearest(end_pos, 1)[0]
        
        self.markers = []
        self.markers.append( {"point": start_point, "color": RED} )
        self.markers.append( {"point": end_point, "color": BLUE} )

        if MOTION_PLANNER_TEST:
            self.motion_planner = MotionPlanner(self.get_logger(), motion_planner_pcd)
            self.path = self.motion_planner.RRT(start_point, end_point)
            if self.path is False:
                self.path = []


        if CPP_TEST:
            self.cpp = NaiveRRTCPPAstar(self.get_logger(), motion_planner_pcd)
            self.path = self.cpp.get_cpp_path(end_pos)            

        if PUBLISH_FULL_PCD:
            pcd_pub = self.create_timer(timer_period, self.point_cloud_publisher)

        if PUBLISH_MARKERS:
            markers_pub = self.create_timer(timer_period, self.marker_publisher)
        
        if PUBLISH_PATH:
            path_pub = self.create_timer(timer_period, self.path_publisher)

        

        if PUBLISH_VISITED_PCD:
            self.point_cloud.detect_visited_points_from_path(self.path, robot_radius = 1)
            self.visited_points_pcd = self.point_cloud.get_pcd_from_visited_points()
            visited_pcd_pub = self.create_timer(timer_period, self.visited_point_cloud_publisher)
        
        if PUBLISH_VISITED_GROUND_PCD:
            self.ground_point_cloud.detect_visited_points_from_path(self.path, robot_radius = 1)
            self.visited_ground_points_pcd = self.ground_point_cloud.get_pcd_from_visited_points()
            visited_ground_pcd_pub = self.create_timer(timer_period, self.visited_ground_point_cloud_publisher)

        if PUBLISH_PATH_ANIMATION:
            path_pub = self.create_timer(animation_time_period, self.animated_path_publisher)
    
    def point_cloud_publisher(self):
        self.pcd_pub.publish(self.point_cloud.pcd)
    
    def ground_point_cloud_publisher(self):
        self.ground_pcd_pub.publish(self.ground_point_cloud.pcd)
    
    def visited_point_cloud_publisher(self):
        self.visited_pcd_pub.publish(self.visited_points_pcd)

    def visited_ground_point_cloud_publisher(self):
        self.visited_ground_pcd_pub.publish(self.visited_ground_points_pcd)

    def path_publisher(self):
        path_msg = ROSMessage.path(self.path)
        self.path_pub.publish(path_msg)

    def animated_path_publisher(self):
        self.animation_iteration += 1

        if self.animation_iteration >= len(self.path):
            self.path_publisher()
            return

        path_msg = ROSMessage.path(self.path[0:self.animation_iteration])
        self.path_pub.publish(path_msg)

    def marker_publisher(self):
        self.markers_msg = visualization_msgs.MarkerArray()
        
        for marker in self.markers:
            stamp = self.get_clock().now().to_msg()
            point = marker["point"]
            color = marker["color"]
            msg = ROSMessage.point_marker(self.last_id, stamp, point, color)
            self.markers_msg.markers.append(msg)
            self.last_id += 1

        self.markers_pub.publish(self.markers_msg)
        

def main(args=None):
    rclpy.init(args=args)
    ros_node = MainNode()
    rclpy.spin(ros_node)
    ros_node.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()