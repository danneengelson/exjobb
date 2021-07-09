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
from exjobb.MotionPlanner import MotionPlanner
from exjobb.NaiveRRTCPPDFS import NaiveRRTCPPDFS
from exjobb.NaiveRRTCPPAStar import NaiveRRTCPPAstar
from exjobb.BoustrophedonCPP import BoustrophedonCPP
from exjobb.BAstar import BAstar
from exjobb.RobotTraversability import RobotTraversability
from exjobb.Spiral import Spiral
from exjobb.RandomSample import RandomSample
from exjobb.BAstarRRT import BAstarRRT
from exjobb.RandomBAstar import RandomBAstar

from exjobb.Parameters import ROBOT_SIZE, ROBOT_STEP_SIZE
from exjobb.ROSMessage import RED, GREEN, BLUE
import exjobb.ROSMessage as ROSMessage

DO_TERRAIN_ASSESSMENT = False
DO_ROBOT_TRAVERSABILITY = False
PUBLISH_FULL_PCD = True
PUBLISH_GROUND_PCD = True
PUBLISH_MARKERS = True
PUBLISH_PATH = False
PUBLISH_PATH_ANIMATION = True
PUBLISH_VISITED_PCD = False 
PUBLISH_VISITED_GROUND_PCD = True
PUBLISH_TRAVERSABLE_PCD = True

MOTION_PLANNER_TEST = True
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
        self.traversable_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'traversable_pcd', 100)
        self.path_pub = self.create_publisher(nav_msgs.Path, 'path', 10)
        self.last_id = 0
        timer_period = 5
        animation_time_period = 0.1
        self.animation_iteration = 0
        
        #Subscribers:
        self.rviz_sub = self.create_subscription(geometry_msgs.PointStamped, "clicked_point", self.clicked_point_cb, 100)

        start_pos = [ -5.2,  -12.7, -10.3]
        end_pos = [ -3.2,  1.7, -5.3]



        self.point_cloud = PointCloud(self.print, file="pointcloud.pcd")
        
        ######### FOR NOW:
        #pcd_pub = self.create_timer(timer_period, self.point_cloud_publisher)
        #return

        traversable_points, ground_points_idx = self.do_terrain_assessment()        
        self.ground_point_cloud = PointCloud(self.print, points= traversable_points)

        traversable_points_for_robot = self.do_robot_traversability(ground_points_idx)
        self.traversable_point_cloud = PointCloud(self.print, points= traversable_points_for_robot)
        
        traversable_pcd = self.traversable_point_cloud
        motion_planner = MotionPlanner(self.get_logger(), traversable_pcd)
        
        if PUBLISH_GROUND_PCD:
            ground_pcd_pub = self.create_timer(timer_period, self.ground_point_cloud_publisher)           
        
        if PUBLISH_TRAVERSABLE_PCD:
            traversable_pcd_pub = self.create_timer(timer_period, self.traversable_point_cloud_publisher)    

        start_point = self.point_cloud.find_k_nearest(start_pos, 1)[0]
        end_point = self.point_cloud.find_k_nearest(end_pos, 1)[0]
        
        self.markers = []
        self.markers.append( {"point": start_point, "color": RED} )
        self.markers.append( {"point": end_point, "color": BLUE} )

        #self.print(len(traversable_points_for_robot) / len(traversable_points))

        if MOTION_PLANNER_TEST:
            self.path = motion_planner.Astar(start_point, end_point)
            self.points_to_mark = np.array([start_point, end_point])
            if self.path is False:
                self.path = []

        if CPP_TEST:
            self.cpp = RandomBAstar(self.get_logger(), motion_planner)
            self.path = self.cpp.get_cpp_path(end_pos)            
            self.points_to_mark = self.cpp.get_points_to_mark()

        if PUBLISH_FULL_PCD:
            pcd_pub = self.create_timer(timer_period, self.point_cloud_publisher)

        if PUBLISH_MARKERS:
            markers_pub = self.create_timer(timer_period, self.marker_publisher)
        
        if PUBLISH_PATH:
            path_pub = self.create_timer(timer_period, self.path_publisher)

        if PUBLISH_VISITED_PCD:
            self.point_cloud.detect_visited_points_from_path(self.path, robot_radius = ROBOT_SIZE/2)
            self.visited_points_pcd = self.point_cloud.get_pcd_from_visited_points()
            visited_pcd_pub = self.create_timer(timer_period, self.visited_point_cloud_publisher)
        
        if PUBLISH_VISITED_GROUND_PCD:
            self.ground_point_cloud.detect_visited_points_from_path(self.path, robot_radius = ROBOT_SIZE/2)
            self.visited_ground_points_pcd = self.ground_point_cloud.get_pcd_from_visited_points()
            visited_ground_pcd_pub = self.create_timer(timer_period, self.visited_ground_point_cloud_publisher)

        if PUBLISH_PATH_ANIMATION:
            self.ground_point_cloud.visited_points_idx = np.array([])
            path_pub = self.create_timer(animation_time_period, self.animated_path_publisher)


    def do_robot_traversability(self, ground_points_idx):
        if DO_ROBOT_TRAVERSABILITY:
            self.robot_traversability = RobotTraversability(self.print, self.point_cloud)
            traversable_points_for_robot = self.robot_traversability.get_traversable_points_for_robot_idx(ground_points_idx)
            with open('cached_traversable_points_for_robot.dictionary', 'wb') as cached_pcd_file:
                cache_data = {"traversable_points_for_robot": traversable_points_for_robot}
                pickle.dump(cache_data, cached_pcd_file)
        else:
            with open('cached_traversable_points_for_robot.dictionary', 'rb') as cached_pcd_file:
                cache_data = pickle.load(cached_pcd_file)
                traversable_points_for_robot = cache_data["traversable_points_for_robot"]

        return traversable_points_for_robot

    
    def do_terrain_assessment(self):
        if DO_TERRAIN_ASSESSMENT:       
            terrain_assessment = TerrainAssessment(self.get_logger(), self.point_cloud, self.print)
            terrain_assessment.analyse_terrain()

            ground_points_idx = np.unique(terrain_assessment.traversable_points_idx).astype(int)
            traversable_points = self.point_cloud.points[ground_points_idx]

            with open('cached_traversable_points.dictionary', 'wb') as cached_pcd_file:
                cache_data = {"traversable_points": traversable_points, "ground_points_idx": ground_points_idx}
                pickle.dump(cache_data, cached_pcd_file)

        else:
            with open('cached_traversable_points.dictionary', 'rb') as cached_pcd_file:
                cache_data = pickle.load(cached_pcd_file)
                traversable_points = cache_data["traversable_points"]
                ground_points_idx = cache_data["ground_points_idx"]
        
        return traversable_points, ground_points_idx

    
    def point_cloud_publisher(self):
        self.pcd_pub.publish(self.point_cloud.pcd)
    
    def ground_point_cloud_publisher(self):
        self.ground_pcd_pub.publish(self.ground_point_cloud.pcd)
    
    def traversable_point_cloud_publisher(self):
        self.traversable_pcd_pub.publish(self.traversable_point_cloud.pcd)
    
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

        point = self.path[self.animation_iteration]
        self.ground_point_cloud.visit_point(point, self.path[self.animation_iteration-1], ROBOT_SIZE/2)
        self.visited_ground_points_pcd = self.ground_point_cloud.get_pcd_from_visited_points()
        self.visited_ground_point_cloud_publisher()

    def marker_publisher(self):
        self.markers_msg = visualization_msgs.MarkerArray()
        
        for idx, point in enumerate(self.points_to_mark):
            stamp = self.get_clock().now().to_msg()
            if idx % 2:
                color = [0.0, 0.0, 0.8]
            else:
                color = [0.0, 1.0, 0.0]
            msg = ROSMessage.point_marker(self.last_id, stamp, point, color)
            self.markers_msg.markers.append(msg)
            self.last_id += 1

        self.markers_pub.publish(self.markers_msg)

    def clicked_point_cb(self, msg):
        self.print(msg)
        point = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.print(self.traversable_point_cloud.distance_to_nearest(point))
        
        #self.robot_traversability.get_info(point)

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