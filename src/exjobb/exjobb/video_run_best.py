
from copy import deepcopy
from exjobb.BAstar import BAstar
from exjobb.Environments import PointCloudEnvironment
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

################################################

#Choose between garage, bridge or crossing
PCD = "bridge"   

#Choose between BA*, Inward Spiral or Sampled BA*
ALGORITHM = "Sampled BA*"   

#Settings:
TIME_LIMIT = 400
GOAL_COVERAGE = 0.97

####################################################

PCD_DATA = {
    "garage": {
        "pcd_file": "garage.pcd",
        "terr_assessment_file": 'garage_terrain_assessment.dictionary',
        "start_point": np.array([28.6, -6.7, -10.3])
    },
    "bridge": {
        "pcd_file": "bridge_2.pcd",
        "terr_assessment_file": 'bridge_terrain_assessment.dictionary',
        "start_point": np.array([-53.7, 54.2, -2.7])
    },
    "crossing": {
        "pcd_file": "cross.pcd",
        "terr_assessment_file": 'cross_terrain_assessment.dictionary',
        "start_point": np.array([-20.7, 43, -1])
    }
}

PARAMETER_DATA = [
    {
        "param" : {  'angle_offset': 4.64664343656672,
                        'step_size': 0.633652049936913,
                        'visited_threshold': 0.472819723019576}, 
        "cpp": "BA*",
        "pcd": "garage"
    },
    {
        "param" : {  'step_size': 0.627870706339337,
                        'visited_threshold': 0.498775709725593} , 
        "cpp": "Inward Spiral",
        "pcd": "garage"
    },
    {
        "param" : {'ba_exploration': 0.853031300592955,
                                      'max_distance': 3.89663024793223,
                                      'max_distance_part_II': 4.80685526433465,
                                      'min_bastar_cost_per_coverage': 9312.530314616084,
                                      'min_spiral_cost_per_coverage': 13196.969167186768,
                                      'step_size': 0.636195719728099,
                                      'visited_threshold': 0.337665370485907}, 
        "cpp": "Sampled BA*",
        "pcd": "garage"
    },
    {
        "param" : {'angle_offset': 5.338811570534332,
                       'step_size': 0.5569273719420402,
                       'visited_threshold': 0.45316918436457587}, 
        "cpp": "BA*",
        "pcd": "bridge"
    },
    {
        "param" : {'step_size': 0.7371140202635981,
                                 'visited_threshold': 0.48308887747347695}, 
        "cpp": "Inward Spiral",
        "pcd": "bridge"
    },
    {
        "param" : {'ba_exploration': 0.9399786469446915,
                                      'max_distance': 4.490537491471363,
                                      'max_distance_part_II': 7.059483126389999,
                                      'min_bastar_cost_per_coverage': 12772.91170649628,
                                      'min_spiral_cost_per_coverage': 25988.65403939978,
                                      'step_size': 0.6187054519800317,
                                      'visited_threshold': 0.3887247448006702}, 
        "cpp": "Sampled BA*",
        "pcd": "bridge"
    },
    {
        "param" : {'angle_offset': 4.701355889577928,
                       'step_size': 0.5236466714162833,
                       'visited_threshold': 0.4036817132888348}, 
        "cpp": "BA*",
        "pcd": "crossing"
    },
    {
        "param" : {'step_size': 0.6646718250765709,
                                 'visited_threshold': 0.49966903877360175}, 
        "cpp": "Inward Spiral",
        "pcd": "crossing"
    },
    {
        "param" : {'ba_exploration': 0.8631454551560509,
                                      'max_distance': 1.6928075586882567,
                                      'max_distance_part_II': 4.483751889847026,
                                      'min_bastar_cost_per_coverage': 6488.439310516093,
                                      'min_spiral_cost_per_coverage': 8141.022522321784,
                                      'step_size': 0.5539770484967692,
                                      'visited_threshold': 0.2576619746522969}, 
        "cpp": "Sampled BA*",
        "pcd": "crossing"
    },          
]

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
        environment = PointCloudEnvironment(self.print, PCD_DATA[PCD]["terr_assessment_file"], PCD_DATA[PCD]["pcd_file"], False)
        point_cloud  = environment.full_pcd
        traversable_point_cloud = environment.traversable_pcd
        coverable_point_cloud = environment.coverable_pcd
        motion_planner = MotionPlanner(self.print, traversable_point_cloud)
        

        parameters =  list(filter(lambda x: x["pcd"] == PCD and x["cpp"] == ALGORITHM, PARAMETER_DATA))[0]["param"]

        

        if ALGORITHM == "BA*":
            cpp = BAstar(self.print, motion_planner, coverable_point_cloud, time_limit=TIME_LIMIT, parameters = parameters)
        if ALGORITHM == "Inward Spiral":
            cpp = Spiral(self.print, motion_planner, coverable_point_cloud, time_limit=TIME_LIMIT, parameters = parameters)
        if ALGORITHM == "Sampled BA*":
            cpp = RandomBAstar3(self.print, motion_planner, coverable_point_cloud, time_limit=TIME_LIMIT, parameters = parameters)
            
        self.path = cpp.get_cpp_path(PCD_DATA[PCD]["start_point"], goal_coverage=GOAL_COVERAGE)
        self.print(cpp.print_stats(self.path))
        
        markers_pub = self.create_timer(5, self.marker_publisher)     

        self.pcd_pub.publish(point_cloud.point_cloud(point_cloud.points, 'my_frame'))
        self.coverable_pcd_pub.publish(coverable_point_cloud.point_cloud(coverable_point_cloud.points, 'my_frame'))
        self.traversable_pcd_pub.publish(traversable_point_cloud.point_cloud(traversable_point_cloud.points, 'my_frame'))
        
    def marker_publisher(self, max=None):
        self.markers_msg = visualization_msgs.MarkerArray()
        stamp = self.get_clock().now().to_msg()
        msg = ROSMessage.line_marker(0, stamp, self.path, [0.1,1.0,1.0], "path")
        self.markers_msg.markers = [msg]
        self.markers_pub.publish(self.markers_msg)

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