
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

FILE = 'src/exjobb/garage_video.dictionary'
ALL_PATHS = [
    {
        "param" : {  'angle_offset': 3.44800051788481,
                        'step_size': 0.963400677899873,
                        'visited_threshold': 0.257015802906527}, 
        "cpp": "BA*",
        "ns": "bastar_1",
        "do_calc": False
    },
    {
        "param" : {  'angle_offset': 3.78341027362029,
                        'step_size': 0.601687134922371,
                        'visited_threshold': 0.328108983656107}, 
        "cpp": "BA*",
        "ns": "bastar_2",
        "do_calc": False
    },
    {
        "param" : {  'angle_offset': 5.27158130667689,
                        'step_size': 0.517468289229711,
                        'visited_threshold': 0.455659073558674}, 
        "cpp": "BA*",
        "ns": "bastar_3",
        "do_calc": False
    },
    {
        "param" : {  'angle_offset': 4.64664343656672,
                        'step_size': 0.633652049936913,
                        'visited_threshold': 0.472819723019576}, 
        "cpp": "BA*",
        "ns": "bastar_4",
        "do_calc": False
    },
    {
        "param" : {  'step_size': 0.999314930298507,
                        'visited_threshold': 0.32443603324225}, 
        "cpp": "Inward Spiral",
        "ns": "spiral_1",
        "do_calc": False
    },
    {
        "param" : {  'step_size': 0.825030992319859,
                        'visited_threshold': 0.433448258850281}, 
        "cpp": "Inward Spiral",
        "ns": "spiral_2",
        "do_calc": False
    },
    {
        "param" : {  'step_size': 0.521396930930628,
                        'visited_threshold': 0.47473068968531} , 
        "cpp": "Inward Spiral",
        "ns": "spiral_3",
        "do_calc": False
    },
    {
        "param" : {  'step_size': 0.627870706339337,
                        'visited_threshold': 0.498775709725593} , 
        "cpp": "Inward Spiral",
        "ns": "spiral_4",
        "do_calc": False
    },
    {
        "param" : {'ba_exploration': 0.90756041115558,
                                      'max_distance': 4.78202945337845,
                                      'max_distance_part_II': 6.75513650527977,
                                      'min_bastar_cost_per_coverage': 8192.530314616084,
                                      'min_spiral_cost_per_coverage': 12157.969167186768,
                                      'step_size': 0.562061544696692,
                                      'visited_threshold': 0.279490436505789}, 
        "cpp": "Sampled BA*",
        "ns": "sampled_1",
        "do_calc": False
    },
    {
        "param" : {'ba_exploration': 0.816319265003861,
                                      'max_distance': 1.02476727664307,
                                      'max_distance_part_II': 4.76356301411862,
                                      'min_bastar_cost_per_coverage': 6616.530314616084,
                                      'min_spiral_cost_per_coverage': 19277.969167186768,
                                      'step_size': 0.950568870175564,
                                      'visited_threshold': 0.484179597225153}, 
        "cpp": "Sampled BA*",
        "ns": "sampled_2",
        "do_calc": False
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
        "ns": "sampled_3",
        "do_calc": False
    },
    {
        "param" : {'ba_exploration': 0.8653615601139727,
                                      'max_distance': 4.129493635268686,
                                      'max_distance_part_II': 6.935911381739787,
                                      'min_bastar_cost_per_coverage': 8238.530314616084,
                                      'min_spiral_cost_per_coverage': 13644.969167186768,
                                      'step_size': 0.54868363557903,
                                      'visited_threshold': 0.3730115058138923}, 
        "cpp": "Sampled BA*",
        "ns": "sampled_4",
        "do_calc": False
    },
    
]

def save_data(data=None):
    with open(FILE, 'wb') as cached_pcd_file:
        if data is None:
            cache_data = deepcopy(ALL_PATHS)
        else:
            cache_data = deepcopy(data)

        pickle.dump(cache_data, cached_pcd_file) 

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
        
        with open(FILE, 'rb') as cached_pcd_file:
            cache_data = pickle.load(cached_pcd_file)
            self.prev_file = deepcopy(cache_data)
            

        #Create Environment
        environment = PointCloudEnvironment(self.print, "garage_terrain_assessment.dictionary", "garage.pcd", False)
        point_cloud  = environment.full_pcd
        traversable_point_cloud = environment.traversable_pcd
        coverable_point_cloud = environment.coverable_pcd
        motion_planner = MotionPlanner(self.print, traversable_point_cloud)
        

        TIME_LIMIT = 400
        start_point = np.array([28.6, -6.7, -10.3])
        goal_coverage = 0.97
        paths_markers = []
        #Get CPP path
        for pub_path in ALL_PATHS:
            
            if not pub_path["do_calc"]:
                continue 

            coverable_pcd = PointCloud(self.print, points=coverable_point_cloud.points)
            if pub_path["cpp"] == "BA*":
                cpp = BAstar(self.print, motion_planner, coverable_pcd, time_limit=TIME_LIMIT, parameters = pub_path["param"])
            if pub_path["cpp"] == "Inward Spiral":
                cpp = Spiral(self.print, motion_planner, coverable_pcd, time_limit=TIME_LIMIT, parameters = pub_path["param"])
            if pub_path["cpp"] == "Sampled BA*":
                cpp = RandomBAstar3(self.print, motion_planner, coverable_pcd, time_limit=TIME_LIMIT, parameters = pub_path["param"])
            
            ns = pub_path["ns"]
            pub_path["path"] = cpp.get_cpp_path(start_point, goal_coverage=goal_coverage)
            pub_path["markers"] = cpp.points_to_mark
            pub_path["stats"] = cpp.print_stats(pub_path["path"])
            save_data(ALL_PATHS)
            #self.print(path_msg)
            #paths_markers.append(path_msg)
            #path_pub = self.create_publisher(nav_msgs.Path, ns + '/path', 10)
            #self.markers_pub.publish(path_msg)
        #
        #self.markers_msg = visualization_msgs.MarkerArray()
        #
        #for idx, msg in enumerate(paths_markers):
        #    stamp = self.get_clock().now().to_msg()
        #    self.markers_msg.markers.append(msg)
        #    #self.last_id += 1
        #
        
        markers_pub = self.create_timer(5, self.marker_publisher)



        #self.markers_pub.publish(self.markers_msg)

        

        for idx, pub_path in enumerate(ALL_PATHS):
            if pub_path.get("stats", False) is False:
                pub_path["stats"] = self.prev_file[idx]["stats"]
            self.print("="*20) 
            self.print(pub_path["ns"]) 
            self.print(pub_path["stats"])

       




        self.pcd_pub.publish(point_cloud.point_cloud(point_cloud.points, 'my_frame'))
        self.coverable_pcd_pub.publish(coverable_point_cloud.point_cloud(coverable_point_cloud.points, 'my_frame'))
        self.traversable_pcd_pub.publish(traversable_point_cloud.point_cloud(traversable_point_cloud.points, 'my_frame'))
        
    def marker_publisher(self, max=None):
        self.markers_msg = visualization_msgs.MarkerArray()
        self.last_id = 0
        for idx, path in enumerate(ALL_PATHS):
            
            stamp = self.get_clock().now().to_msg()

            if path.get("path", False) is False:
                path["path"] = self.prev_file[idx]["path"]

            msg = ROSMessage.line_marker(self.last_id, stamp, path["path"], [0.1,1.0,1.0], path["ns"])
            self.markers_msg.markers.append(msg)
            self.last_id += 1

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