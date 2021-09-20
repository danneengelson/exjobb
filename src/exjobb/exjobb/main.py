from exjobb.Environments import PointCloudEnvironment, MapEnvironment

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

from exjobb.PointCloud import PointCloud
from exjobb.TerrainAssessment import TerrainAssessment
from exjobb.MotionPlanner import MotionPlanner
from exjobb.TA_PointClassification import PointClassification
from exjobb.BAstar import BAstar
from exjobb.BAstarVariant import BAstarVariant
from exjobb.Spiral import Spiral
from exjobb.RandomBAstar import RandomBAstar
from exjobb.RandomBAstar2 import RandomBAstar2
from exjobb.RandomBAstar3 import RandomBAstar3
from exjobb.Map import Map
from exjobb.BFS import BFS

from exjobb.ROSMessage import RED, GREEN, BLUE
import exjobb.ROSMessage as ROSMessage

DO_TERRAIN_ASSESSMENT = False
PUBLISH_FULL_PCD = True
PUBLISH_GROUND_PCD = True
PUBLISH_MARKERS = False
PUBLISH_PATH = False
PUBLISH_PATH_ANIMATION = True
PUBLISH_SEGMENTS_ANIMATION = False
PUBLISH_VISITED_PCD = False 
PUBLISH_VISITED_GROUND_PCD = False
PUBLISH_TRAVERSABLE_PCD = True
PUBLISH_INACCESSIBLE_PCD = False

MOTION_PLANNER_TEST = False
CPP_TEST = True
COLLECT_RESULT = False
SMALL_POINT_CLOUD = False
PCD_FROM_MAP = False

class MainNode(Node):

    def __init__(self):
        super().__init__('MainNode')

        #Publishers:
        self.markers_pub = self.create_publisher(visualization_msgs.MarkerArray, 'marker', 3000)
        self.pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'pcd', 10)
        self.coverable_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'coverable_pcd', 100)
        self.visited_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'visited_pcd', 100)
        self.visited_ground_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'visited_ground_pcd', 100)
        self.traversable_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'traversable_pcd', 100)
        self.inaccessible_pcd_pub = self.create_publisher(sensor_msgs.PointCloud2, 'inaccessible_pcd', 100)
        self.path_pub = self.create_publisher(nav_msgs.Path, 'path', 10)
        
        #Varaiables for publishers
        self.last_id = 0
        timer_period = 5
        animation_time_period = 0.01
        self.animation_iteration = 0
        self.path = []
        
        #Subscribers:
        self.rviz_sub = self.create_subscription(geometry_msgs.PointStamped, "clicked_point", self.clicked_point_cb, 100)


        #environment = PointCloudEnvironment(self.print, "cached_coverable_points.dictionary", "pointcloud.pcd")
        #environment = MapEnvironment(self.print, "simple_open.dictionary", "src/exjobb/maps/map_simple_open.png", 0.015)
        #environment = MapEnvironment(self.print, "map_ipa_apartment.dictionary", "src/exjobb/maps/map_ipa_apartment.png", 0.05)




        NEW_POINTCLOUD = False
        if NEW_POINTCLOUD:
            
            environment = MapEnvironment(self.print, "simple_open.dictionary", "src/exjobb/maps/map_simple_open.png", 0.015)
            self.point_cloud = PointCloud(self.print, file="cross.pcd", new_point_cloud=True) 
        else:
            #environment = PointCloudEnvironment(self.print, "pointcloud2.dictionary", "pointcloud2.pcd", False) #x,y = #351451.84, 4022966.5   Street
            #environment = PointCloudEnvironment(self.print, "pointcloud3.dictionary", "pointcloud3.pcd", False) #x,y = (351609.25, 4022912.0)  Same Underground garage one floor
            #environment = PointCloudEnvironment(self.print, "pointcloud4.dictionary", "pointcloud4.pcd", False) #x,y = (326815.75, 4152473.25)  Busy street, with cars
            #environment = PointCloudEnvironment(self.print, "cached_coverable_points.dictionary", "pointcloud.pcd")
            #environment = MapEnvironment(self.print, "map_ipa_apartment.dictionary", "src/exjobb/maps/map_ipa_apartment.png", 0.05)
            #environment = PointCloudEnvironment(self.print, "bridge_2_small.dictionary", "bridge_2_small.pcd", False) 
            #environment = PointCloudEnvironment(self.print, "cross_terrain_assessment.dictionary", "cross.pcd", False) 
            #environment = PointCloudEnvironment(self.print, "pre_garage_terrain_assessment.dictionary", "garage.pcd", False)'
            environment = PointCloudEnvironment(self.print, "garage_terrain_assessment.dictionary", "garage.pcd", False)
            #environment = PointCloudEnvironment(self.print, "bridge_terrain_assessment.dictionary", "bridge_2.pcd", False)
            #environment = MapEnvironment(self.print, "simple_open.dictionary", "src/exjobb/maps/map_simple_open.png", 0.015)

            self.point_cloud  = environment.full_pcd

        #traversable_points, coverable_points, inaccessible_points = self.do_terrain_assessment()        
        

        #self.traversable_point_cloud = PointCloud(self.print, points= traversable_points)
        #self.coverable_point_cloud = PointCloud(self.print, points= coverable_points)
        #self.inaccessible_point_cloud = PointCloud(self.print, points= inaccessible_points)

        #
        self.point_cloud.pcd = self.point_cloud.point_cloud(self.point_cloud.points, 'my_frame')

        self.traversable_point_cloud = environment.traversable_pcd
        self.traversable_point_cloud.pcd = self.traversable_point_cloud.point_cloud(self.traversable_point_cloud.points, 'my_frame')
        
        self.coverable_point_cloud = environment.coverable_pcd
        self.coverable_point_cloud.pcd = self.coverable_point_cloud.point_cloud(self.coverable_point_cloud.points, 'my_frame')
        
        self.inaccessible_point_cloud = environment.inaccessible_pcd
        self.inaccessible_point_cloud.pcd = self.inaccessible_point_cloud.point_cloud(self.inaccessible_point_cloud.points, 'my_frame')


        if SMALL_POINT_CLOUD:
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                [10, 15, -5.3],
                [15, 21, 10]
            )
            trav_points_idx = bbox.get_point_indices_within_bounding_box(self.traversable_point_cloud.raw_pcd.points)
            self.traversable_point_cloud = PointCloud(self.print, points= self.traversable_point_cloud.points[trav_points_idx])
            cov_points_idx = bbox.get_point_indices_within_bounding_box(self.coverable_point_cloud.raw_pcd.points)
            self.coverable_point_cloud = PointCloud(self.print, points= self.coverable_point_cloud.points[cov_points_idx])

        self.markers = []
        
        motion_planner = MotionPlanner(self.print, self.traversable_point_cloud)
        
        
        
        if PUBLISH_INACCESSIBLE_PCD:
            inaccessible_pcd_pub = self.create_timer(timer_period, self.inaccessible_point_cloud_publisher)                

        if MOTION_PLANNER_TEST:
            #start_pos = [5.0625    ,  91.05000305, -32.58319855]
            #end_pos = [  0.8125    ,  93.30000305, -32.33319855]
            end_pos = np.array([  6.05999994, -13.  ,        -5.71468687])
            start_pos = np.array([28.6, -6.7, -10.3])
            start_point = self.traversable_point_cloud.find_k_nearest(start_pos, 1)[0]
            end_point = self.traversable_point_cloud.find_k_nearest(end_pos, 1)[0]
            self.path = motion_planner.Astar(start_point, end_point)
            
            self.markers.append( {"point": start_point, "color": RED} )
            self.markers.append( {"point": end_point, "color": BLUE} )
            if self.path is False:
                self.path = []

        if CPP_TEST:
            random_idx = np.random.choice(len(self.traversable_point_cloud.points), 1, replace=False)[0]
            #start_point = [-14, -16, -3.6] #
            #start_point = [-23, 30, -0.9]
            start_point = self.traversable_point_cloud.points[random_idx]


            #SAMPLED BA*            
            #cost 4953, length: 3684, rotation: 1269
            ######real: cost: ?? lkength: 3235, rotation: 1108
            sparam1 = {'ba_exploration': 0.90756041115558,
                                      'max_distance': 4.78202945337845,
                                      'max_distance_part_II': 6.75513650527977,
                                      'min_bastar_cost_per_coverage': 8192.530314616084,
                                      'min_spiral_cost_per_coverage': 12157.969167186768,
                                      'step_size': 0.562061544696692,
                                      'visited_threshold': 0.279490436505789}

            #cost 4615, length: 3294, rotation: 1321
            ######real: cost: ??, length: 3334, rotation: 1304
            sparam2 = {'ba_exploration': 0.816319265003861,
                                      'max_distance': 1.02476727664307,
                                      'max_distance_part_II': 4.76356301411862,
                                      'min_bastar_cost_per_coverage': 6616.530314616084,
                                      'min_spiral_cost_per_coverage': 19277.969167186768,
                                      'step_size': 0.950568870175564,
                                      'visited_threshold': 0.484179597225153}

            #cost 4261, length: 3158, rotation: 1103
            #######real: cost: ??, length: 3078, rotation: 1114
            sparam3 = {'ba_exploration': 0.853031300592955,
                                      'max_distance': 3.89663024793223,
                                      'max_distance_part_II': 4.80685526433465,
                                      'min_bastar_cost_per_coverage': 9312.530314616084,
                                      'min_spiral_cost_per_coverage': 13196.969167186768,
                                      'step_size': 0.636195719728099,
                                      'visited_threshold': 0.337665370485907}
            
            #length: 3596, rotation: 1296, 97% - annan step size (0.6..)
            #real: cost: 4306, length: 3073, rotation: 1233
            param4 = {'ba_exploration': 0.8653615601139727,
                                      'max_distance': 4.129493635268686,
                                      'max_distance_part_II': 6.935911381739787,
                                      'min_bastar_cost_per_coverage': 8238.530314616084,
                                      'min_spiral_cost_per_coverage': 13644.969167186768,
                                      'step_size': 0.54868363557903,
                                      'visited_threshold': 0.3730115058138923}


            #cost: 5797, length: 4643, rotation: 1154, 97% - annan step size (0.6..)
            #real: cost: 6422, length: 5116, rotation: 1306
            param_best_brdige = {'ba_exploration': 0.939978646944692,
                                      'max_distance': 4.49053749147136,
                                      'max_distance_part_II': 7.05948312639,
                                      'min_bastar_cost_per_coverage': 12772.530314616084,
                                      'min_spiral_cost_per_coverage': 25988.969167186768,
                                      'step_size': 0.618705451980032,
                                      'visited_threshold': 0.38872474480067}
            
            #cost: 3001, length: 2186, rotation: 815
            #real: cost: 3083, length: 2281, rotation: 802 
            param_best_cross = {'ba_exploration': 0.863145455156051,
                                      'max_distance': 1.69280755868826,
                                      'max_distance_part_II': 4.48375188984703,
                                      'min_bastar_cost_per_coverage': 6488.530314616084,
                                      'min_spiral_cost_per_coverage': 8141.257661974652297,
                                      'step_size': 0.553977048496769,
                                      'visited_threshold': 0.38872474480067}

            #BASTAR:
            #cost: 16062, lenth: 10575 rotation: 5487
            param1 = {  'angle_offset': 3.44800051788481,
                        'step_size': 0.963400677899873,
                        'visited_threshold': 0.257015802906527}

            #cost: 7583, lenth: 4452 rotation: 3131
            param2 = {  'angle_offset': 3.78341027362029,
                        'step_size': 0.601687134922371,
                        'visited_threshold': 0.328108983656107}

            #cost: 5013, lenth: 3049 rotation: 1964
            param3 = {  'angle_offset': 5.27158130667689,
                        'step_size': 0.517468289229711,
                        'visited_threshold': 0.455659073558674}        		
            
            #cost: 4238, lenth: 2896 rotation: 1342
            param4 = {  'angle_offset': 4.64664343656672,
                        'step_size': 0.633652049936913,
                        'visited_threshold': 0.472819723019576}   

            #cost: 3262, lenth: 2249 rotation: 1013
            param_best_cross = {  'angle_offset': 4.70135588957793,
                        'step_size': 0.523646671416283,
                        'visited_threshold': 0.403681713288835}  
            
            #cost: 6385, lenth: 4562 rotation: 1823
            param_best_brdige = {  'angle_offset': 5.33881157053433,
                        'step_size': 0.55692737194204,
                        'visited_threshold': 0.453169184364576} 

            #SPIRAL: 
            #cost: 14292, lenth: 7523 rotation: 6769
            param1 = {  'step_size': 0.999314930298507,
                        'visited_threshold': 0.32443603324225}    		

            #cost: 7431, lenth: 3990 rotation: 3441
            param2 = {  'step_size': 0.825030992319859,
                        'visited_threshold': 0.433448258850281}   

            #cost: 6466, lenth: 3218 rotation: 3248
            param3 = {  'step_size': 0.521396930930628,
                        'visited_threshold': 0.47473068968531}   

            #cost: 5787, lenth: 3101 rotation: 2686
            param4 = {  'step_size': 0.627870706339337,
                        'visited_threshold': 0.498775709725593}  

            #cost: 7213, lenth: 4440 rotation: 2773
            param_best_brdige = {
                        'step_size': 0.737114020263598,
                        'visited_threshold': 0.483088877473477}  

            #cost: 4054, lenth: 2239 rotation: 1815
            param_best_cross = {
                        'step_size': 0.664671825076571,
                        'visited_threshold': 0.499669038773602}

            #param = {'step_size': 0.5,
            #                     'visited_threshold': 0.4}
            start_point = [-20.7, 43, -1]
            start_point = np.array([28.6, -6.7, -10.3]) #garage
            #start_point =  np.array([-53.7, 54.2, -2.7]) #bridge
            #start_point = np.array([-20.7, 43, -1]) #cross
            #start_point = np.array([15.6, -16.7, -5.3])
            #start_point = np.array([0.6,0.6,0])
            #start_points = {}
            #for n in range(10):
            #    random_idx = np.random.choice(len(self.traversable_point_cloud.points), 1, replace=False)[0]
            #    start_points[n] = self.traversable_point_cloud.points[random_idx]
            #    #self.markers.append( {"point": self.traversable_point_cloud.points[random_idx], "color": RED} )
            #self.print(start_points)

            self.cpp = RandomBAstar3(self.print, motion_planner, self.coverable_point_cloud, time_limit=300, parameters = sparam3)
            self.path = self.cpp.get_cpp_path(start_point, goal_coverage=0.97)
            #self.path = self.cpp.breadth_first_search(start_point)
            #self.print(self.cpp.print_results())
            #self.path = self.cpp.get_cpp_node_path(start_point)
            self.print(self.cpp.print_stats(self.path))

            for marker in self.cpp.points_to_mark:
                self.markers.append(marker)
            
            #self.markers.append( {"point": self.path[-1], "color": RED} )            
            #self.points_to_mark = [self.path[-1]]

        if PUBLISH_FULL_PCD:
            #pcd_pub = self.create_timer(timer_period, self.point_cloud_publisher)
            self.point_cloud_publisher()

        if PUBLISH_GROUND_PCD:
            #coverable_pcd_pub = self.create_timer(timer_period, self.coverable_point_cloud_publisher)
            self.coverable_point_cloud_publisher()           
        
        if PUBLISH_TRAVERSABLE_PCD:
            #traversable_pcd_pub = self.create_timer(timer_period, self.traversable_point_cloud_publisher)  
            self.traversable_point_cloud_publisher()  
        
        HYPER_START_POS = np.array([-53.7, 54.2, -2.7])
        start_points = {
            0: np.array([-43.10443115,   3.99802136,   4.46702003]), 
            1: np.array([ 21.61431885, -33.00197983,  -2.77298403]), 
            2: np.array([-34.51068115,  12.49802208,  -4.17298126]), 
            3: np.array([ 15.9268198 , -36.00197983,  -2.6929822 ]), 
            4: np.array([38.98931885, 45.49802399,  1.19701743]), 
            5: np.array([ 3.73931861, 40.74802399,  2.83701849]), 
            6: np.array([ 15.5205698 , -31.50197792,  -2.8729825 ]), 
            7: np.array([-16.44818115, -19.25197792,  -3.58298159]), 
            8: np.array([10.52056885, 42.74802399,  2.46701956]), 
            9: np.array([53.89556885, 35.99802399,  0.33701676])}
        for point in start_points.values():
             self.markers.append({
                 "point": point,
                 "color": [0.0,0.0,1.0]
             })
        self.markers.append({
                 "point": HYPER_START_POS,
                 "color": [0.0,1.0,0.0]
             })
        #CPP_TEST = True
        if PUBLISH_MARKERS and len(self.markers):
            #for marker in self.cpp.points_to_mark:
            #    self.markers.append(marker)
            markers_pub = self.create_timer(timer_period, self.marker_publisher)
        
        if PUBLISH_PATH and len(self.path) > 0 and not PUBLISH_PATH_ANIMATION:
            path_pub = self.create_timer(timer_period, self.path_publisher)

        if PUBLISH_VISITED_PCD:
            self.point_cloud.visit_path(self.path)
            self.visited_points_pcd = self.point_cloud.get_covered_points_as_pcd()
            visited_pcd_pub = self.create_timer(timer_period, self.visited_point_cloud_publisher)
        
        if PUBLISH_VISITED_GROUND_PCD and len(self.path):
            #self.coverable_point_cloud = PointCloud(self.print, points= coverable_points)
            self.coverable_point_cloud.visit_path(self.path)
            self.visited_ground_points_pcd = self.coverable_point_cloud.get_covered_points_as_pcd()
            visited_ground_pcd_pub = self.create_timer(timer_period, self.visited_ground_point_cloud_publisher)

        if PUBLISH_PATH_ANIMATION and len(self.path) > 0:
            time.sleep(8)
            self.coverable_point_cloud.covered_points_idx = np.array([])
            path_pub = self.create_timer(animation_time_period, self.animated_path_publisher)

        if PUBLISH_SEGMENTS_ANIMATION and len(self.cpp.all_segments) > 0:
            time.sleep(8)
            self.coverable_point_cloud.covered_points_idx = np.array([])
            self.segments =  self.cpp.all_segments
            path_pub = self.create_timer(animation_time_period, self.animated_segment_publisher)
    
    def point_cloud_publisher(self):
        self.pcd_pub.publish(self.point_cloud.pcd)
    
    def coverable_point_cloud_publisher(self):
        self.coverable_pcd_pub.publish(self.coverable_point_cloud.pcd)
    
    def traversable_point_cloud_publisher(self):
        self.traversable_pcd_pub.publish(self.traversable_point_cloud.pcd)

    def inaccessible_point_cloud_publisher(self):
        self.inaccessible_pcd_pub.publish(self.inaccessible_point_cloud.pcd)
    
    def visited_point_cloud_publisher(self):
        self.visited_pcd_pub.publish(self.visited_points_pcd)

    def visited_ground_point_cloud_publisher(self):
        self.visited_ground_pcd_pub.publish(self.visited_ground_points_pcd)

    def path_publisher(self):
        path_msg = ROSMessage.path(self.path)
        self.path_pub.publish(path_msg)

    def animated_path_publisher(self):
        '''Publishes the path point by point to create an animation in RViz.
        '''
        self.animation_iteration += 10

        if self.animation_iteration >= len(self.path):
            self.path_publisher()
            return

        path_msg = ROSMessage.path(self.path[0:self.animation_iteration])
        self.path_pub.publish(path_msg)


        new_path = self.path[self.animation_iteration-10:self.animation_iteration]
        self.coverable_point_cloud.visit_path(new_path)

        #point = self.path[self.animation_iteration]
        #self.coverable_point_cloud.visit_path_to_position(point, self.path[self.animation_iteration-1])
        self.visited_ground_points_pcd = self.coverable_point_cloud.get_covered_points_as_pcd()
        self.visited_ground_point_cloud_publisher()

    def animated_segment_publisher(self):
        '''Publishes the path point by point to create an animation in RViz.
        '''
        

        if self.animation_iteration >= len(self.segments):
            #self.path_publisher()
            self.print("DONE!")
            return

        current_path = np.empty((0,3))
        for idx in range(self.animation_iteration):
            current_path = np.append(current_path, self.segments[idx].path, axis=0)

        path_msg = ROSMessage.path(current_path)
        self.path_pub.publish(path_msg)

        latest = self.segments[self.animation_iteration].path
        self.coverable_point_cloud.visit_path(latest)
        self.visited_ground_points_pcd = self.coverable_point_cloud.get_covered_points_as_pcd()
        self.visited_ground_point_cloud_publisher()
        self.animation_iteration += 1
        self.marker_publisher(self.animation_iteration)
        

    def marker_publisher(self, max=None):
        self.markers_msg = visualization_msgs.MarkerArray()
        
        for idx, marker in enumerate(self.markers[0:max]):
            stamp = self.get_clock().now().to_msg()
            msg = ROSMessage.point_marker(self.last_id, stamp, marker["point"], marker["color"])
            self.markers_msg.markers.append(msg)
            self.last_id += 1

        self.markers_pub.publish(self.markers_msg)

    def clicked_point_cb(self, msg):
        self.print(msg)
        point = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.print(self.traversable_point_cloud.distance_to_nearest(point))
        
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