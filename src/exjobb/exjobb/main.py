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

from exjobb.PointCloud import PointCloud
from exjobb.TerrainAssessment import TerrainAssessment
from exjobb.MotionPlanner import MotionPlanner
from exjobb.TA_PointClassification import PointClassification
from exjobb.BAstar import BAstar
from exjobb.BAstarVariant import BAstarVariant
from exjobb.Spiral import Spiral
from exjobb.RandomBAstar import RandomBAstar
from exjobb.Map import Map

from exjobb.ROSMessage import RED, GREEN, BLUE
import exjobb.ROSMessage as ROSMessage

DO_TERRAIN_ASSESSMENT = False
PUBLISH_FULL_PCD = True
PUBLISH_GROUND_PCD = True
PUBLISH_MARKERS = True
PUBLISH_PATH = True
PUBLISH_PATH_ANIMATION = False
PUBLISH_VISITED_PCD = False 
PUBLISH_VISITED_GROUND_PCD = True
PUBLISH_TRAVERSABLE_PCD = True
PUBLISH_INACCESSIBLE_PCD = True

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
        animation_time_period = 0.1
        self.animation_iteration = 0
        self.path = []
        
        #Subscribers:
        self.rviz_sub = self.create_subscription(geometry_msgs.PointStamped, "clicked_point", self.clicked_point_cb, 100)


        #environment = PointCloudEnvironment(self.print, "cached_coverable_points.dictionary", "pointcloud.pcd")
        #environment = MapEnvironment(self.print, "simple_open.dictionary", "src/exjobb/maps/map_simple_open.png", 0.015)
        #environment = MapEnvironment(self.print, "map_ipa_apartment.dictionary", "src/exjobb/maps/map_ipa_apartment.png", 0.05)




        NEW_POINTCLOUD = False
        if NEW_POINTCLOUD:
            
            #environment = MapEnvironment(self.print, "simple_open.dictionary", "src/exjobb/maps/map_simple_open.png", 0.015)
            self.point_cloud = PointCloud(self.print, file="bridge.pcd", new_point_cloud=True) 
        else:
            #environment = PointCloudEnvironment(self.print, "pointcloud2.dictionary", "pointcloud2.pcd", False) #x,y = #351451.84, 4022966.5   Street
            #environment = PointCloudEnvironment(self.print, "pointcloud3.dictionary", "pointcloud3.pcd", False) #x,y = (351609.25, 4022912.0)  Same Underground garage one floor
            #environment = PointCloudEnvironment(self.print, "pointcloud4.dictionary", "pointcloud4.pcd", False) #x,y = (326815.75, 4152473.25)  Busy street, with cars
            #environment = PointCloudEnvironment(self.print, "cached_coverable_points.dictionary", "pointcloud.pcd")
            #environment = MapEnvironment(self.print, "map_ipa_apartment.dictionary", "src/exjobb/maps/map_ipa_apartment.png", 0.05)
            #environment = PointCloudEnvironment(self.print, "bridge_2_small.dictionary", "bridge_2_small.pcd", False) 
            environment = PointCloudEnvironment(self.print, "bridge_terrain_assessment.dictionary", "bridge_2.pcd", False) 
            #environment = PointCloudEnvironment(self.print, "pre_garage_terrain_assessment.dictionary", "garage.pcd", False)
            #self.point_cloud = PointCloud(self.print, file="garage.pcd", new_point_cloud=False) 


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
        
        if PUBLISH_GROUND_PCD:
            coverable_pcd_pub = self.create_timer(timer_period, self.coverable_point_cloud_publisher)           
        
        if PUBLISH_TRAVERSABLE_PCD:
            traversable_pcd_pub = self.create_timer(timer_period, self.traversable_point_cloud_publisher)    
        
        if PUBLISH_INACCESSIBLE_PCD:
            inaccessible_pcd_pub = self.create_timer(timer_period, self.inaccessible_point_cloud_publisher)                

        if MOTION_PLANNER_TEST:
            start_pos = [5.0625    ,  91.05000305, -32.58319855]
            end_pos = [  0.8125    ,  93.30000305, -32.33319855]
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

            param = {'angle_offset': 3.1639298965427103, 'step_size': 0.9363295148117978, 'visited_threshold': 0.7037583980794171}

            start_point = [-53.7, 54.2, -2.7]
            start_points = {}
            for n in range(10):
                random_idx = np.random.choice(len(self.traversable_point_cloud.points), 1, replace=False)[0]
                start_points[n] = self.traversable_point_cloud.points[random_idx]
                self.markers.append( {"point": self.traversable_point_cloud.points[random_idx], "color": RED} )
            self.print(start_points)

            self.cpp = BAstar(self.print, motion_planner, self.coverable_point_cloud, time_limit=400, parameters = param)
            self.path = self.cpp.get_cpp_path(start_point)
            #self.path = self.cpp.get_cpp_node_path(start_point)
            self.print(self.cpp.print_stats(self.path))
            self.markers.append( {"point": self.path[-1], "color": RED} )            
            self.points_to_mark = [self.path[-1]]

        if PUBLISH_FULL_PCD:
            pcd_pub = self.create_timer(timer_period, self.point_cloud_publisher)

        if PUBLISH_MARKERS:
            for marker in self.cpp.points_to_mark:
                self.markers.append({"point": marker, "color": GREEN})
            markers_pub = self.create_timer(timer_period, self.marker_publisher)
        
        if PUBLISH_PATH and len(self.path) > 0:
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
            self.coverable_point_cloud.visited_points_idx = np.array([])
            path_pub = self.create_timer(animation_time_period, self.animated_path_publisher)
    
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
        self.animation_iteration += 1

        if self.animation_iteration >= len(self.path):
            self.path_publisher()
            return

        path_msg = ROSMessage.path(self.path[0:self.animation_iteration])
        self.path_pub.publish(path_msg)

        point = self.path[self.animation_iteration]
        self.coverable_point_cloud.visit_path_to_position(point, self.path[self.animation_iteration-1])
        self.visited_ground_points_pcd = self.coverable_point_cloud.get_covered_points_as_pcd()
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