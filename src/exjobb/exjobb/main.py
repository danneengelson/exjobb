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
from exjobb.NaiveRRTCPPAStar import NaiveRRTCPPAstar
from exjobb.BAstar import BAstar
from exjobb.BAstarVariant import BAstarVariant
from exjobb.Spiral import Spiral
from exjobb.RandomBAstar import RandomBAstar
from exjobb.AstarCPP import AstarCPP 

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
SMALL_POINT_CLOUD = True

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
        
        #Subscribers:
        self.rviz_sub = self.create_subscription(geometry_msgs.PointStamped, "clicked_point", self.clicked_point_cb, 100)

        self.point_cloud = PointCloud(self.print, file="pointcloud.pcd")

        traversable_points, coverable_points, inaccessible_points = self.do_terrain_assessment()        
        

        self.traversable_point_cloud = PointCloud(self.print, points= traversable_points)
        self.coverable_point_cloud = PointCloud(self.print, points= coverable_points)
        self.inaccessible_point_cloud = PointCloud(self.print, points= inaccessible_points)


        if SMALL_POINT_CLOUD:
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                [11, 15, -5.3],
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

        if not COLLECT_RESULT:
            start_pos = [ -1.35,  -0.69, -5.3]
            start_pos = [ -5.30999994, -20.29999924, -10.27468491]
            end_pos = [ -5.2,  -12.7, -10.3]  
            #start_pos = [ 24.37999916, 12.69999981, -5.31468582]
            start_pos = [12.26889705657959, 15.767305374145508, -5.22789192199707]
            end_pos = [24.03000069, 13.  ,       -5.40468597]
            start_point = self.point_cloud.find_k_nearest(start_pos, 1)[0]
            end_point = self.point_cloud.find_k_nearest(end_pos, 1)[0]
            
            


        if MOTION_PLANNER_TEST:
            self.path = motion_planner.Astar(start_point, end_point)
            
            self.markers.append( {"point": start_point, "color": RED} )
            self.markers.append( {"point": end_point, "color": BLUE} )
            if self.path is False:
                self.path = []

        if CPP_TEST:
            self.cpp = Spiral(self.print, motion_planner, self.coverable_point_cloud, time_limit=400)
            self.path = self.cpp.get_cpp_path(start_point)
            self.markers.append( {"point": self.path[-1], "color": RED} )            
            self.points_to_mark = [self.path[-1]]

        if PUBLISH_FULL_PCD:
            pcd_pub = self.create_timer(timer_period, self.point_cloud_publisher)

        if PUBLISH_MARKERS:
            for marker in self.cpp.points_to_mark:
                self.markers.append({"point": marker, "color": GREEN})
            markers_pub = self.create_timer(timer_period, self.marker_publisher)
        
        if PUBLISH_PATH:
            path_pub = self.create_timer(timer_period, self.path_publisher)

        if PUBLISH_VISITED_PCD:
            self.point_cloud.visit_path(self.path)
            self.visited_points_pcd = self.point_cloud.get_covered_points_as_pcd()
            visited_pcd_pub = self.create_timer(timer_period, self.visited_point_cloud_publisher)
        
        if PUBLISH_VISITED_GROUND_PCD:
            self.coverable_point_cloud = PointCloud(self.print, points= coverable_points)
            self.coverable_point_cloud.visit_path(self.path)
            self.visited_ground_points_pcd = self.coverable_point_cloud.get_covered_points_as_pcd()
            visited_ground_pcd_pub = self.create_timer(timer_period, self.visited_ground_point_cloud_publisher)

        if PUBLISH_PATH_ANIMATION:
            self.coverable_point_cloud.visited_points_idx = np.array([])
            path_pub = self.create_timer(animation_time_period, self.animated_path_publisher)
    
    def do_terrain_assessment(self):
        '''Calculating points which are theoretically possible to cover, iignoring
        the size of the robot.
        Returns:
            Coverable points and their indices in the point cloud as NumPy arrays.
        '''
        if DO_TERRAIN_ASSESSMENT:       
            terrain_assessment = TerrainAssessment(self.print, self.point_cloud)
            traversable_points, coverable_points, inaccessible_points = terrain_assessment.get_classified_points()

            with open('cached_coverable_points.dictionary', 'wb') as cached_pcd_file:
                cache_data = {
                    "coverable_points": coverable_points, 
                    "traversable_points": traversable_points,
                    "inaccessible_points": inaccessible_points
                    }
                pickle.dump(cache_data, cached_pcd_file)

        else:
            with open('cached_coverable_points.dictionary', 'rb') as cached_pcd_file:
                cache_data = pickle.load(cached_pcd_file)
                coverable_points = cache_data["coverable_points"]
                traversable_points = cache_data["traversable_points"]
                inaccessible_points = cache_data["inaccessible_points"]
        
        return traversable_points, coverable_points, inaccessible_points

    
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