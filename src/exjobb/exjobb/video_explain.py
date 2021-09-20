
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
PATH_NS = "spiral_4"
PATH_NS = "bastar_4"

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
            all_results = deepcopy(cache_data)
            param = list(filter(lambda x: x["ns"] == PATH_NS, all_results))[0]["param"]
            cpp_name = list(filter(lambda x: x["ns"] == PATH_NS, all_results))[0]["cpp"]
            

        #Create Environment
        environment = PointCloudEnvironment(self.print, "garage_terrain_assessment.dictionary", "garage.pcd", False)
        point_cloud  = environment.full_pcd
        traversable_point_cloud = environment.traversable_pcd
        coverable_point_cloud = environment.coverable_pcd
        motion_planner = MotionPlanner(self.print, traversable_point_cloud) 


        TIME_LIMIT = 400
        goal_coverage = 0.97
        start_point = np.array([28.6, -6.7, -10.3]) #garage
        #start_point =  np.array([-53.7, 54.2, -2.7]) #bridge
        #start_point = np.array([-20.7, 43, -1]) #cross

        if cpp_name == "BA*":
            cpp = BAstar(self.print, motion_planner, coverable_point_cloud, time_limit=TIME_LIMIT, parameters = param)
        if cpp_name == "Inward Spiral":
            cpp = Spiral(self.print, motion_planner, coverable_point_cloud, time_limit=TIME_LIMIT, parameters = param)
        if cpp_name == "Sampled BA*":
            cpp = RandomBAstar3(self.print, motion_planner, coverable_point_cloud, time_limit=TIME_LIMIT, parameters = param)

        path = cpp.get_cpp_path(start_point, goal_coverage=goal_coverage)
        all_segments = cpp.all_segments 
        all_movements = cpp.all_movements
        markers = cpp.points_to_mark

        self.pcd_pub.publish(point_cloud.point_cloud(point_cloud.points, 'my_frame'))
        self.coverable_pcd_pub.publish(coverable_point_cloud.point_cloud(coverable_point_cloud.points, 'my_frame'))
        self.traversable_pcd_pub.publish(traversable_point_cloud.point_cloud(traversable_point_cloud.points, 'my_frame'))

        #Only markers
        self.markers_msg = visualization_msgs.MarkerArray()
        self.last_id = 0
        for idx, marker in enumerate(markers):
            stamp = self.get_clock().now().to_msg()
            msg = ROSMessage.point_marker(self.last_id, stamp, marker["point"], marker["color"], "markers")
            self.markers_msg.markers.append(msg)
            self.last_id += 1
        

        #Add segments
        coverable_point_cloud.covered_points_idx = np.array([])
        for idx, segment in enumerate(all_segments):
            stamp = self.get_clock().now().to_msg()
            msg = ROSMessage.line_marker(self.last_id, stamp, segment.path, [0.1,1.0,1.0], "segment"+str(idx))
            self.markers_msg.markers.append(msg)
            self.last_id += 1
            coverable_point_cloud.visit_path(segment.path)

        #Add Movements
        for idx, move in enumerate(all_movements):
            stamp = self.get_clock().now().to_msg()
            msg = ROSMessage.line_marker(self.last_id, stamp, move.path, [1.0,0.5,0.5], "move"+str(idx))
            self.markers_msg.markers.append(msg)
            self.last_id += 1
            
            

        visited_ground_points_pcd = coverable_point_cloud.get_covered_points_as_pcd()
        self.visited_ground_pcd_pub.publish(visited_ground_points_pcd)
        self.markers_pub.publish(self.markers_msg)
        return


        current_max = 0
        while current_max < len(path):
            self.markers_msg = visualization_msgs.MarkerArray()
            stamp = self.get_clock().now().to_msg()
            msg = ROSMessage.line_marker(0, stamp, path[0:current_max], [0.1,1.0,1.0], PATH_NS)
            self.markers_msg.markers = [msg]
            current_max += 10
            self.markers_pub.publish(self.markers_msg)

    def marker_publisher(self, max=None):
        self.markers_msg = visualization_msgs.MarkerArray()
        self.last_id = 0
        for idx, path in enumerate([]):
            
            stamp = self.get_clock().now().to_msg()

            if path.get("path", False) is False:
                path["path"] = self.prev_file[idx]["path"]

            msg = ROSMessage.line_marker(self.last_id, stamp, path["path"], [0.1,1.0,1.0], path["ns"])
            self.markers_msg.markers.append(msg)
            self.last_id += 1

        self.markers_pub.publish(self.markers_msg)

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