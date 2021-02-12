import rclpy
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import visualization_msgs.msg as visualization_msgs

import open3d as o3d
import numpy as np

import os

#Code from https://github.com/SebastianGrans/ROS2-Point-Cloud-Demo/blob/master/pcd_demo/pcd_publisher/pcd_publisher_node.py

class PCDPublisher(Node):

    def __init__(self):
        super().__init__('PCDPublisher')
        self.pcd_publisher = self.create_publisher(visualization_msgs.Marker, 'marker', 1)
        self.timer = self.create_timer(1, self.send_marker)
        #as
    
    def send_marker(self):
        msg = visualization_msgs.Marker()
        msg.header.frame_id = "my_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.scale.x = 1.0
        msg.scale.y = 1.0
        msg.scale.z = 1.0
        msg.type = 2
        self.pcd_publisher.publish(msg)
        self.get_logger().info(str(msg))

        '''
        pcd_path = "out.pcd"
        #print(os.getcwd())
        # I use Open3D to read point clouds and meshes. It's a great library!
        self.get_logger().info("Reading point cloud file...")
        pcd = o3d.io.read_point_cloud(os.getcwd() + "/src/exjobb/exjobb/" + pcd_path)
        # I then convert it into a numpy array.
        #print(pcd.points)
        self.points = np.asarray(pcd.points)[0:100, :]
        first = self.points[0,:]
        self.points = np.asarray([p - first for p in self.points])
        #for p in self.points:
        #    print(p)  

        print(self.points.shape)
        print(self.points)
        
        # I create a publisher that publishes sensor_msgs.PointCloud2 to the 
        # topic 'pcd'. The value '10' refers to the history_depth, which I 
        # believe is related to the ROS1 concept of queue size. 
        # Read more here: 
        # http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
        self.pcd_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd', 10)
        timer_period = 1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        #self.get_logger().info("hej")
        # This rotation matrix is used for visualization purposes. It rotates
        # the point cloud on each timer callback. 
        #self.R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, np.pi/48])


        #self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)
        '''
    def timer_callback(self):
        #self.get_logger().info("hej2")
        # For visualization purposes, I rotate the point cloud with self.R 
        # to make it spin. 
        #self.points = self.points @ self.R
        # Here I use the point_cloud() function to convert the numpy array 
        # into a sensor_msgs.PointCloud2 object. The second argument is the 
        # name of the frame the point cloud will be represented in. The default
        # (fixed) frame in RViz is called 'map'
        self.pcd = self.point_cloud(self.points, '/map')
        #self.get_logger().info("hej6")   
        # Then I publish the PointCloud2 object 
        self.pcd_publisher.publish(self.pcd)
        #self.get_logger().info('Publishing: "%s"' % self.pcd)

    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        Code source:
            https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
        References:
            http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
            http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
            http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html
        """
        # In a PointCloud2 message, the point cloud is stored as an byte 
        # array. In order to unpack it, we also include some parameters 
        # which desribes the size of each individual point.
        #self.get_logger().info("hej3")    
        
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes() 

        #self.get_logger().info("hej4")
        # The fields specify what the bytes represents. The first 4 bytes 
        # represents the x-coordinate, the next 4 the y-coordinate, etc.
        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]

        # The PointCloud2 message also has a header which specifies which 
        # coordinate frame it is represented in. 
        header = std_msgs.Header(frame_id=parent_frame)

        #self.get_logger().info("hej5")

        return sensor_msgs.PointCloud2(
            header=header,
            height=1, 
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3), # Every point consists of three float32s.
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )

def main(args=None):
    rclpy.init(args=args)
    ros_node = PCDPublisher()
    rclpy.spin(ros_node)
    pcd_publisher.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()