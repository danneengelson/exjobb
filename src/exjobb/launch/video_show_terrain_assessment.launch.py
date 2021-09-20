from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
def generate_launch_description():
    base_path = os.path.realpath(get_package_share_directory('exjobb'))
    print(base_path)
    rviz_path=base_path+'/config/rviz_conf.rviz'
    return LaunchDescription([
        Node(
            package='exjobb',
            executable='video_show_terrain_assessment',
            parameters=[
                {"use_sim_time": False},
            ]
        ),    
        Node(
            package='rviz2',
            executable='rviz2', arguments=['-d'+str(rviz_path)],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=["0", "0", "0", "0", "0", "0", "map", "my_frame"]
        ),
        ])
