from launch import LaunchDescription
from launch_ros.actions import Node
def generate_launch_description():
    return LaunchDescription([
        Node(
            package='exjobb',
            executable='video_run_best',
            parameters=[
                {"use_sim_time": False},
            ]
        ),    
        Node(
            package='rviz2',
            executable='rviz2',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=["0", "0", "0", "0", "0", "0", "map", "my_frame"]
        ),
        ])
