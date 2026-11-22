from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"
            ])
        ),
        launch_arguments={
            "camera_namespace": "camera",
            "camera_name": "camera",
            "serial_no": LaunchConfiguration("serial_no"),
            "enable_color": "true",
            "rgb_camera.color_profile": "640,480,30",  # matches the exported model input
            "enable_depth": "false",
        }.items(),
    )

    camera_listener = Node(
        package="meldog_perception",
        executable="camera_listener",
        name="camera_listener",
        output="screen",
        parameters=[{"camera_topic": "/camera/camera/color/image_raw"}],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "serial_no",
            default_value="''",
            description="RealSense serial with a leading underscore. Empty selects a camera.",
        ),
        realsense,
        camera_listener,
    ])
