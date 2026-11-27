from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("meldog_perception")

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
        condition=IfCondition(LaunchConfiguration("use_camera")),
    )

    person_detector = Node(
        package="meldog_perception",
        executable="person_detector",
        name="person_detector",
        output="screen",
        parameters=[
            PathJoinSubstitution([pkg_share, "config", "person_detector.yaml"]),
            {
                "model_path": LaunchConfiguration("model_path"),
                "publish_debug_image": LaunchConfiguration("debug_image"),
            },
        ],
    )

    person_tracker = Node(
        package="meldog_perception",
        executable="person_tracker",
        name="person_tracker",
        output="screen",
        parameters=[
            PathJoinSubstitution([pkg_share, "config", "person_tracker.yaml"]),
            {"publish_debug_image": LaunchConfiguration("debug_image")},
        ],
        remappings=[("image", "/camera/camera/color/image_raw")],  # only for the debug image
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "serial_no",
            default_value="''",
            description="RealSense serial with a leading underscore. Empty selects a camera.",
        ),
        DeclareLaunchArgument(
            "use_camera",
            default_value="true",
            description="Start the RealSense driver. Set false when playing a rosbag.",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value=PathJoinSubstitution([pkg_share, "models", "yolo26n.onnx"]),
            description="YOLO model: .onnx (ONNX Runtime) or .engine (TensorRT, scripts/build_engine.sh).",
        ),
        DeclareLaunchArgument(
            "debug_image",
            default_value="false",
            description="Publish detections/image and tracks/image with drawn boxes.",
        ),
        realsense,
        person_detector,
        person_tracker,
    ])
