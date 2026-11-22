#include "meldog_perception/camera_listener.hpp"

namespace meldog_perception {

    CameraListener::CameraListener() : Node("camera_listener") 
    {
        const auto camera_topic = this->declare_parameter("camera_topic", "/camera/camera/color/image_raw");

        auto cameraQos = rclcpp::SensorDataQoS();
        cameraQos.keep_last(1);

        _cameraSub = this->create_subscription<sensor_msgs::msg::Image>(camera_topic, cameraQos, 
            [this](sensor_msgs::msg::Image::SharedPtr msg) { cameraListenerCallback(msg); });
    }

    void CameraListener::cameraListenerCallback(sensor_msgs::msg::Image::SharedPtr msg)
    {
        // TO DO
    }

}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<meldog_perception::CameraListener>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}