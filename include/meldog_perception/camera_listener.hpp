#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace meldog_perception {

    class CameraListener : public rclcpp::Node
    {
    
    public:
        
        CameraListener();

    private:
    
        void cameraListenerCallback(sensor_msgs::msg::Image::SharedPtr msg);
    
    private:

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _cameraSub;

    };
}