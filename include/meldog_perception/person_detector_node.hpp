#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "meldog_perception/yolo_detector.hpp"

namespace meldog_perception {

    // Runs YoloDetector on the color stream and publishes person detections.
    class PersonDetectorNode : public rclcpp::Node
    {

    public:

        explicit PersonDetectorNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    private:

        void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg);

        void publishDebugImage(const sensor_msgs::msg::Image& source, const cv::Mat& rgb,
                               const std::vector<Detection>& detections);

    private:

        std::unique_ptr<YoloDetector> _detector;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _imageSub;
        rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr _detectionsPub;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _debugImagePub;
    };
}
