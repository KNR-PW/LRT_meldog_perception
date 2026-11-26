#pragma once

#include <deque>
#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "meldog_perception/tracking/byte_tracker.hpp"

namespace meldog_perception {

    // Tracks people from person_detector's detections and gives each one a stable ID.
    // Publishes the tracks as Detection2DArray with Detection2D::id set to the track ID.
    class PersonTrackerNode : public rclcpp::Node
    {

    public:

        explicit PersonTrackerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    private:

        void detectionsCallback(vision_msgs::msg::Detection2DArray::ConstSharedPtr msg);
        void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg);

        void publishDebugImage(const std_msgs::msg::Header& header, const std::vector<tracking::Track>& tracks);

    private:

        tracking::ByteTracker _tracker;

        rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr _detectionsSub;
        rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr _tracksPub;

        // Debug image: the frame the detections came from, matched by timestamp.
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _imageSub;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _debugImagePub;
        std::deque<sensor_msgs::msg::Image::ConstSharedPtr> _recentImages;
        std::mutex _imagesMutex;
    };
}
