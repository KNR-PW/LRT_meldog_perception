#include "meldog_perception/person_tracker_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace meldog_perception {

    namespace {

        constexpr size_t kMaxRecentImages = 30;

        Detection fromMessage(const vision_msgs::msg::Detection2D& msg)
        {
            Detection detection;
            const auto& bbox = msg.bbox;
            detection.box = cv::Rect2f(static_cast<float>(bbox.center.position.x - bbox.size_x / 2.0),
                                       static_cast<float>(bbox.center.position.y - bbox.size_y / 2.0),
                                       static_cast<float>(bbox.size_x), static_cast<float>(bbox.size_y));

            for (const auto& result : msg.results) {
                if (result.hypothesis.score <= detection.score) {
                    continue;
                }
                detection.score = static_cast<float>(result.hypothesis.score);
                try {
                    detection.classId = std::stoi(result.hypothesis.class_id);
                } catch (const std::exception&) {
                    detection.classId = -1;
                }
            }
            return detection;
        }

        double toSeconds(const builtin_interfaces::msg::Time& stamp)
        {
            return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
        }

        cv::Scalar colorForId(int id)
        {
            cv::Mat3b hsv(1, 1, cv::Vec3b(static_cast<uchar>((id * 47) % 180), 220, 255));
            cv::Mat3b rgb;
            cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
            const cv::Vec3b c = rgb(0, 0);
            return {static_cast<double>(c[0]), static_cast<double>(c[1]), static_cast<double>(c[2])};
        }
    }

    PersonTrackerNode::PersonTrackerNode(const rclcpp::NodeOptions& options)
        : Node("person_tracker", options)
    {
        tracking::TrackerConfig config;
        config.highThreshold = static_cast<float>(declare_parameter<double>("high_threshold", 0.5));
        config.lowThreshold = static_cast<float>(declare_parameter<double>("low_threshold", 0.1));
        config.newTrackThreshold = static_cast<float>(declare_parameter<double>("new_track_threshold", 0.6));
        config.matchThreshold = static_cast<float>(declare_parameter<double>("match_threshold", 0.8));
        config.secondMatchThreshold = static_cast<float>(declare_parameter<double>("second_match_threshold", 0.5));
        config.unconfirmedMatchThreshold =
            static_cast<float>(declare_parameter<double>("unconfirmed_match_threshold", 0.7));
        config.fuseScore = declare_parameter<bool>("fuse_score", true);
        config.lostTimeout = declare_parameter<double>("lost_timeout", 1.0);
        config.referenceFps = declare_parameter<double>("reference_fps", 30.0);
        _tracker = tracking::ByteTracker(config);

        const auto detectionsTopic = declare_parameter<std::string>("detections_topic", "detections");
        const auto imageTopic = declare_parameter<std::string>("image_topic", "/camera/camera/color/image_raw");
        const bool publishDebugImage = declare_parameter<bool>("publish_debug_image", false);

        _tracksPub = create_publisher<vision_msgs::msg::Detection2DArray>("tracks", 10);
        _detectionsSub = create_subscription<vision_msgs::msg::Detection2DArray>(detectionsTopic, 10,
            [this](vision_msgs::msg::Detection2DArray::ConstSharedPtr msg) { detectionsCallback(std::move(msg)); });

        if (publishDebugImage) {
            _debugImagePub = create_publisher<sensor_msgs::msg::Image>("tracks/image", rclcpp::SensorDataQoS());
            auto qos = rclcpp::SensorDataQoS();
            qos.keep_last(5);
            _imageSub = create_subscription<sensor_msgs::msg::Image>(imageTopic, qos,
                [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { imageCallback(std::move(msg)); });
        }

        RCLCPP_INFO(get_logger(), "Tracking people from '%s' (lost timeout %.1f s)",
                    _detectionsSub->get_topic_name(), config.lostTimeout);
    }

    void PersonTrackerNode::detectionsCallback(vision_msgs::msg::Detection2DArray::ConstSharedPtr msg)
    {
        std::vector<Detection> detections;
        detections.reserve(msg->detections.size());
        for (const auto& detection : msg->detections) {
            detections.push_back(fromMessage(detection));
        }

        const auto& tracks = _tracker.update(detections, toSeconds(msg->header.stamp));

        for (const auto& event : _tracker.events()) {
            const bool entered = event.type == tracking::TrackEvent::Type::Entered;
            RCLCPP_INFO(get_logger(), "Person %d %s (%zu visible, %zu lost)", event.id,
                        entered ? "entered" : "left", tracks.size(), _tracker.lostCount());
        }

        vision_msgs::msg::Detection2DArray array;
        array.header = msg->header;
        array.detections.reserve(tracks.size());
        for (const auto& track : tracks) {
            vision_msgs::msg::Detection2D out;
            out.header = msg->header;
            out.id = std::to_string(track.id);
            out.bbox.center.position.x = track.box.x + track.box.width / 2.0;
            out.bbox.center.position.y = track.box.y + track.box.height / 2.0;
            out.bbox.size_x = track.box.width;
            out.bbox.size_y = track.box.height;

            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = std::to_string(track.classId);
            hypothesis.hypothesis.score = track.score;
            out.results.push_back(hypothesis);

            array.detections.push_back(std::move(out));
        }
        _tracksPub->publish(array);

        if (_debugImagePub && _debugImagePub->get_subscription_count() > 0) {
            publishDebugImage(msg->header, tracks);
        }
    }

    void PersonTrackerNode::imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(_imagesMutex);
        _recentImages.push_back(std::move(msg));
        while (_recentImages.size() > kMaxRecentImages) {
            _recentImages.pop_front();
        }
    }

    void PersonTrackerNode::publishDebugImage(const std_msgs::msg::Header& header,
                                              const std::vector<tracking::Track>& tracks)
    {
        sensor_msgs::msg::Image::ConstSharedPtr image;
        {
            std::lock_guard<std::mutex> lock(_imagesMutex);
            for (auto it = _recentImages.rbegin(); it != _recentImages.rend(); ++it) {
                if ((*it)->header.stamp == header.stamp) {
                    image = *it;
                    break;
                }
            }
        }
        if (!image) {
            return;     // the frame has not arrived (or was dropped); skip this debug image
        }

        cv_bridge::CvImageConstPtr rgb;
        try {
            rgb = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::RGB8);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Cannot convert '%s' image: %s",
                                 image->encoding.c_str(), e.what());
            return;
        }

        cv_bridge::CvImage debug(header, sensor_msgs::image_encodings::RGB8, rgb->image.clone());
        for (const auto& track : tracks) {
            const cv::Scalar color = colorForId(track.id);
            cv::rectangle(debug.image, track.box, color, 2);

            // Where the person will be in 0.5 s at the current velocity.
            const cv::Point2f center(track.box.x + track.box.width / 2.0F, track.box.y + track.box.height / 2.0F);
            cv::arrowedLine(debug.image, center, center + track.velocity * 0.5F, color, 2);

            const std::string label = cv::format("#%d %.2f", track.id, track.score);
            cv::putText(debug.image, label, track.box.tl() + cv::Point2f(2.0F, 16.0F),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
        _debugImagePub->publish(*debug.toImageMsg());
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(meldog_perception::PersonTrackerNode)
