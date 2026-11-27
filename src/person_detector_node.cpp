#include "meldog_perception/person_detector_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace meldog_perception {

    PersonDetectorNode::PersonDetectorNode(const rclcpp::NodeOptions& options)
        : Node("person_detector", options)
    {
        inference::BackendConfig backendConfig;
        backendConfig.onnxPath = declare_parameter<std::string>("model_path", "");
        backendConfig.type = inference::backendTypeFromString(declare_parameter<std::string>("backend", "auto"));
        backendConfig.engineCacheDir = declare_parameter<std::string>("engine_cache_dir", "");
        backendConfig.fp16 = declare_parameter<bool>("fp16", true);
        backendConfig.cpuThreads = static_cast<int>(declare_parameter<int64_t>("cpu_threads", 0));

        DetectorConfig detectorConfig;
        detectorConfig.scoreThreshold = static_cast<float>(declare_parameter<double>("score_threshold", 0.4));
        detectorConfig.nmsThreshold = static_cast<float>(declare_parameter<double>("nms_threshold", 0.45));
        detectorConfig.maxDetections = static_cast<int>(declare_parameter<int64_t>("max_detections", 100));
        const auto classIds = declare_parameter<std::vector<int64_t>>("class_ids", std::vector<int64_t>{0});
        detectorConfig.classIds.assign(classIds.begin(), classIds.end());

        const auto imageTopic = declare_parameter<std::string>("image_topic", "/camera/camera/color/image_raw");
        const bool publishDebugImage = declare_parameter<bool>("publish_debug_image", false);

        if (backendConfig.onnxPath.empty()) {
            throw std::invalid_argument("Parameter 'model_path' (path to the .onnx model) is required");
        }

        RCLCPP_INFO(get_logger(), "Loading %s (backend: %s)", backendConfig.onnxPath.c_str(),
                    inference::toString(backendConfig.type).c_str());
        _detector = std::make_unique<YoloDetector>(inference::createBackend(backendConfig), detectorConfig);
        RCLCPP_INFO(get_logger(), "Detector ready: %s, input %dx%d", _detector->backend().name().c_str(),
                    _detector->inputSize().width, _detector->inputSize().height);

        _detectionsPub = create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);
        if (publishDebugImage) {
            _debugImagePub = create_publisher<sensor_msgs::msg::Image>("detections/image", rclcpp::SensorDataQoS());
        }

        // Only the newest frame matters; older ones are dropped while inference runs.
        auto qos = rclcpp::SensorDataQoS();
        qos.keep_last(1);
        _imageSub = create_subscription<sensor_msgs::msg::Image>(imageTopic, qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { imageCallback(std::move(msg)); });

        _timingTimer = create_wall_timer(std::chrono::seconds(5), [this]() { logTiming(); });
    }

    void PersonDetectorNode::imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        cv_bridge::CvImageConstPtr rgb;
        try {
            // No copy when the camera already publishes rgb8 (RealSense default).
            rgb = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Cannot convert '%s' image: %s",
                                 msg->encoding.c_str(), e.what());
            return;
        }

        std::vector<Detection> detections;
        try {
            detections = _detector->detect(rgb->image);
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Inference failed: %s", e.what());
            return;
        }

        const auto& timing = _detector->lastTiming();
        _timingSum.preprocessMs += timing.preprocessMs;
        _timingSum.inferenceMs += timing.inferenceMs;
        _timingSum.postprocessMs += timing.postprocessMs;
        ++_timedFrames;

        vision_msgs::msg::Detection2DArray array;
        array.header = msg->header;
        array.detections.reserve(detections.size());

        for (const auto& detection : detections) {
            vision_msgs::msg::Detection2D out;
            out.header = msg->header;
            out.bbox.center.position.x = detection.box.x + detection.box.width / 2.0;
            out.bbox.center.position.y = detection.box.y + detection.box.height / 2.0;
            out.bbox.size_x = detection.box.width;
            out.bbox.size_y = detection.box.height;

            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = std::to_string(detection.classId);
            hypothesis.hypothesis.score = detection.score;
            out.results.push_back(hypothesis);

            array.detections.push_back(std::move(out));
        }
        _detectionsPub->publish(array);

        if (_debugImagePub && _debugImagePub->get_subscription_count() > 0) {
            publishDebugImage(*msg, rgb->image, detections);
        }
    }

    void PersonDetectorNode::publishDebugImage(const sensor_msgs::msg::Image& source, const cv::Mat& rgb,
                                               const std::vector<Detection>& detections)
    {
        cv_bridge::CvImage debug(source.header, sensor_msgs::image_encodings::RGB8, rgb.clone());

        for (const auto& detection : detections) {
            cv::rectangle(debug.image, detection.box, cv::Scalar(0, 255, 0), 2);
            const std::string label = cv::format("%d %.2f", detection.classId, detection.score);
            cv::putText(debug.image, label, detection.box.tl() + cv::Point2f(2.0F, 14.0F),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        _debugImagePub->publish(*debug.toImageMsg());
    }

    void PersonDetectorNode::logTiming()
    {
        if (_timedFrames == 0) {
            return;
        }
        const double n = static_cast<double>(_timedFrames);
        RCLCPP_INFO(get_logger(), "%.1f FPS | pre %.2f ms, infer %.2f ms, post %.2f ms",
                    n / 5.0, _timingSum.preprocessMs / n, _timingSum.inferenceMs / n, _timingSum.postprocessMs / n);
        _timingSum = DetectorTiming{};
        _timedFrames = 0;
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(meldog_perception::PersonDetectorNode)
