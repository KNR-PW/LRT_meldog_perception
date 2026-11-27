// Runs the person detector on an image file without ROS, to check that a model
// (.onnx or .engine) gives the expected boxes. Compare with
// scripts/reference_detect.py; speed is measured by scripts/build_engine.sh.
//
//   ros2 run meldog_perception detect_image models/yolo26n.engine bus.jpg

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "meldog_perception/yolo_detector.hpp"

namespace {

    void usage()
    {
        std::cerr << "Usage: detect_image <model.onnx|model.engine> <image> [--threshold 0.4] [--out result.jpg]\n";
    }
}

int main(int argc, char** argv)
{
    using namespace meldog_perception;

    if (argc < 3) {
        usage();
        return 1;
    }

    const std::string modelPath = argv[1];
    const std::string imagePath = argv[2];

    DetectorConfig config;
    std::string outPath;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threshold" && i + 1 < argc) {
            config.scoreThreshold = std::stof(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        } else {
            usage();
            return 1;
        }
    }

    const cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Cannot read image " << imagePath << "\n";
        return 1;
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::unique_ptr<YoloDetector> detectorPtr;
    try {
        detectorPtr = std::make_unique<YoloDetector>(modelPath, config);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    YoloDetector& detector = *detectorPtr;
    std::cout << "model " << modelPath << ", input " << detector.inputSize().width << "x"
              << detector.inputSize().height << ", image " << rgb.cols << "x" << rgb.rows << "\n";

    const std::vector<Detection> detections = detector.detect(rgb);

    std::cout << detections.size() << " detections\n";
    for (const auto& d : detections) {
        std::printf("  class %2d  score %.3f  box [%.1f, %.1f, %.1f, %.1f]\n", d.classId, d.score, d.box.x, d.box.y,
                    d.box.x + d.box.width, d.box.y + d.box.height);
    }

    if (!outPath.empty()) {
        cv::Mat out = bgr.clone();
        for (const auto& d : detections) {
            cv::rectangle(out, d.box, cv::Scalar(0, 255, 0), 2);
            cv::putText(out, cv::format("%.2f", d.score), d.box.tl() + cv::Point2f(2.0F, 14.0F),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        cv::imwrite(outPath, out);
        std::cout << "saved " << outPath << "\n";
    }
    return 0;
}
