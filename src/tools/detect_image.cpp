// Runs the detector on an image file without ROS. Use it to check a model
// export and to benchmark a backend, e.g. on the Jetson:
//
//   ros2 run meldog_perception detect_image models/yolo26n.onnx bus.jpg --backend tensorrt --runs 200

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "meldog_perception/yolo_detector.hpp"

namespace {

    void usage()
    {
        std::cerr << "Usage: detect_image <model.onnx> <image> [--backend auto|tensorrt|onnxruntime]\n"
                     "                    [--runs N] [--threshold 0.4] [--all-classes] [--fp32]\n"
                     "                    [--engine-cache DIR] [--out result.jpg]\n";
    }
}

int main(int argc, char** argv)
{
    using namespace meldog_perception;

    if (argc < 3) {
        usage();
        return 1;
    }

    inference::BackendConfig backendConfig;
    backendConfig.onnxPath = argv[1];
    const std::string imagePath = argv[2];

    DetectorConfig detectorConfig;
    int runs = 1;
    std::string outPath;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(arg + " needs a value");
            }
            return argv[++i];
        };

        if (arg == "--backend") {
            backendConfig.type = inference::backendTypeFromString(next());
        } else if (arg == "--runs") {
            runs = std::max(1, std::stoi(next()));
        } else if (arg == "--threshold") {
            detectorConfig.scoreThreshold = std::stof(next());
        } else if (arg == "--all-classes") {
            detectorConfig.classIds.clear();
        } else if (arg == "--fp32") {
            backendConfig.fp16 = false;
        } else if (arg == "--engine-cache") {
            backendConfig.engineCacheDir = next();
        } else if (arg == "--out") {
            outPath = next();
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

    YoloDetector detector(inference::createBackend(backendConfig), detectorConfig);
    std::cout << "backend: " << detector.backend().name() << ", input " << detector.inputSize().width << "x"
              << detector.inputSize().height << ", image " << rgb.cols << "x" << rgb.rows << "\n";

    // The first run includes lazy initialisation, so it is left out of the averages.
    std::vector<Detection> detections = detector.detect(rgb);

    DetectorTiming sum;
    for (int run = 0; run < runs; ++run) {
        detections = detector.detect(rgb);
        sum.preprocessMs += detector.lastTiming().preprocessMs;
        sum.inferenceMs += detector.lastTiming().inferenceMs;
        sum.postprocessMs += detector.lastTiming().postprocessMs;
    }

    std::cout << detections.size() << " detections\n";
    for (const auto& d : detections) {
        std::printf("  class %2d  score %.3f  box [%.1f, %.1f, %.1f, %.1f]\n", d.classId, d.score, d.box.x, d.box.y,
                    d.box.x + d.box.width, d.box.y + d.box.height);
    }
    std::printf("mean over %d runs: pre %.2f ms, infer %.2f ms, post %.2f ms\n", runs, sum.preprocessMs / runs,
                sum.inferenceMs / runs, sum.postprocessMs / runs);

    if (!outPath.empty()) {
        cv::Mat out = bgr.clone();
        for (const auto& d : detections) {
            cv::rectangle(out, d.box, cv::Scalar(0, 255, 0), 2);
            cv::putText(out, cv::format("%d %.2f", d.classId, d.score), d.box.tl() + cv::Point2f(2.0F, 14.0F),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        cv::imwrite(outPath, out);
        std::cout << "saved " << outPath << "\n";
    }
    return 0;
}
