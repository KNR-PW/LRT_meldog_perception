#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "meldog_perception/detection.hpp"
#include "meldog_perception/inference/i_yolo_engine.hpp"

namespace meldog_perception {

    struct DetectorConfig
    {
        float scoreThreshold = 0.4F;
        std::size_t maxDetections = 100;
    };

    // Person detector on YOLO26 (COCO class 0). Expects an RGB image and
    // letterboxes it when its size differs from the network input.
    class YoloDetector
    {

    public:

        // networkPath: .onnx (ONNX Runtime) or .engine (TensorRT)
        YoloDetector(const std::string& networkPath, const DetectorConfig& config);

        std::vector<Detection> detect(const cv::Mat& rgbImage);

        cv::Size inputSize() const { return _inputSize; }

    private:

        struct Letterbox
        {
            float scale = 1.0F;
            float padX = 0.0F;
            float padY = 0.0F;
        };

        Letterbox preprocess(const cv::Mat& rgbImage);

        std::vector<Detection> decode(const std::vector<float>& output) const;

        static void validateConfig(const DetectorConfig& config);

    private:

        std::unique_ptr<inference::IYoloEngine> _yoloEngine;
        DetectorConfig _config;

        cv::Size _inputSize;
        std::vector<float> _input;     // CHW float, reused between frames
        cv::Mat _letterboxed;
    };
}
