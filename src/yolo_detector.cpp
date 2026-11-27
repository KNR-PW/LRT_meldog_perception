#include "meldog_perception/yolo_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace meldog_perception {

    namespace {

        constexpr int kPersonClass = 0;     // COCO

        // Ultralytics pads with grey 114.
        const cv::Scalar kPadColor(114, 114, 114);
    }

    YoloDetector::YoloDetector(const std::string& networkPath, const DetectorConfig& config)
        : _config(config)
    {
        validateConfig(config);

        _yoloEngine = inference::createYoloEngine(networkPath);

        const auto [height, width] = _yoloEngine->getInputDim();
        _inputSize = cv::Size(static_cast<int>(width), static_cast<int>(height));
        _input.resize(3 * height * width);
    }

    std::vector<Detection> YoloDetector::detect(const cv::Mat& rgbImage)
    {
        if (rgbImage.empty() || rgbImage.type() != CV_8UC3) {
            throw std::invalid_argument("[YoloDetector] detect() expects a non-empty 8-bit RGB image");
        }

        const Letterbox letterbox = preprocess(rgbImage);
        std::vector<Detection> detections = decode(_yoloEngine->infer(_input));

        // Back from network input pixels to the original image.
        const cv::Rect2f imageRect(0.0F, 0.0F, static_cast<float>(rgbImage.cols), static_cast<float>(rgbImage.rows));
        for (auto& detection : detections) {
            detection.box.x = (detection.box.x - letterbox.padX) / letterbox.scale;
            detection.box.y = (detection.box.y - letterbox.padY) / letterbox.scale;
            detection.box.width /= letterbox.scale;
            detection.box.height /= letterbox.scale;
            detection.box &= imageRect;
        }
        detections.erase(std::remove_if(detections.begin(), detections.end(),
            [](const Detection& d) { return d.box.area() <= 0.0F; }), detections.end());

        return detections;
    }

    YoloDetector::Letterbox YoloDetector::preprocess(const cv::Mat& rgbImage)
    {
        Letterbox letterbox;
        const cv::Mat* source = &rgbImage;

        if (rgbImage.size() != _inputSize) {
            letterbox.scale = std::min(static_cast<float>(_inputSize.width) / static_cast<float>(rgbImage.cols),
                                       static_cast<float>(_inputSize.height) / static_cast<float>(rgbImage.rows));
            const int width = static_cast<int>(std::round(static_cast<float>(rgbImage.cols) * letterbox.scale));
            const int height = static_cast<int>(std::round(static_cast<float>(rgbImage.rows) * letterbox.scale));
            const int left = (_inputSize.width - width) / 2;
            const int top = (_inputSize.height - height) / 2;
            letterbox.padX = static_cast<float>(left);
            letterbox.padY = static_cast<float>(top);

            _letterboxed.create(_inputSize, CV_8UC3);
            _letterboxed.setTo(kPadColor);
            cv::resize(rgbImage, _letterboxed(cv::Rect(left, top, width, height)), cv::Size(width, height),
                       0, 0, cv::INTER_LINEAR);
            source = &_letterboxed;
        }

        // HWC uint8 -> CHW float in [0, 1], written straight into the input buffer.
        const int planeSize = _inputSize.area();
        for (int channel = 0; channel < 3; ++channel) {
            cv::Mat plane(_inputSize, CV_32FC1, _input.data() + channel * planeSize);
            cv::Mat channel8u;
            cv::extractChannel(*source, channel8u, channel);
            channel8u.convertTo(plane, CV_32F, 1.0 / 255.0);
        }

        return letterbox;
    }

    std::vector<Detection> YoloDetector::decode(const std::vector<float>& output) const
    {
        const std::size_t rows = _yoloEngine->getDetectionCount();
        const std::size_t stride = _yoloEngine->getDetectionSize();

        // Rows: x1 y1 x2 y2 score class
        std::vector<Detection> detections;
        for (std::size_t i = 0; i < rows; ++i) {
            const float* row = output.data() + i * stride;
            const float score = row[4];
            const int classId = static_cast<int>(row[5]);
            if (score < _config.scoreThreshold || classId != kPersonClass) {
                continue;
            }
            detections.push_back({cv::Rect2f(cv::Point2f(row[0], row[1]), cv::Point2f(row[2], row[3])), score, classId});
        }

        std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
        if (detections.size() > _config.maxDetections) {
            detections.resize(_config.maxDetections);
        }
        return detections;
    }

    void YoloDetector::validateConfig(const DetectorConfig& config)
    {
        if (config.scoreThreshold < 0.0F || config.scoreThreshold > 1.0F) {
            throw std::invalid_argument("[YoloDetector] scoreThreshold must be in [0, 1]");
        }
        if (config.maxDetections == 0) {
            throw std::invalid_argument("[YoloDetector] maxDetections must be at least 1");
        }
    }
}
