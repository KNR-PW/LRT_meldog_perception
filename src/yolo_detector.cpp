#include "meldog_perception/yolo_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace meldog_perception {

    namespace {

        using Clock = std::chrono::steady_clock;

        double millisecondsSince(Clock::time_point start)
        {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }

        // Ultralytics pads with grey 114.
        const cv::Scalar kPadColor(114, 114, 114);
    }

    YoloDetector::YoloDetector(std::unique_ptr<inference::InferenceBackend> backend, DetectorConfig config)
        : _backend(std::move(backend)), _config(std::move(config))
    {
        if (!_backend) {
            throw std::invalid_argument("YoloDetector needs a backend");
        }

        const auto& input = _backend->inputs().front().shape;
        if (input.size() != 4 || input[1] != 3 || input[2] <= 0 || input[3] <= 0) {
            throw std::runtime_error("Expected a fixed-size [1, 3, H, W] input; export the model with a fixed imgsz");
        }
        _inputSize = cv::Size(static_cast<int>(input[3]), static_cast<int>(input[2]));
        _inputTensor.shape = {1, 3, input[2], input[3]};
        _inputTensor.data.resize(_inputTensor.elementCount());

        // Detections come from the first rank-3 output; a segmentation model adds
        // a rank-4 mask prototype output that is ignored here.
        const auto& outputs = _backend->outputs();
        const auto it = std::find_if(outputs.begin(), outputs.end(),
            [](const inference::TensorInfo& info) { return info.shape.size() == 3; });
        if (it == outputs.end()) {
            throw std::runtime_error("No [1, N, C] detection output found in the model");
        }
        _detectionOutput = static_cast<size_t>(std::distance(outputs.begin(), it));
    }

    std::vector<Detection> YoloDetector::detect(const cv::Mat& rgbImage)
    {
        if (rgbImage.empty() || rgbImage.type() != CV_8UC3) {
            throw std::invalid_argument("detect() expects a non-empty 8-bit RGB image");
        }

        auto start = Clock::now();
        const Letterbox letterbox = preprocess(rgbImage);
        _timing.preprocessMs = millisecondsSince(start);

        start = Clock::now();
        const auto& outputs = _backend->run(_inputTensor);
        _timing.inferenceMs = millisecondsSince(start);

        start = Clock::now();
        const inference::Tensor& output = outputs.at(_detectionOutput);
        if (output.shape.size() != 3) {
            throw std::runtime_error("Unexpected detection output rank");
        }

        // End-to-end head: [1, N, 6+] rows of x1 y1 x2 y2 score class.
        // NMS head:        [1, 4+classes, anchors] columns of cx cy w h scores...
        const bool endToEnd = output.shape[1] > output.shape[2];
        std::vector<Detection> detections = endToEnd ? decodeEndToEnd(output) : decodeWithNms(output);

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

        _timing.postprocessMs = millisecondsSince(start);
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

        // HWC uint8 -> CHW float in [0, 1], written straight into the input tensor.
        const int planeSize = _inputSize.area();
        for (int channel = 0; channel < 3; ++channel) {
            cv::Mat plane(_inputSize, CV_32FC1, _inputTensor.data.data() + channel * planeSize);
            cv::Mat channel8u;
            cv::extractChannel(*source, channel8u, channel);
            channel8u.convertTo(plane, CV_32F, 1.0 / 255.0);
        }

        return letterbox;
    }

    std::vector<Detection> YoloDetector::decodeEndToEnd(const inference::Tensor& output) const
    {
        const auto rows = static_cast<size_t>(output.shape[1]);
        const auto stride = static_cast<size_t>(output.shape[2]);
        if (stride < 6) {
            throw std::runtime_error("End-to-end output needs at least 6 values per detection");
        }

        std::vector<Detection> detections;
        for (size_t i = 0; i < rows; ++i) {
            const float* row = output.data.data() + i * stride;
            const float score = row[4];
            const int classId = static_cast<int>(row[5]);
            if (score < _config.scoreThreshold || !keepClass(classId)) {
                continue;
            }
            detections.push_back({cv::Rect2f(cv::Point2f(row[0], row[1]), cv::Point2f(row[2], row[3])), score, classId});
        }

        std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
        if (detections.size() > static_cast<size_t>(_config.maxDetections)) {
            detections.resize(static_cast<size_t>(_config.maxDetections));
        }
        return detections;
    }

    std::vector<Detection> YoloDetector::decodeWithNms(const inference::Tensor& output) const
    {
        const auto channels = static_cast<size_t>(output.shape[1]);
        const auto anchors = static_cast<size_t>(output.shape[2]);
        const auto numClasses = std::min(static_cast<size_t>(_config.numClasses), channels - 4);

        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        std::vector<int> classIds;

        const float* data = output.data.data();
        for (size_t a = 0; a < anchors; ++a) {
            int bestClass = -1;
            float bestScore = _config.scoreThreshold;
            for (size_t c = 0; c < numClasses; ++c) {
                const float score = data[(4 + c) * anchors + a];
                if (score >= bestScore && keepClass(static_cast<int>(c))) {
                    bestScore = score;
                    bestClass = static_cast<int>(c);
                }
            }
            if (bestClass < 0) {
                continue;
            }

            const float cx = data[0 * anchors + a];
            const float cy = data[1 * anchors + a];
            const float w = data[2 * anchors + a];
            const float h = data[3 * anchors + a];
            boxes.emplace_back(cx - w / 2.0F, cy - h / 2.0F, w, h);
            scores.push_back(bestScore);
            classIds.push_back(bestClass);
        }

        // Per-class NMS: shift each class into its own region so boxes of different
        // classes never overlap (OpenCV 4.5 on Ubuntu 22.04 has no NMSBoxesBatched).
        constexpr double kClassOffset = 4096.0;
        std::vector<cv::Rect2d> shifted(boxes);
        for (size_t i = 0; i < shifted.size(); ++i) {
            shifted[i].x += classIds[i] * kClassOffset;
            shifted[i].y += classIds[i] * kClassOffset;
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxes(shifted, scores, _config.scoreThreshold, _config.nmsThreshold, keep,
                          1.0F, _config.maxDetections);

        std::vector<Detection> detections;
        detections.reserve(keep.size());
        for (const int index : keep) {
            const auto& box = boxes[static_cast<size_t>(index)];
            detections.push_back({cv::Rect2f(static_cast<float>(box.x), static_cast<float>(box.y),
                                             static_cast<float>(box.width), static_cast<float>(box.height)),
                                  scores[static_cast<size_t>(index)], classIds[static_cast<size_t>(index)]});
        }
        return detections;
    }

    bool YoloDetector::keepClass(int classId) const
    {
        return _config.classIds.empty() ||
               std::find(_config.classIds.begin(), _config.classIds.end(), classId) != _config.classIds.end();
    }
}
