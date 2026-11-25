#pragma once

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "meldog_perception/inference/inference_backend.hpp"

namespace meldog_perception {

    struct Detection
    {
        cv::Rect2f box;     // pixels of the original image
        float score = 0.0F;
        int classId = -1;
    };

    struct DetectorConfig
    {
        float scoreThreshold = 0.4F;
        float nmsThreshold = 0.45F;     // only used for models with an NMS-style head
        int numClasses = 80;            // only used for models with an NMS-style head
        std::vector<int> classIds{0};   // COCO 0 = person; empty keeps every class
        int maxDetections = 100;
    };

    struct DetectorTiming
    {
        double preprocessMs = 0.0;
        double inferenceMs = 0.0;
        double postprocessMs = 0.0;
    };

    // Ultralytics YOLO detector (YOLO26 end-to-end head; YOLOv8/11 NMS head also
    // works). Expects an RGB image; letterboxes it when its size differs from
    // the network input.
    class YoloDetector
    {

    public:

        YoloDetector(std::unique_ptr<inference::InferenceBackend> backend, DetectorConfig config);

        std::vector<Detection> detect(const cv::Mat& rgbImage);

        const DetectorTiming& lastTiming() const { return _timing; }
        const inference::InferenceBackend& backend() const { return *_backend; }
        cv::Size inputSize() const { return _inputSize; }

    private:

        struct Letterbox
        {
            float scale = 1.0F;
            float padX = 0.0F;
            float padY = 0.0F;
        };

        Letterbox preprocess(const cv::Mat& rgbImage);

        std::vector<Detection> decodeEndToEnd(const inference::Tensor& output) const;
        std::vector<Detection> decodeWithNms(const inference::Tensor& output) const;

        bool keepClass(int classId) const;

    private:

        std::unique_ptr<inference::InferenceBackend> _backend;
        DetectorConfig _config;

        cv::Size _inputSize;
        size_t _detectionOutput = 0;    // index of the [1, N, C] output tensor

        inference::Tensor _inputTensor;
        cv::Mat _letterboxed;
        DetectorTiming _timing;
    };
}
