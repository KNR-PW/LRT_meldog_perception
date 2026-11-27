#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace meldog_perception::inference {

    // Raw YOLO26 inference (end-to-end head, exported with nms=False).
    // Input: one RGB image as CHW float32 in [0, 1], 3 * H * W values.
    // Output: getDetectionCount() rows of getDetectionSize() values,
    //         each row x1, y1, x2, y2, score, class in input-image pixels.
    class IYoloEngine
    {

    public:

        virtual ~IYoloEngine() = default;

        virtual std::vector<float> infer(const std::vector<float>& data) = 0;

        virtual std::pair<std::size_t, std::size_t> getInputDim() const = 0;   // (height, width)

        virtual std::size_t getDetectionCount() const = 0;

        virtual std::size_t getDetectionSize() const = 0;
    };

    // .engine -> TensorRT, .onnx -> ONNX Runtime
    std::unique_ptr<IYoloEngine> createYoloEngine(const std::string& networkPath);
}
