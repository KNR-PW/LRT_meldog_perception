#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

#include "meldog_perception/inference/onnx_yolo_engine.hpp"

namespace meldog_perception::inference {

    OnnxYoloEngine::OnnxYoloEngine(const std::string& networkPath)
    {
        if (networkPath.empty()) {
            throw std::runtime_error("[OnnxYoloEngine] Invalid network path");
        }

        initializeNetwork(networkPath);
        validateTensors();
    }

    OnnxYoloEngine::~OnnxYoloEngine() = default;

    std::vector<float> OnnxYoloEngine::infer(const std::vector<float>& data)
    {
        if (data.size() != _sampleSize) {
            throw std::runtime_error("[OnnxYoloEngine] Input must have exactly " + std::to_string(_sampleSize) +
                " values, got " + std::to_string(data.size()));
        }

        const std::array<int64_t, 4> inputShape = {
            1, 3, static_cast<int64_t>(_inputHeight), static_cast<int64_t>(_inputWidth)
        };

        const Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // ONNX Runtime only reads the input, the const_cast avoids copying the image.
        const Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(data.data()),
            data.size(), inputShape.data(), inputShape.size());

        const char* inputNames[] = {_inputName.c_str()};
        const char* outputNames[] = {_outputName.c_str()};

        std::vector<Ort::Value> outputs;
        try {
            outputs = _ortSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("[OnnxYoloEngine] Inference failed: ") + e.what());
        }

        if (outputs.size() != 1 || !outputs[0].IsTensor()) {
            throw std::runtime_error("[OnnxYoloEngine] Expected a single output tensor");
        }

        const std::size_t outputSize = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (outputSize != _detectionCount * _detectionSize) {
            throw std::runtime_error("[OnnxYoloEngine] Unexpected output size " + std::to_string(outputSize));
        }

        const float* modelResult = outputs[0].GetTensorData<float>();
        return std::vector<float>(modelResult, modelResult + outputSize);
    }

    void OnnxYoloEngine::initializeNetwork(const std::string& networkPath)
    {
        _ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "meldog_perception");

        Ort::SessionOptions sessionOpt;
        sessionOpt.SetIntraOpNumThreads(0);
        sessionOpt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        _ortSession = std::make_unique<Ort::Session>(*_ortEnv, networkPath.c_str(), sessionOpt);
    }

    void OnnxYoloEngine::validateTensors()
    {
        const std::size_t inputCount = _ortSession->GetInputCount();
        const std::size_t outputCount = _ortSession->GetOutputCount();

        if (inputCount != 1 || outputCount != 1) {
            throw std::runtime_error("[OnnxYoloEngine] Expected 1 input and 1 output tensor, got " +
                std::to_string(inputCount) + " inputs and " + std::to_string(outputCount) + " outputs");
        }

        const Ort::TypeInfo inputTypeInfo = _ortSession->GetInputTypeInfo(0);
        const Ort::TypeInfo outputTypeInfo = _ortSession->GetOutputTypeInfo(0);

        if (inputTypeInfo.GetONNXType() != ONNX_TYPE_TENSOR || outputTypeInfo.GetONNXType() != ONNX_TYPE_TENSOR) {
            throw std::runtime_error("[OnnxYoloEngine] Model input and output must be tensors");
        }

        const auto inputInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        const auto outputInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

        if (inputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("[OnnxYoloEngine] Model input and output must be float32");
        }

        const std::vector<int64_t> inputShape = inputInfo.GetShape();
        if (inputShape.size() != 4 || (inputShape[0] != 1 && inputShape[0] != -1) ||
            inputShape[1] != 3 || inputShape[2] <= 0 || inputShape[3] <= 0) {
            throw std::runtime_error("[OnnxYoloEngine] Invalid input shape, expected [1, 3, H, W] "
                                     "(export with a fixed imgsz)");
        }

        const std::vector<int64_t> outputShape = outputInfo.GetShape();
        if (outputShape.size() != 3 || outputShape[1] <= 0 || outputShape[2] < 6 || outputShape[1] < outputShape[2]) {
            throw std::runtime_error("[OnnxYoloEngine] Invalid output shape, expected [1, N, 6] "
                                     "(YOLO26 exported with nms=False, see scripts/export_yolo.py)");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        _inputName = _ortSession->GetInputNameAllocated(0, allocator).get();
        _outputName = _ortSession->GetOutputNameAllocated(0, allocator).get();

        _inputHeight = static_cast<std::size_t>(inputShape[2]);
        _inputWidth = static_cast<std::size_t>(inputShape[3]);
        _sampleSize = 3 * _inputHeight * _inputWidth;
        _detectionCount = static_cast<std::size_t>(outputShape[1]);
        _detectionSize = static_cast<std::size_t>(outputShape[2]);
    }

    std::pair<std::size_t, std::size_t> OnnxYoloEngine::getInputDim() const
    {
        return std::pair<std::size_t, std::size_t>(_inputHeight, _inputWidth);
    }
}
