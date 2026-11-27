#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

#include "meldog_perception/inference/onnx_osnet_engine.hpp"

namespace meldog_perception::inference {

    OnnxOsNetEngine::OnnxOsNetEngine(const std::string& networkPath)
    {
        if (networkPath.empty()) {
            throw std::runtime_error("[OnnxOsNetEngine] Invalid network path");
        }

        initializeNetwork(networkPath);
        validateTensors();
    }

    OnnxOsNetEngine::~OnnxOsNetEngine() = default;

    std::vector<std::vector<float>> OnnxOsNetEngine::infer(const std::vector<std::vector<float>>& data)
    {
        if (std::any_of(data.begin(), data.end(), [this](const std::vector<float>& x) { return x.size() != _sampleSize; })) {
            throw std::runtime_error("[OnnxOsNetEngine] Every input sample must have exactly " + std::to_string(_sampleSize) + " values");
        }

        const std::size_t batchSize = data.size();

        if (batchSize == 0) {
            throw std::runtime_error("[OnnxOsNetEngine] Batch must contain at least one sample");
        }

        const bool dynamicBatch = (_maxBatch == std::numeric_limits<std::size_t>::max());
        if (!dynamicBatch && batchSize != _maxBatch) {
            throw std::runtime_error("[OnnxOsNetEngine] Model has a fixed batch of " + std::to_string(_maxBatch) + ", got " + std::to_string(batchSize) + " samples");
        }

        std::vector<float> flatInput(batchSize * _sampleSize);
        for (std::size_t i = 0; i < batchSize; ++i) {
            std::copy(data[i].begin(), data[i].end(), flatInput.begin() + (i * _sampleSize));
        }

        const std::size_t channels = _sampleSize / (_inputHeight * _inputWidth);
        const std::array<int64_t, 4> inputShape = {
            static_cast<int64_t>(batchSize),
            static_cast<int64_t>(channels),
            static_cast<int64_t>(_inputHeight),
            static_cast<int64_t>(_inputWidth)
        };

        const Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, flatInput.data(), flatInput.size(), inputShape.data(), inputShape.size());

        const char* inputNames[] = {_inputName.c_str()};
        const char* outputNames[] = {_outputName.c_str()};

        std::vector<Ort::Value> outputs;
        try {
            outputs = _ortSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("[OnnxOsNetEngine] Inference failed: ") + e.what());
        }

        if (outputs.size() != 1 || !outputs[0].IsTensor()) {
            throw std::runtime_error("[OnnxOsNetEngine] Expected a single output tensor");
        }

        const std::vector<int64_t> outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() != 2 ||
            outputShape[0] != static_cast<int64_t>(batchSize) ||
            outputShape[1] != static_cast<int64_t>(_embeddingSize)) {
            throw std::runtime_error("[OnnxOsNetEngine] Unexpected output shape, expected [" + std::to_string(batchSize) +
                ", " + std::to_string(_embeddingSize) + "]");
        }

        const float* modelResult = outputs[0].GetTensorData<float>();

        std::vector<std::vector<float>> result;
        result.reserve(batchSize);

        for (std::size_t i = 0; i < batchSize; ++i) {
            const float* start = modelResult + (i * _embeddingSize);
            result.emplace_back(start, start + _embeddingSize);
        }

        return result;
    }


    void OnnxOsNetEngine::initializeNetwork(const std::string& networkPath)
    {
        _ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "meldog_perception");

        Ort::SessionOptions sessionOpt;
        sessionOpt.SetIntraOpNumThreads(0);
        sessionOpt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
        _ortSession = std::make_unique<Ort::Session>(*_ortEnv, networkPath.c_str(), sessionOpt);
    }

    void OnnxOsNetEngine::validateTensors()
    {
        const std::size_t inputCount = _ortSession->GetInputCount();
        const std::size_t outputCount = _ortSession->GetOutputCount();

        if (inputCount != 1 || outputCount != 1) {
            throw std::runtime_error("[OnnxOsNetEngine] Expected 1 input and 1 output tensor, got " +
                std::to_string(inputCount) + " inputs and " + std::to_string(outputCount) + " outputs");
        }

        const Ort::TypeInfo inputTypeInfo = _ortSession->GetInputTypeInfo(0);
        const Ort::TypeInfo outputTypeInfo = _ortSession->GetOutputTypeInfo(0);

        if (inputTypeInfo.GetONNXType() != ONNX_TYPE_TENSOR || outputTypeInfo.GetONNXType() != ONNX_TYPE_TENSOR) {
            throw std::runtime_error("[OnnxOsNetEngine] Model input and output must be tensors");
        }

        const auto inputInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        const auto outputInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

        if (inputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("[OnnxOsNetEngine] Model input and output must be float32");
        }

        const std::vector<int64_t> inputShape = inputInfo.GetShape();
        if (inputShape.size() != 4 || inputShape[0] == 0 || inputShape[0] < -1 ||
            inputShape[1] <= 0 || inputShape[2] <= 0 || inputShape[3] <= 0) {
            throw std::runtime_error("[OnnxOsNetEngine] Invalid input shape, expected [batch, C, H, W] with C, H, W > 0");
        }

        const std::vector<int64_t> outputShape = outputInfo.GetShape();
        if (outputShape.size() != 2 || outputShape[1] <= 0) {
            throw std::runtime_error("[OnnxOsNetEngine] Invalid output shape, expected [batch, features] with features > 0");
        }

        if (outputShape[0] != inputShape[0]) {
            throw std::runtime_error("[OnnxOsNetEngine] Input and output batch dimensions differ (" +
                std::to_string(inputShape[0]) + " vs " + std::to_string(outputShape[0]) + ")");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        _inputName = _ortSession->GetInputNameAllocated(0, allocator).get();
        _outputName = _ortSession->GetOutputNameAllocated(0, allocator).get();

        _inputHeight = static_cast<std::size_t>(inputShape[2]);
        _inputWidth = static_cast<std::size_t>(inputShape[3]);
        _sampleSize = static_cast<std::size_t>(inputShape[1]) * _inputHeight * _inputWidth;
        _embeddingSize = static_cast<std::size_t>(outputShape[1]);
        // A dynamic batch has no upper bound in ONNX Runtime; the caller (ReidConfig::maxBatchSize) picks the chunk size
        _maxBatch = (inputShape[0] == -1) ? std::numeric_limits<std::size_t>::max() : static_cast<std::size_t>(inputShape[0]);
    }

    std::pair<std::size_t, std::size_t> OnnxOsNetEngine::getInputDim() const
    {
        return std::pair<std::size_t, std::size_t>(_inputHeight, _inputWidth);
    }
}