#include "meldog_perception/inference/onnxruntime_backend.hpp"

#include <stdexcept>

#include <onnxruntime_cxx_api.h>

namespace meldog_perception::inference {

    namespace {

        std::vector<int64_t> shapeOf(const Ort::TypeInfo& typeInfo)
        {
            return typeInfo.GetTensorTypeAndShapeInfo().GetShape();
        }
    }

    OnnxRuntimeBackend::OnnxRuntimeBackend(const BackendConfig& config)
    {
        if (config.onnxPath.empty()) {
            throw std::invalid_argument("ONNX Runtime backend needs an .onnx model path");
        }

        _env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "meldog_perception");

        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (config.cpuThreads > 0) {
            options.SetIntraOpNumThreads(config.cpuThreads);
        }

        _session = std::make_unique<Ort::Session>(*_env, config.onnxPath.c_str(), options);

        if (_session->GetInputCount() != 1) {
            throw std::runtime_error("Expected a single-input network, " + config.onnxPath + " has " +
                                     std::to_string(_session->GetInputCount()) + " inputs");
        }

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < _session->GetInputCount(); ++i) {
            const auto name = _session->GetInputNameAllocated(i, allocator);
            _inputs.push_back({name.get(), shapeOf(_session->GetInputTypeInfo(i))});
        }

        for (size_t i = 0; i < _session->GetOutputCount(); ++i) {
            const auto name = _session->GetOutputNameAllocated(i, allocator);
            const auto typeInfo = _session->GetOutputTypeInfo(i);
            if (typeInfo.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                throw std::runtime_error(std::string("Output '") + name.get() + "' is not float32");
            }
            _outputs.push_back({name.get(), shapeOf(typeInfo)});
        }

        _results.resize(_outputs.size());
    }

    OnnxRuntimeBackend::~OnnxRuntimeBackend() = default;

    const std::vector<Tensor>& OnnxRuntimeBackend::run(const Tensor& input)
    {
        const auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // ONNX Runtime does not modify the input buffer.
        auto inputValue = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(input.data.data()), input.data.size(),
            input.shape.data(), input.shape.size());

        std::vector<const char*> outputNames;
        outputNames.reserve(_outputs.size());
        for (const auto& output : _outputs) {
            outputNames.push_back(output.name.c_str());
        }
        const char* inputName = _inputs.front().name.c_str();

        auto outputValues = _session->Run(Ort::RunOptions{nullptr}, &inputName, &inputValue, 1,
                                          outputNames.data(), outputNames.size());

        for (size_t i = 0; i < outputValues.size(); ++i) {
            const auto info = outputValues[i].GetTensorTypeAndShapeInfo();
            const float* data = outputValues[i].GetTensorData<float>();

            _results[i].shape = info.GetShape();
            _results[i].data.assign(data, data + info.GetElementCount());
        }

        return _results;
    }
}
