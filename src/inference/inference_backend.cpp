#include "meldog_perception/inference/inference_backend.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <numeric>
#include <stdexcept>

#ifdef MELDOG_HAS_TENSORRT
#include "meldog_perception/inference/tensorrt_backend.hpp"
#endif
#ifdef MELDOG_HAS_ONNXRUNTIME
#include "meldog_perception/inference/onnxruntime_backend.hpp"
#endif

namespace meldog_perception::inference {

    size_t Tensor::elementCount() const
    {
        return std::accumulate(shape.begin(), shape.end(), size_t{1},
            [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
    }

    BackendType backendTypeFromString(const std::string& name)
    {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (lower.empty() || lower == "auto") {
            return BackendType::Auto;
        }
        if (lower == "tensorrt" || lower == "trt") {
            return BackendType::TensorRt;
        }
        if (lower == "onnxruntime" || lower == "ort") {
            return BackendType::OnnxRuntime;
        }
        throw std::invalid_argument("Unknown inference backend '" + name +
                                    "' (expected auto, tensorrt or onnxruntime)");
    }

    std::string toString(BackendType type)
    {
        switch (type) {
            case BackendType::Auto: return "auto";
            case BackendType::TensorRt: return "tensorrt";
            case BackendType::OnnxRuntime: return "onnxruntime";
        }
        return "unknown";
    }

    std::vector<std::string> availableBackends()
    {
        std::vector<std::string> names;
#ifdef MELDOG_HAS_TENSORRT
        names.emplace_back("tensorrt");
#endif
#ifdef MELDOG_HAS_ONNXRUNTIME
        names.emplace_back("onnxruntime");
#endif
        return names;
    }

    std::unique_ptr<InferenceBackend> createBackend(const BackendConfig& config)
    {
        BackendType type = config.type;

        if (type == BackendType::Auto) {
#if defined(MELDOG_HAS_TENSORRT)
            type = BackendType::TensorRt;
#elif defined(MELDOG_HAS_ONNXRUNTIME)
            type = BackendType::OnnxRuntime;
#else
            throw std::runtime_error("meldog_perception was built without any inference backend");
#endif
        }

        switch (type) {
            case BackendType::TensorRt:
#ifdef MELDOG_HAS_TENSORRT
                return std::make_unique<TensorRtBackend>(config);
#else
                throw std::runtime_error("TensorRT backend requested, but this build has no TensorRT "
                                         "(available: onnxruntime)");
#endif
            case BackendType::OnnxRuntime:
#ifdef MELDOG_HAS_ONNXRUNTIME
                return std::make_unique<OnnxRuntimeBackend>(config);
#else
                throw std::runtime_error("ONNX Runtime backend requested, but this build has no ONNX Runtime");
#endif
            case BackendType::Auto:
                break;
        }
        throw std::logic_error("unreachable backend type");
    }
}
