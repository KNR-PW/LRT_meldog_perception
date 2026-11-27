#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace meldog_perception::inference {

    // Dense float32 tensor on the host (row-major).
    struct Tensor
    {
        std::vector<int64_t> shape;
        std::vector<float> data;

        size_t elementCount() const;
    };

    // Static description of a network input/output; dynamic dimensions are -1.
    struct TensorInfo
    {
        std::string name;
        std::vector<int64_t> shape;
    };

    enum class BackendType
    {
        Auto,         // TensorRT when compiled in, otherwise ONNX Runtime
        TensorRt,
        OnnxRuntime,
    };

    BackendType backendTypeFromString(const std::string& name);
    std::string toString(BackendType type);

    struct BackendConfig
    {
        BackendType type = BackendType::Auto;

        // Source model. TensorRT builds (and caches) an engine from it.
        std::string onnxPath;

        // TensorRT only
        std::string engineCacheDir;     // empty -> directory of onnxPath
        bool fp16 = true;
        int maxBatch = 1;               // upper bound for a dynamic batch dimension
        size_t workspaceBytes = 1ULL << 30;

        // ONNX Runtime only
        int cpuThreads = 0;             // 0 -> ONNX Runtime default
    };

    // Single-input network runner shared by the detector and the ReID encoder.
    class InferenceBackend
    {

    public:

        virtual ~InferenceBackend() = default;

        virtual const std::vector<TensorInfo>& inputs() const = 0;
        virtual const std::vector<TensorInfo>& outputs() const = 0;

        // Runs the network on one NCHW float32 input. Returned tensors stay valid
        // until the next call.
        virtual const std::vector<Tensor>& run(const Tensor& input) = 0;

        virtual std::string name() const = 0;
    };

    // Throws std::runtime_error when the requested backend is not compiled in
    // or the model cannot be loaded.
    std::unique_ptr<InferenceBackend> createBackend(const BackendConfig& config);

    // Backends compiled into this build, e.g. {"tensorrt", "onnxruntime"}.
    std::vector<std::string> availableBackends();
}
