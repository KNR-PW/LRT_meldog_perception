#pragma once

#include <memory>
#include <string>
#include <vector>

#include "meldog_perception/inference/inference_backend.hpp"

namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace meldog_perception::inference {

    // TensorRT backend (Jetson / x86 NVIDIA). Loads a cached engine built for
    // this GPU and TensorRT version, or builds one from the ONNX model.
    class TensorRtBackend : public InferenceBackend
    {

    public:

        explicit TensorRtBackend(const BackendConfig& config);
        ~TensorRtBackend() override;

        TensorRtBackend(const TensorRtBackend&) = delete;
        TensorRtBackend& operator=(const TensorRtBackend&) = delete;

        const std::vector<TensorInfo>& inputs() const override { return _inputs; }
        const std::vector<TensorInfo>& outputs() const override { return _outputs; }

        const std::vector<Tensor>& run(const Tensor& input) override;

        std::string name() const override { return "tensorrt"; }

        const std::string& enginePath() const { return _enginePath; }

    private:

        std::string buildEngine(const BackendConfig& config) const;
        void loadEngine(const std::string& serialized);
        void allocateBuffers();

    private:

        struct Binding
        {
            std::string name;
            void* device = nullptr;
            size_t capacityBytes = 0;
        };

        std::string _enginePath;

        std::unique_ptr<nvinfer1::IRuntime> _runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> _engine;
        std::unique_ptr<nvinfer1::IExecutionContext> _context;
        void* _stream = nullptr;   // cudaStream_t

        Binding _inputBinding;
        std::vector<Binding> _outputBindings;

        std::vector<TensorInfo> _inputs;
        std::vector<TensorInfo> _outputs;
        std::vector<Tensor> _results;
    };
}
