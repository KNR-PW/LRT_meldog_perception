#pragma once

#include <memory>
#include <string>
#include <vector>

#include "meldog_perception/inference/inference_backend.hpp"

namespace Ort {
    struct Env;
    struct Session;
}

namespace meldog_perception::inference {

    // CPU backend for development machines without an NVIDIA GPU and for
    // checking TensorRT outputs against a reference.
    class OnnxRuntimeBackend : public InferenceBackend
    {

    public:

        explicit OnnxRuntimeBackend(const BackendConfig& config);
        ~OnnxRuntimeBackend() override;

        const std::vector<TensorInfo>& inputs() const override { return _inputs; }
        const std::vector<TensorInfo>& outputs() const override { return _outputs; }

        const std::vector<Tensor>& run(const Tensor& input) override;

        std::string name() const override { return "onnxruntime"; }

    private:

        std::unique_ptr<Ort::Env> _env;
        std::unique_ptr<Ort::Session> _session;

        std::vector<TensorInfo> _inputs;
        std::vector<TensorInfo> _outputs;
        std::vector<Tensor> _results;
    };
}
