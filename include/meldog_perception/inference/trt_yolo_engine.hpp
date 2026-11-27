#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "meldog_perception/inference/i_yolo_engine.hpp"

namespace nvinfer1 {
    class ILogger;
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace meldog_perception::inference {

    // Loads a prebuilt .engine (scripts/build_engine.sh on the target device).
    class TrtYoloEngine final : public IYoloEngine
    {

    public:

        explicit TrtYoloEngine(const std::string& networkPath);

        ~TrtYoloEngine() override;

        std::vector<float> infer(const std::vector<float>& data) override;

        std::pair<std::size_t, std::size_t> getInputDim() const override;
        std::size_t getDetectionCount() const override { return _detectionCount; }
        std::size_t getDetectionSize() const override { return _detectionSize; }

    private:

        void initializeNetwork(const std::string& networkPath);

        void validateTensors();

        void initializeGpuBuffers();

        struct CudaBufferDeleter { void operator()(void* buffer) const noexcept { cudaFree(buffer); } };
        struct CudaStreamDeleter { void operator()(CUstream_st* stream) const noexcept { cudaStreamDestroy(stream); } };

        static void checkCuda(cudaError_t status, const char* what);

    private:

        std::unique_ptr<nvinfer1::ILogger> _nvLogger;
        std::unique_ptr<nvinfer1::IRuntime> _nvRuntime;
        std::unique_ptr<nvinfer1::ICudaEngine> _nvCudaEngine;
        std::unique_ptr<nvinfer1::IExecutionContext> _nvExecutionContext;

        std::unique_ptr<CUstream_st, CudaStreamDeleter> _cudaStream;
        std::unique_ptr<void, CudaBufferDeleter> _inputBuffer;
        std::unique_ptr<void, CudaBufferDeleter> _outputBuffer;

        const char* _inputName = nullptr;
        const char* _outputName = nullptr;

        std::size_t _inputHeight = 0;
        std::size_t _inputWidth = 0;
        std::size_t _sampleSize = 0;
        std::size_t _detectionCount = 0;
        std::size_t _detectionSize = 0;
    };
}
