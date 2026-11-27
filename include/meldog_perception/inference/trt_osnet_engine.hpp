#pragma once 

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "meldog_perception/inference/i_osnet_engine.hpp"

namespace nvinfer1{
    class ILogger;
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace meldog_perception::inference {

    class TrtOsNetEngine final : public IOsNetEngine
    {

    public: 

        explicit  TrtOsNetEngine(const std::string& networkPath);
        
        ~TrtOsNetEngine() override;

        std::vector<std::vector<float>> infer(const std::vector<std::vector<float>>& data) override;

        std::pair<std::size_t, std::size_t> getInputDim() const override;
        std::size_t getMaxBatchSize() const override { return _maxBatch; }
        std::size_t getEmbeddingSize() const override { return _embeddingSize; }

    private:

        void initializeNetwork(const std::string& networkPath);

        void validateTensors() const;

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

        std::size_t _maxBatch = 0;
        const char* _inputName = nullptr;
        std::size_t _sampleSize = 0;
        std::size_t _embeddingSize = 0;
        std::size_t _inputHeight = 0;
        std::size_t _inputWidth = 0;
    };
}