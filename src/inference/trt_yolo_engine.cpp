#include <fstream>
#include <stdexcept>
#include <string>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "meldog_perception/inference/trt_logger.hpp"
#include "meldog_perception/inference/trt_yolo_engine.hpp"

namespace meldog_perception::inference {

    TrtYoloEngine::TrtYoloEngine(const std::string& networkPath)
    {
        if (networkPath.empty()) {
            throw std::runtime_error("[TrtYoloEngine] Invalid network path");
        }

        _nvLogger = std::make_unique<TrtLogger>();

        initializeNetwork(networkPath);
        validateTensors();
        initializeGpuBuffers();
    }

    TrtYoloEngine::~TrtYoloEngine() = default;

    std::vector<float> TrtYoloEngine::infer(const std::vector<float>& data)
    {
        if (data.size() != _sampleSize) {
            throw std::runtime_error("[TrtYoloEngine] Input must have exactly " + std::to_string(_sampleSize) +
                " values, got " + std::to_string(data.size()));
        }

        checkCuda(cudaMemcpyAsync(_inputBuffer.get(), data.data(), _sampleSize * sizeof(float),
            cudaMemcpyHostToDevice, _cudaStream.get()), "cudaMemcpyAsync(hostToDevice)");

        if (!_nvExecutionContext->enqueueV3(_cudaStream.get())) {
            throw std::runtime_error("[TrtYoloEngine] Couldn't enqueue inference (enqueueV3 failed)");
        }

        std::vector<float> result(_detectionCount * _detectionSize);

        checkCuda(cudaMemcpyAsync(result.data(), _outputBuffer.get(), result.size() * sizeof(float),
            cudaMemcpyDeviceToHost, _cudaStream.get()), "cudaMemcpyAsync(deviceToHost)");

        checkCuda(cudaStreamSynchronize(_cudaStream.get()), "cudaStreamSynchronize");

        return result;
    }

    void TrtYoloEngine::initializeNetwork(const std::string& networkPath)
    {
        std::ifstream file(networkPath, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("[TrtYoloEngine] Couldn't open engine file: " + networkPath);
        }

        const std::streamoff size = file.tellg();
        if (size <= 0) {
            throw std::runtime_error("[TrtYoloEngine] Engine file is empty: " + networkPath);
        }

        std::vector<char> data(static_cast<std::size_t>(size));
        file.seekg(0, std::ios::beg);

        if (!file.read(data.data(), static_cast<std::streamsize>(size))) {
            throw std::runtime_error("[TrtYoloEngine] Couldn't read engine file: " + networkPath);
        }

        _nvRuntime.reset(nvinfer1::createInferRuntime(*_nvLogger));
        if (!_nvRuntime) {
            throw std::runtime_error("[TrtYoloEngine] Couldn't create TensorRT runtime");
        }

        _nvCudaEngine.reset(_nvRuntime->deserializeCudaEngine(data.data(), data.size()));
        if (!_nvCudaEngine) {
            throw std::runtime_error("[TrtYoloEngine] Couldn't deserialize engine (built with a different TensorRT "
                                     "version or GPU, or exported by Ultralytics?): " + networkPath);
        }

        _nvExecutionContext.reset(_nvCudaEngine->createExecutionContext());
        if (!_nvExecutionContext) {
            throw std::runtime_error("[TrtYoloEngine] Couldn't create execution context");
        }
    }

    void TrtYoloEngine::validateTensors()
    {
        const int32_t tensorCount = _nvCudaEngine->getNbIOTensors();
        if (tensorCount != 2) {
            throw std::runtime_error("[TrtYoloEngine] Expected 1 input and 1 output tensor, got " +
                std::to_string(tensorCount) + " tensors");
        }

        for (int32_t i = 0; i < tensorCount; ++i) {
            const char* name = _nvCudaEngine->getIOTensorName(i);

            if (_nvCudaEngine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error(std::string("[TrtYoloEngine] Tensor ") + name + " is not float32");
            }

            if (_nvCudaEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                _inputName = name;
            } else {
                _outputName = name;
            }
        }

        if (_inputName == nullptr || _outputName == nullptr) {
            throw std::runtime_error("[TrtYoloEngine] Engine needs one input and one output tensor");
        }

        nvinfer1::Dims inputDims = _nvCudaEngine->getTensorShape(_inputName);
        if (inputDims.nbDims != 4 || (inputDims.d[0] != 1 && inputDims.d[0] != -1) ||
            inputDims.d[1] != 3 || inputDims.d[2] <= 0 || inputDims.d[3] <= 0) {
            throw std::runtime_error("[TrtYoloEngine] Invalid input shape, expected [1, 3, H, W]");
        }

        // A dynamic batch is pinned to 1, the detector sends one image at a time.
        if (inputDims.d[0] == -1) {
            inputDims.d[0] = 1;
            if (!_nvExecutionContext->setInputShape(_inputName, inputDims)) {
                throw std::runtime_error("[TrtYoloEngine] Couldn't set input batch size to 1");
            }
        }

        const nvinfer1::Dims outputDims = _nvExecutionContext->getTensorShape(_outputName);
        if (outputDims.nbDims != 3 || outputDims.d[1] <= 0 || outputDims.d[2] < 6 || outputDims.d[1] < outputDims.d[2]) {
            throw std::runtime_error("[TrtYoloEngine] Invalid output shape, expected [1, N, 6] "
                                     "(YOLO26 exported with nms=False, see scripts/export_yolo.py)");
        }

        _inputHeight = static_cast<std::size_t>(inputDims.d[2]);
        _inputWidth = static_cast<std::size_t>(inputDims.d[3]);
        _sampleSize = 3 * _inputHeight * _inputWidth;
        _detectionCount = static_cast<std::size_t>(outputDims.d[1]);
        _detectionSize = static_cast<std::size_t>(outputDims.d[2]);
    }

    void TrtYoloEngine::checkCuda(cudaError_t status, const char* what)
    {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("[TrtYoloEngine] ") + what + " failed: " + cudaGetErrorString(status));
        }
    }

    void TrtYoloEngine::initializeGpuBuffers()
    {
        cudaStream_t stream = nullptr;
        checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        _cudaStream.reset(stream);

        void* buffer = nullptr;
        checkCuda(cudaMalloc(&buffer, _sampleSize * sizeof(float)), "cudaMalloc(input)");
        _inputBuffer.reset(buffer);

        buffer = nullptr;
        checkCuda(cudaMalloc(&buffer, _detectionCount * _detectionSize * sizeof(float)), "cudaMalloc(output)");
        _outputBuffer.reset(buffer);

        if (!_nvExecutionContext->setTensorAddress(_inputName, _inputBuffer.get()) ||
            !_nvExecutionContext->setTensorAddress(_outputName, _outputBuffer.get())) {
            throw std::runtime_error("[TrtYoloEngine] Couldn't bind tensor addresses");
        }
    }

    std::pair<std::size_t, std::size_t> TrtYoloEngine::getInputDim() const
    {
        return std::pair<std::size_t, std::size_t>(_inputHeight, _inputWidth);
    }
}
