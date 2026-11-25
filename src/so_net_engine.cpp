#include <meldog_perception/so_net_engine.hpp>
#include <meldog_perception/trt_logger.hpp>

#include <stdexcept>
#include <fstream>
#include <algorithm>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace meldog_perception {

    SoNetEngine::SoNetEngine(const std::string& networkPath)
    {
        if (networkPath.empty()) {
            throw std::runtime_error("[SoNetEngine] Invalid network path");
        }

        _nvLogger = std::make_unique<TrtLogger>();

        initializeNetwork(networkPath);
        validateTensors();
        initializeGpuBuffers();
    }

    SoNetEngine::~SoNetEngine() = default;

    std::vector<std::vector<float>> SoNetEngine::infer(const std::vector<std::vector<float>>& data)
    {
        if (std::any_of(data.begin(), data.end(), [this](const std::vector<float>& x) { return x.size() != _sampleSize; })) {
            throw std::runtime_error("[SoNetEngine] Every input sample must have exactly " + std::to_string(_sampleSize) + " values");
        }

        const size_t batchSize = data.size();

        if (batchSize < 1 || batchSize > _maxBatch) {
            throw std::runtime_error("[SoNetEngine] Batch size must be in range 1.." + std::to_string(_maxBatch) + 
                ", got " + std::to_string(batchSize));
        }

        auto dims = _nvCudaEngine->getTensorShape(_inputName);
        dims.d[0] = static_cast<int>(batchSize); 

        if (!_nvExecutionContext->setInputShape(_inputName, dims)) {
            throw std::runtime_error("[SoNetEngine] Couldn't set input shape for batch size " + std::to_string(batchSize));
        }

        std::vector<float> flatInput(batchSize * _sampleSize);
        for (size_t i = 0; i < batchSize; ++i) {
            std::copy(data[i].begin(), data[i].end(), flatInput.begin() + (i * _sampleSize));
        }

        checkCuda(cudaMemcpyAsync(_inputBuffer.get(), flatInput.data(), batchSize * _sampleSize * sizeof(float), 
            cudaMemcpyHostToDevice, _cudaStream.get()), "cudaMemcpyAsync(hostToDevice)");

        if (!_nvExecutionContext->enqueueV3(_cudaStream.get())) {
            throw std::runtime_error("[SoNetEngine] Couldn't enqueue inference (enqueueV3 failed)");
        }

        std::vector<float> modelResult(batchSize * _embeddingSize);

        checkCuda(cudaMemcpyAsync(modelResult.data(), _outputBuffer.get(), batchSize * _embeddingSize * sizeof(float),
            cudaMemcpyDeviceToHost, _cudaStream.get()), "cudaMemcpyAsync(deviceToHost)");

        checkCuda(cudaStreamSynchronize(_cudaStream.get()), "cudaStreamSynchronize");

        std::vector<std::vector<float>> result;
        result.reserve(batchSize);

        for (size_t i = 0; i < batchSize; ++i) {
            auto start = modelResult.begin() + (i * _embeddingSize);
            result.emplace_back(start, start + _embeddingSize);
        }

        return result;
    }

    void SoNetEngine::initializeNetwork(const std::string& networkPath)
    {
        std::ifstream file(networkPath, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("[SoNetEngine] Couldn't open engine file: " + networkPath);
        }

        const std::streamoff size = file.tellg();
        if (size <= 0) {
            throw std::runtime_error("[SoNetEngine] Engine file is empty: " + networkPath);
        }

        std::vector<char> data(static_cast<std::size_t>(size));
        file.seekg(0, std::ios::beg);

        if (!file.read(data.data(), static_cast<std::streamsize>(size))) {
            throw std::runtime_error("[SoNetEngine] Couldn't read engine file: " + networkPath);
        }

        _nvRuntime.reset(nvinfer1::createInferRuntime(*_nvLogger));
        if (!_nvRuntime) {
            throw std::runtime_error("[SoNetEngine] Couldn't create TensorRT runtime");
        }

        _nvCudaEngine.reset(_nvRuntime->deserializeCudaEngine(data.data(), data.size()));
        if (!_nvCudaEngine) {
            throw std::runtime_error("[SoNetEngine] Couldn't deserialize engine: " + networkPath);
        }

        _nvExecutionContext.reset(_nvCudaEngine->createExecutionContext());
        if (!_nvExecutionContext) {
            throw std::runtime_error("[SoNetEngine] Couldn't create execution context");
        }
    }

    void SoNetEngine::validateTensors() const
    {
        const int32_t tensorCount = _nvCudaEngine->getNbIOTensors();
        if (tensorCount != 2) {
            throw std::runtime_error("[SoNetEngine] Expected 1 input and 1 output tensor, got " + std::to_string(tensorCount) + " tensors");
        }

        const char* inputName = nullptr;
        const char* outputName = nullptr;

        for (int32_t i = 0; i < tensorCount; ++i) {
            const char* name = _nvCudaEngine->getIOTensorName(i);

            if (_nvCudaEngine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error(std::string("[SoNetEngine] Tensor ") + name + " is not float32");
            }

            if (_nvCudaEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                inputName = name;
            } else {
                outputName = name;
            }
        }

        if (inputName == nullptr || outputName == nullptr) {
            throw std::runtime_error("[SoNetEngine] Engine needs one input and one output tensor");
        }

        const nvinfer1::Dims inputDims = _nvCudaEngine->getTensorShape(inputName);
        if (inputDims.nbDims != 4 || inputDims.d[0] == 0 || inputDims.d[0] < -1 || 
            inputDims.d[1] <= 0 || inputDims.d[2] <= 0 || inputDims.d[3] <= 0) {
            throw std::runtime_error("[SoNetEngine] Invalid input shape, expected [batch, C, H, W] with C, H, W > 0");
        }

        if (inputDims.d[0] == -1) {
            const nvinfer1::Dims minDims = _nvCudaEngine->getProfileShape(inputName, 0, nvinfer1::OptProfileSelector::kMIN);
            const nvinfer1::Dims maxDims = _nvCudaEngine->getProfileShape(inputName, 0, nvinfer1::OptProfileSelector::kMAX);

            if (minDims.nbDims != 4 || maxDims.nbDims != 4 || minDims.d[0] <= 0 || maxDims.d[0] < minDims.d[0]) {
                throw std::runtime_error("[SoNetEngine] Invalid batch range in the engine optimization profile");
            }
        }

        const nvinfer1::Dims outputDims = _nvCudaEngine->getTensorShape(outputName);
        if (outputDims.nbDims != 2 || outputDims.d[1] <= 0) {
            throw std::runtime_error("[SoNetEngine] Invalid output shape, expected [batch, features] with features > 0");
        }
    }

    void SoNetEngine::checkCuda(cudaError_t status, const char* what)
    {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("[SoNetEngine] ") + what + " failed: " + cudaGetErrorString(status));
        }
    }

    void SoNetEngine::initializeGpuBuffers()
    {
        const char* outputName = nullptr;

        for (int32_t i = 0; i < _nvCudaEngine->getNbIOTensors(); ++i) {
            const char* name = _nvCudaEngine->getIOTensorName(i);

            if (_nvCudaEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                _inputName = name;
            } else {
                outputName = name;
            }
        }

        const nvinfer1::Dims inputDims = _nvCudaEngine->getTensorShape(_inputName);
        const nvinfer1::Dims outputDims = _nvCudaEngine->getTensorShape(outputName);

        if (inputDims.d[0] == -1) {
            _maxBatch = static_cast<std::size_t>(_nvCudaEngine->getProfileShape(_inputName, 0, nvinfer1::OptProfileSelector::kMAX).d[0]);
        }
        else {
            _maxBatch = static_cast<std::size_t>(inputDims.d[0]);
        }

        _sampleSize = static_cast<std::size_t>(inputDims.d[1] * inputDims.d[2] * inputDims.d[3]);  
        _embeddingSize = static_cast<std::size_t>(outputDims.d[1]);                               

        const std::size_t inputBytes = _maxBatch * _sampleSize * sizeof(float);       
        const std::size_t outputBytes = _maxBatch * _embeddingSize * sizeof(float);   

        cudaStream_t stream = nullptr;
        checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        _cudaStream.reset(stream);

        void* buffer = nullptr;
        checkCuda(cudaMalloc(&buffer, inputBytes), "cudaMalloc(input)");
        _inputBuffer.reset(buffer);

        buffer = nullptr;
        checkCuda(cudaMalloc(&buffer, outputBytes), "cudaMalloc(output)");
        _outputBuffer.reset(buffer);

        if (!_nvExecutionContext->setTensorAddress(_inputName, _inputBuffer.get()) ||
            !_nvExecutionContext->setTensorAddress(outputName, _outputBuffer.get())) {
            throw std::runtime_error("[SoNetEngine] Couldn't bind tensor addresses");
        }
    }
}