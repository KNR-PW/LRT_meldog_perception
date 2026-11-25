#include "meldog_perception/inference/tensorrt_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

namespace fs = std::filesystem;

namespace meldog_perception::inference {

    namespace {

        class TrtLogger : public nvinfer1::ILogger
        {

        public:

            void log(Severity severity, const char* msg) noexcept override
            {
                if (severity <= Severity::kWARNING) {
                    std::cerr << "[TensorRT] " << msg << std::endl;
                }
            }
        };

        TrtLogger& logger()
        {
            static TrtLogger instance;
            return instance;
        }

        void checkCuda(cudaError_t status, const char* what)
        {
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
            }
        }

        std::vector<int64_t> toVector(const nvinfer1::Dims& dims)
        {
            return std::vector<int64_t>(dims.d, dims.d + dims.nbDims);
        }

        nvinfer1::Dims toDims(const std::vector<int64_t>& shape)
        {
            // Dims holds int32_t (TensorRT 8.6) or int64_t (TensorRT 10) values.
            using DimType = std::remove_reference_t<decltype(nvinfer1::Dims{}.d[0])>;

            nvinfer1::Dims dims{};
            if (shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS)) {
                throw std::runtime_error("Tensor rank exceeds TensorRT's limit");
            }
            dims.nbDims = static_cast<int32_t>(shape.size());
            for (size_t i = 0; i < shape.size(); ++i) {
                dims.d[i] = static_cast<DimType>(shape[i]);
            }
            return dims;
        }

        size_t volume(const nvinfer1::Dims& dims)
        {
            size_t count = 1;
            for (int32_t i = 0; i < dims.nbDims; ++i) {
                if (dims.d[i] < 0) {
                    throw std::runtime_error("Cannot size a buffer with an unresolved dynamic dimension");
                }
                count *= static_cast<size_t>(dims.d[i]);
            }
            return count;
        }

        std::string gpuTag()
        {
            int device = 0;
            checkCuda(cudaGetDevice(&device), "cudaGetDevice");
            cudaDeviceProp props{};
            checkCuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");

            std::string tag = props.name;
            std::transform(tag.begin(), tag.end(), tag.begin(), [](unsigned char c) {
                return std::isalnum(c) ? static_cast<char>(std::tolower(c)) : '_';
            });
            return tag + "_sm" + std::to_string(props.major) + std::to_string(props.minor);
        }

        // Engines only load on the GPU and TensorRT version they were built with,
        // so both go into the cache file name.
        std::string enginePathFor(const BackendConfig& config)
        {
            const fs::path onnx(config.onnxPath);
            const fs::path dir = config.engineCacheDir.empty() ? onnx.parent_path()
                                                               : fs::path(config.engineCacheDir);
            std::ostringstream name;
            name << onnx.stem().string() << '.' << gpuTag()
                 << ".trt" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH
                 << (config.fp16 ? ".fp16" : ".fp32")
                 << ".b" << config.maxBatch << ".engine";
            return (dir / name.str()).string();
        }

        std::string readFile(const std::string& path)
        {
            std::ifstream file(path, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        }
    }

    TensorRtBackend::TensorRtBackend(const BackendConfig& config)
    {
        if (config.onnxPath.empty()) {
            throw std::invalid_argument("TensorRT backend needs an .onnx model path");
        }
        if (config.maxBatch < 1) {
            throw std::invalid_argument("maxBatch must be >= 1");
        }

        _enginePath = enginePathFor(config);

        if (fs::exists(_enginePath)) {
            try {
                loadEngine(readFile(_enginePath));
            } catch (const std::exception& e) {
                std::cerr << "[TensorRT] Cached engine " << _enginePath << " is unusable (" << e.what()
                          << "), rebuilding" << std::endl;
                _context.reset();
                _engine.reset();
            }
        }

        if (!_engine) {
            if (!fs::exists(config.onnxPath)) {
                throw std::runtime_error("ONNX model not found: " + config.onnxPath);
            }
            std::cerr << "[TensorRT] Building engine from " << config.onnxPath
                      << " (this takes a few minutes on Jetson)" << std::endl;
            loadEngine(buildEngine(config));
            std::cerr << "[TensorRT] Engine saved to " << _enginePath << std::endl;
        }

        cudaStream_t stream = nullptr;
        checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        _stream = stream;

        allocateBuffers();
    }

    TensorRtBackend::~TensorRtBackend()
    {
        _context.reset();
        _engine.reset();
        _runtime.reset();

        cudaFree(_inputBinding.device);
        for (auto& binding : _outputBindings) {
            cudaFree(binding.device);
        }
        if (_stream) {
            cudaStreamDestroy(static_cast<cudaStream_t>(_stream));
        }
    }

    std::string TensorRtBackend::buildEngine(const BackendConfig& config) const
    {
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
        if (!builder) {
            throw std::runtime_error("createInferBuilder failed");
        }

#if NV_TENSORRT_MAJOR >= 10
        const uint32_t flags = 0;   // explicit batch is the only mode
#else
        const uint32_t flags =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger()));

        if (!parser->parseFromFile(config.onnxPath.c_str(),
                                   static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::ostringstream errors;
            for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
                errors << "\n  " << parser->getError(i)->desc();
            }
            throw std::runtime_error("Failed to parse " + config.onnxPath + errors.str());
        }

        std::unique_ptr<nvinfer1::IBuilderConfig> builderConfig(builder->createBuilderConfig());
        builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.workspaceBytes);
        if (config.fp16) {
            builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        // A dynamic batch dimension (OSNet crops) gets a 1..maxBatch profile.
        nvinfer1::IOptimizationProfile* profile = nullptr;
        for (int32_t i = 0; i < network->getNbInputs(); ++i) {
            auto* input = network->getInput(i);
            const nvinfer1::Dims dims = input->getDimensions();

            bool dynamic = false;
            for (int32_t d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] < 0) {
                    if (d != 0) {
                        throw std::runtime_error(std::string("Input '") + input->getName() +
                                                 "' has a dynamic non-batch dimension; export it with a fixed size");
                    }
                    dynamic = true;
                }
            }
            if (!dynamic) {
                continue;
            }

            if (!profile) {
                profile = builder->createOptimizationProfile();
            }
            nvinfer1::Dims minDims = dims, maxDims = dims;
            minDims.d[0] = 1;
            maxDims.d[0] = config.maxBatch;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, maxDims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);
        }
        if (profile) {
            builderConfig->addOptimizationProfile(profile);
        }

        std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *builderConfig));
        if (!plan) {
            throw std::runtime_error("TensorRT engine build failed for " + config.onnxPath);
        }

        std::string serialized(static_cast<const char*>(plan->data()), plan->size());

        // Write next to the final path and rename, so a crash never leaves a broken cache.
        fs::create_directories(fs::path(_enginePath).parent_path());
        const std::string tmpPath = _enginePath + ".tmp";
        {
            std::ofstream file(tmpPath, std::ios::binary);
            file.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
            if (!file) {
                throw std::runtime_error("Cannot write engine to " + tmpPath);
            }
        }
        fs::rename(tmpPath, _enginePath);

        return serialized;
    }

    void TensorRtBackend::loadEngine(const std::string& serialized)
    {
        if (!_runtime) {
            _runtime.reset(nvinfer1::createInferRuntime(logger()));
        }
        _engine.reset(_runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
        if (!_engine) {
            throw std::runtime_error("deserializeCudaEngine failed");
        }
        _context.reset(_engine->createExecutionContext());
        if (!_context) {
            throw std::runtime_error("createExecutionContext failed");
        }
    }

    void TensorRtBackend::allocateBuffers()
    {
        _inputs.clear();
        _outputs.clear();
        _outputBindings.clear();

        std::vector<std::string> outputNames;

        for (int32_t i = 0; i < _engine->getNbIOTensors(); ++i) {
            const char* name = _engine->getIOTensorName(i);

            if (_engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error(std::string("Tensor '") + name + "' is not float32");
            }

            const TensorInfo info{name, toVector(_engine->getTensorShape(name))};

            if (_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                if (!_inputs.empty()) {
                    throw std::runtime_error("Expected a single-input network");
                }
                _inputs.push_back(info);
            } else {
                _outputs.push_back(info);
                outputNames.emplace_back(name);
            }
        }
        if (_inputs.empty()) {
            throw std::runtime_error("Engine has no inputs");
        }

        // Size every buffer for the largest input the profile allows.
        const std::string& inputName = _inputs.front().name;
        nvinfer1::Dims maxInput = _engine->getTensorShape(inputName.c_str());
        if (std::any_of(maxInput.d, maxInput.d + maxInput.nbDims, [](auto d) { return d < 0; })) {
            maxInput = _engine->getProfileShape(inputName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        }
        if (!_context->setInputShape(inputName.c_str(), maxInput)) {
            throw std::runtime_error("setInputShape failed for " + inputName);
        }

        _inputBinding.name = inputName;
        _inputBinding.capacityBytes = volume(maxInput) * sizeof(float);
        checkCuda(cudaMalloc(&_inputBinding.device, _inputBinding.capacityBytes), "cudaMalloc(input)");
        _context->setTensorAddress(inputName.c_str(), _inputBinding.device);

        for (const auto& name : outputNames) {
            Binding binding;
            binding.name = name;
            binding.capacityBytes = volume(_context->getTensorShape(name.c_str())) * sizeof(float);
            checkCuda(cudaMalloc(&binding.device, binding.capacityBytes), "cudaMalloc(output)");
            _context->setTensorAddress(name.c_str(), binding.device);
            _outputBindings.push_back(binding);
        }

        _results.resize(_outputBindings.size());
    }

    const std::vector<Tensor>& TensorRtBackend::run(const Tensor& input)
    {
        const size_t inputBytes = input.data.size() * sizeof(float);
        if (inputBytes > _inputBinding.capacityBytes) {
            throw std::runtime_error("Input larger than the engine's maximum shape (raise maxBatch)");
        }
        if (!_context->setInputShape(_inputBinding.name.c_str(), toDims(input.shape))) {
            throw std::runtime_error("Input shape does not match the engine");
        }

        auto stream = static_cast<cudaStream_t>(_stream);

        checkCuda(cudaMemcpyAsync(_inputBinding.device, input.data.data(), inputBytes,
                                  cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(H2D)");

        if (!_context->enqueueV3(stream)) {
            throw std::runtime_error("enqueueV3 failed");
        }

        for (size_t i = 0; i < _outputBindings.size(); ++i) {
            const auto shape = _context->getTensorShape(_outputBindings[i].name.c_str());
            _results[i].shape = toVector(shape);
            _results[i].data.resize(volume(shape));

            checkCuda(cudaMemcpyAsync(_results[i].data.data(), _outputBindings[i].device,
                                      _results[i].data.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(D2H)");
        }

        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        return _results;
    }
}
