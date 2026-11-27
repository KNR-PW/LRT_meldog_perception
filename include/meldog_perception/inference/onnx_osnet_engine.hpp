#pragma once 

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "meldog_perception/inference/i_osnet_engine.hpp"

namespace Ort {
    struct Env;
    struct Session;
}

namespace meldog_perception::inference {

    class OnnxOsNetEngine final : public IOsNetEngine
    {

    public: 

        explicit  OnnxOsNetEngine(const std::string& networkPath);
        
        ~OnnxOsNetEngine() override;

        std::vector<std::vector<float>> infer(const std::vector<std::vector<float>>& data) override;

        std::pair<std::size_t, std::size_t> getInputDim() const override;
        std::size_t getMaxBatchSize() const override { return _maxBatch; }
        std::size_t getEmbeddingSize() const override { return _embeddingSize; }

    private:

        void initializeNetwork(const std::string& networkPath);

        void validateTensors();

    private:
        std::unique_ptr<Ort::Env> _ortEnv;
        std::unique_ptr<Ort::Session> _ortSession;

        std::string _inputName;
        std::string _outputName;

        std::size_t _maxBatch = 0;
        std::size_t _sampleSize = 0;
        std::size_t _embeddingSize = 0;
        std::size_t _inputHeight = 0;
        std::size_t _inputWidth = 0;
    };
}