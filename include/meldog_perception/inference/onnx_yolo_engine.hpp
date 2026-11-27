#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "meldog_perception/inference/i_yolo_engine.hpp"

namespace Ort {
    struct Env;
    struct Session;
}

namespace meldog_perception::inference {

    class OnnxYoloEngine final : public IYoloEngine
    {

    public:

        explicit OnnxYoloEngine(const std::string& networkPath);

        ~OnnxYoloEngine() override;

        std::vector<float> infer(const std::vector<float>& data) override;

        std::pair<std::size_t, std::size_t> getInputDim() const override;
        std::size_t getDetectionCount() const override { return _detectionCount; }
        std::size_t getDetectionSize() const override { return _detectionSize; }

    private:

        void initializeNetwork(const std::string& networkPath);

        void validateTensors();

    private:

        std::unique_ptr<Ort::Env> _ortEnv;
        std::unique_ptr<Ort::Session> _ortSession;

        std::string _inputName;
        std::string _outputName;

        std::size_t _inputHeight = 0;
        std::size_t _inputWidth = 0;
        std::size_t _sampleSize = 0;
        std::size_t _detectionCount = 0;
        std::size_t _detectionSize = 0;
    };
}
