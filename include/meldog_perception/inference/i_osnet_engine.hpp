#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace meldog_perception::inference {

    class IOsNetEngine
    {

    public:

        virtual ~IOsNetEngine() = default;

        virtual std::vector<std::vector<float>> infer(const std::vector<std::vector<float>>& data) = 0;

        virtual std::pair<std::size_t, std::size_t> getInputDim() const = 0;

        virtual std::size_t getMaxBatchSize() const = 0;

        virtual std::size_t getEmbeddingSize() const = 0;
    };

    std::unique_ptr<IOsNetEngine> createOsNetEngine(const std::string& networkPath);
}
