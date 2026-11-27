#pragma once 

#include <cstddef>
#include <optional>
#include <vector>

namespace meldog_perception {

    struct TargetConfig {
        double matchThreshold = 0.6;
    };

    class TargetIdentifier {

    public:

        explicit TargetIdentifier(const TargetConfig& config);

        std::optional<std::size_t> update(std::vector<std::vector<float>> embeddings); 

        void reset();

    private:

        static void validateConfig(const TargetConfig& config);

        double calculateMatchScore(const std::vector<float>& embedding) const;

    private:
        TargetConfig _config{};
        std::vector<float> _targetEmbedding; 
    };

}