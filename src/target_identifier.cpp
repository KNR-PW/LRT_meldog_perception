#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>

#include "meldog_perception/target_identifier.hpp"

namespace meldog_perception {

    TargetIdentifier::TargetIdentifier(const TargetConfig& config) : _config(config)
    {
        validateConfig(config);
    }

    std::optional<std::size_t> TargetIdentifier::update(std::vector<std::vector<float>> embeddings)
    {
        if (embeddings.empty()) {
            return std::nullopt;
        }

        if (_targetEmbedding.empty()) {
            if (embeddings.size() != 1 || embeddings.front().empty()) {
                return std::nullopt;
            }

            _targetEmbedding = std::move(embeddings.front());
            return 0;
        }

        std::vector<double> scores;
        scores.reserve(embeddings.size());

        for (const auto& embedding : embeddings) {
            scores.emplace_back(calculateMatchScore(embedding));
        }

        const auto best = std::max_element(scores.begin(), scores.end());
        if (*best < _config.matchThreshold) {
            return std::nullopt;
        }

        const std::size_t targetIdx = static_cast<std::size_t>(std::distance(scores.begin(), best));

        _targetEmbedding = std::move(embeddings[targetIdx]);

        return targetIdx;
    }

    double TargetIdentifier::calculateMatchScore(const std::vector<float>& embedding) const
    {
        if (embedding.size() != _targetEmbedding.size()) {
            return 0.0;
        }

        return std::inner_product(embedding.begin(), embedding.end(), _targetEmbedding.begin(), 0.0);
    }

    void TargetIdentifier::validateConfig(const TargetConfig& config)
    {
        if (!std::isfinite(config.matchThreshold) || config.matchThreshold <= 0.0 || config.matchThreshold > 1.0) {
            throw std::invalid_argument("[TargetIdentifier] matchThreshold must be in range (0, 1], got " + std::to_string(config.matchThreshold));
        }
    }

    void TargetIdentifier::reset()
    {
        _targetEmbedding.clear();
    }
}