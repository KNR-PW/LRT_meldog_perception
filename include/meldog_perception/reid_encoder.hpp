#pragma once 

#include <vector>
#include <cstddef>
#include <string>

#include <opencv2/core/mat.hpp>

#include "meldog_perception/inference/i_osnet_engine.hpp"

namespace meldog_perception {

    struct ReidConfig {
        std::size_t minBoxHeight = 32; // px
        std::size_t minBoxWidth = 12; // px
        std::size_t maxBatchSize = 8; // crops per inference call, capped by the engine's own limit (TensorRT profile kMAX)
    };

    class ReidEncoder 
    {

    public:

        ReidEncoder(const std::string& networkPath, const ReidConfig& config);

        std::vector<std::vector<float>> encode(const cv::Mat& rgbImage, const cv::Mat& mask, const std::vector<cv::Rect2f>& bboxs);

    private:

        std::vector<float> preprocess(const cv::Mat& rgbImage, const cv::Mat& mask, const cv::Rect& bbox) const;

        void validateConfig(const ReidConfig& config);

    private:
        std::unique_ptr<inference::IOsNetEngine> _osNetengine; 
        ReidConfig _config;
    };
}