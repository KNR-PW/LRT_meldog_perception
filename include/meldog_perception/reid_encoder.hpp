#pragma once 

#include <vector>
#include <cstddef>
#include <string>

#include <opencv2/core/mat.hpp>

#include "so_net_engine.hpp"

namespace meldog_perception {

    struct ReidConfig {
        std::size_t minBoxHeight = 32; // px
        std::size_t minBoxWidth = 12; // px 
    };

    class ReidEncoder 
    {

    public:

        ReidEncoder(const std::string& networkPath, const ReidConfig& config);

        std::vector<std::vector<float>> encode(const cv::Mat& rgbImage, const cv::Mat& mask, const std::vector<cv::Rect2f>& bboxs);

    private:

        std::vector<float> preprocess(const cv::Mat& rgbImage, const cv::Mat& mask, const cv::Rect& bbox) const;
 
    private:
        SoNetEngine _soNetModel; 
        ReidConfig _config;
    };
}