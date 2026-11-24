#pragma once 

#include <vector>
#include <cstddef>
#include <string>

#include <opencv2/core/mat.hpp>

namespace meldog_perception {

    class ReidEncoder 
    {

    public:

        ReidEncoder(const std::string& networkPath);

        std::vector<float> encode(const cv::Mat& rgbImage, const cv::Mat& mask, const cv::Rect2f bbox);

    private:

        cv::Mat preprocess(const cv::Mat& rgbImage, const cv::Mat& mask, const cv::Rect2f bbox);  

        void loadNetwork(const std::string& networkPath); 
    };
}