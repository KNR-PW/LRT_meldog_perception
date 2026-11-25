#include "meldog_perception/reid_encoder.hpp"

namespace meldog_perception {

    ReidEncoder::ReidEncoder(const std::string& networkPath)
    {
        // TO DO
    }

    std::vector<float> ReidEncoder::encode(const cv::Mat& rgbImage, const cv::Mat& mask, const cv::Rect2f bbox)
    {
        cv::Mat processedMat = preprocess(rgbImage, mask, bbox);

        if (processedMat.empty()) {

        }
        
        // TO DO
    }

    cv::Mat ReidEncoder::preprocess(const cv::Mat& rgbImage, const cv::Mat& mask, const cv::Rect2f bbox)
    {
        if (rgbImage.empty() || mask.empty() || bbox.empty()) {
            return cv::Mat{};
        }

        if (rgbImage.size() != mask.size()) {
            return cv::Mat{};
        }

        cv::Mat result = cv::Mat::zeros(bbox.size(), rgbImage.type());

        rgbImage(bbox).copyTo(result, mask(bbox) == 255);

        return result;
    }
}