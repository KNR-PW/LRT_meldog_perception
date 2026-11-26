#include "meldog_perception/reid_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace meldog_perception {

    ReidEncoder::ReidEncoder(const std::string& networkPath, const ReidConfig& config) : _soNetModel(networkPath), _config(config)
    {

    }

    std::vector<std::vector<float>> ReidEncoder::encode(const cv::Mat& rgbImage, const cv::Mat& mask, const std::vector<cv::Rect2f>& bboxs)
    {
        if (rgbImage.empty() || rgbImage.type() != CV_8UC3) {
            throw std::invalid_argument("[ReidEncoder] encode() expects a non-empty 8-bit RGB image");
        }

        if (!mask.empty() && (mask.size() != rgbImage.size() || mask.type() != CV_8UC1)) {
            throw std::invalid_argument("[ReidEncoder] Mask must be CV_8UC1 with the same size as the image");
        }

        std::vector<std::vector<float>> embeddings(bboxs.size());

        const cv::Rect imageRect(0, 0, rgbImage.cols, rgbImage.rows);

        std::vector<std::vector<float>> samples;
        std::vector<std::size_t> sampleToBox;     
        samples.reserve(bboxs.size());
        sampleToBox.reserve(bboxs.size());

        for (std::size_t i = 0; i < bboxs.size(); ++i) {
            const cv::Rect rect = cv::Rect(bboxs[i]) & imageRect;

            // Skip boxes with no area or too small to give a reliable embedding (people far away);
            // they keep an empty entry in `embeddings`. Size is checked after clipping to the image.
            if (rect.empty() ||
                static_cast<std::size_t>(rect.width) < _config.minBoxWidth ||
                static_cast<std::size_t>(rect.height) < _config.minBoxHeight) {
                continue;
            }

            samples.push_back(preprocess(rgbImage, mask, rect));
            sampleToBox.push_back(i);
        }

        const std::size_t maxBatch = _soNetModel.getMaxBatchSize();

        for (std::size_t start = 0; start < samples.size(); start += maxBatch) {
            const std::size_t end = std::min(start + maxBatch, samples.size());

            std::vector<std::vector<float>> batch(std::make_move_iterator(samples.begin() + static_cast<std::ptrdiff_t>(start)),
                                                  std::make_move_iterator(samples.begin() + static_cast<std::ptrdiff_t>(end)));

            std::vector<std::vector<float>> features = _soNetModel.infer(batch);

            for (std::size_t k = 0; k < features.size(); ++k) {
                std::vector<float>& embedding = features[k];

                double squaredSum = 0.0;
                for (const float value : embedding) {
                    squaredSum += static_cast<double>(value) * value;
                }

                const float norm = static_cast<float>(std::sqrt(squaredSum));
                if (norm > 1e-12F) {
                    for (float& value : embedding) {
                        value /= norm;
                    }
                }

                embeddings[sampleToBox[start + k]] = std::move(embedding);
            }
        }

        return embeddings;
    }

    std::vector<float> ReidEncoder::preprocess(const cv::Mat& rgbImage, const cv::Mat& mask, const cv::Rect& bbox) const
    {
        // ImageNet statistics used when OSNet was trained (RGB order)
        const cv::Scalar kMean(0.485, 0.456, 0.406);
        const cv::Scalar kStd(0.229, 0.224, 0.225);

        // Mean colour in 0..255: masked-out background becomes exactly 0 after normalization
        const cv::Scalar kMeanColor(0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0);

        const auto [inputHeight, inputWidth] = _soNetModel.getInputDim();

        cv::Mat crop;
        if (mask.empty()) {
            crop = rgbImage(bbox);
        } else {
            crop = cv::Mat(bbox.size(), CV_8UC3, kMeanColor);
            rgbImage(bbox).copyTo(crop, mask(bbox));
        }

        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(inputHeight, inputWidth), 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat normalized;
        resized.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
        cv::subtract(normalized, kMean, normalized);
        cv::divide(normalized, kStd, normalized);

        //    HWC -> CHW: three single-channel Mats pointing into the sample buffer,
        //    cv::split writes every channel straight into its plane (no extra copy)
        std::vector<float> sample(static_cast<std::size_t>(3 * inputHeight * inputWidth));
        const std::size_t planeSize = static_cast<std::size_t>(inputHeight * inputWidth);

        cv::Mat planes[3] = {
            cv::Mat(inputHeight, inputWidth, CV_32FC1, sample.data()),
            cv::Mat(inputHeight, inputWidth, CV_32FC1, sample.data() + planeSize),
            cv::Mat(inputHeight, inputWidth, CV_32FC1, sample.data() + 2 * planeSize),
        };
        cv::split(normalized, planes);

        return sample;
    }
}
