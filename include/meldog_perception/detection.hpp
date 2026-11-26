#pragma once

#include <opencv2/core.hpp>

namespace meldog_perception {

    struct Detection
    {
        cv::Rect2f box;     // pixels of the original image
        float score = 0.0F;
        int classId = -1;
    };
}
