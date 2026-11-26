#pragma once

#include <opencv2/core.hpp>

namespace meldog_perception::tracking {

    // Constant-velocity Kalman filter for one bounding box, as in ByteTrack/DeepSORT.
    // State: [cx, cy, aspect (w/h), h, and their velocities]. Noise scales with the
    // box height, so the same tuning works for near and far people.
    //
    // Time is measured in "steps": one step is one frame at the tracker's reference
    // rate. Velocities are therefore in pixels per step.
    class KalmanBoxFilter
    {

    public:

        using StateVector = cv::Vec<double, 8>;
        using StateCovariance = cv::Matx<double, 8, 8>;

        explicit KalmanBoxFilter(const cv::Rect2f& box);

        // freezeHeightVelocity: ByteTrack stops height changes while a track is lost,
        // so a box that was shrinking does not collapse during an occlusion.
        void predict(double steps, bool freezeHeightVelocity = false);
        void update(const cv::Rect2f& box);

        cv::Rect2f box() const;
        cv::Point2f velocity() const;   // box center, pixels per step

        const StateVector& mean() const { return _mean; }
        const StateCovariance& covariance() const { return _covariance; }

    private:

        static cv::Vec4d toMeasurement(const cv::Rect2f& box);

    private:

        StateVector _mean;
        StateCovariance _covariance;
    };
}
