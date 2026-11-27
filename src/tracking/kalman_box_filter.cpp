#include "meldog_perception/tracking/kalman_box_filter.hpp"

namespace meldog_perception::tracking {

    namespace {

        // ByteTrack's tuning, relative to the box height.
        constexpr double kStdWeightPosition = 1.0 / 20.0;
        constexpr double kStdWeightVelocity = 1.0 / 160.0;

        using Mat44 = cv::Matx<double, 4, 4>;
        using Mat48 = cv::Matx<double, 4, 8>;
        using Mat84 = cv::Matx<double, 8, 4>;

        double square(double value) { return value * value; }
    }

    KalmanBoxFilter::KalmanBoxFilter(const cv::Rect2f& box)
    {
        const cv::Vec4d z = toMeasurement(box);
        for (int i = 0; i < 4; ++i) {
            _mean[i] = z[i];
            _mean[i + 4] = 0.0;
        }

        const double h = z[3];
        const StateVector stdDev(
            2.0 * kStdWeightPosition * h, 2.0 * kStdWeightPosition * h, 1e-2, 2.0 * kStdWeightPosition * h,
            10.0 * kStdWeightVelocity * h, 10.0 * kStdWeightVelocity * h, 1e-5, 10.0 * kStdWeightVelocity * h);

        _covariance = StateCovariance::zeros();
        for (int i = 0; i < 8; ++i) {
            _covariance(i, i) = square(stdDev[i]);
        }
    }

    void KalmanBoxFilter::predict(double steps, bool freezeHeightVelocity)
    {
        if (freezeHeightVelocity) {
            _mean[7] = 0.0;
        }

        StateCovariance transition = StateCovariance::eye();
        for (int i = 0; i < 4; ++i) {
            transition(i, i + 4) = steps;
        }

        // Process noise is per step; a longer gap accumulates proportionally more.
        const double h = _mean[3];
        const StateVector stdDev(
            kStdWeightPosition * h, kStdWeightPosition * h, 1e-2, kStdWeightPosition * h,
            kStdWeightVelocity * h, kStdWeightVelocity * h, 1e-5, kStdWeightVelocity * h);
        StateCovariance processNoise = StateCovariance::zeros();
        for (int i = 0; i < 8; ++i) {
            processNoise(i, i) = square(stdDev[i]) * steps;
        }

        _mean = transition * _mean;
        _covariance = transition * _covariance * transition.t() + processNoise;
    }

    void KalmanBoxFilter::update(const cv::Rect2f& box)
    {
        const cv::Vec4d z = toMeasurement(box);
        const double h = _mean[3];

        Mat48 observation = Mat48::zeros();
        for (int i = 0; i < 4; ++i) {
            observation(i, i) = 1.0;
        }

        Mat44 measurementNoise = Mat44::zeros();
        measurementNoise(0, 0) = square(kStdWeightPosition * h);
        measurementNoise(1, 1) = square(kStdWeightPosition * h);
        measurementNoise(2, 2) = square(1e-1);
        measurementNoise(3, 3) = square(kStdWeightPosition * h);

        const Mat44 innovationCovariance = observation * _covariance * observation.t() + measurementNoise;
        const Mat84 gain = _covariance * observation.t() * innovationCovariance.inv(cv::DECOMP_CHOLESKY);
        const cv::Vec4d innovation = z - observation * _mean;

        _mean += gain * innovation;
        _covariance = _covariance - gain * innovationCovariance * gain.t();
    }

    cv::Rect2f KalmanBoxFilter::box() const
    {
        const double h = _mean[3];
        const double w = _mean[2] * h;
        return {static_cast<float>(_mean[0] - w / 2.0), static_cast<float>(_mean[1] - h / 2.0),
                static_cast<float>(w), static_cast<float>(h)};
    }

    cv::Point2f KalmanBoxFilter::velocity() const
    {
        return {static_cast<float>(_mean[4]), static_cast<float>(_mean[5])};
    }

    cv::Vec4d KalmanBoxFilter::toMeasurement(const cv::Rect2f& box)
    {
        return {box.x + box.width / 2.0, box.y + box.height / 2.0,
                static_cast<double>(box.width) / box.height, static_cast<double>(box.height)};
    }
}
