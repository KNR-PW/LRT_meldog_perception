#include <gtest/gtest.h>

#include "meldog_perception/tracking/kalman_box_filter.hpp"

using meldog_perception::tracking::KalmanBoxFilter;

TEST(KalmanBoxFilter, StartsAtTheMeasurement)
{
    const cv::Rect2f box(100.0F, 50.0F, 40.0F, 120.0F);
    const KalmanBoxFilter filter(box);
    const cv::Rect2f out = filter.box();
    EXPECT_NEAR(out.x, box.x, 1e-3);
    EXPECT_NEAR(out.y, box.y, 1e-3);
    EXPECT_NEAR(out.width, box.width, 1e-3);
    EXPECT_NEAR(out.height, box.height, 1e-3);
    EXPECT_FLOAT_EQ(filter.velocity().x, 0.0F);
}

TEST(KalmanBoxFilter, LearnsConstantVelocity)
{
    cv::Rect2f box(100.0F, 50.0F, 40.0F, 120.0F);
    KalmanBoxFilter filter(box);
    for (int i = 0; i < 30; ++i) {
        box.x += 4.0F;
        filter.predict(1.0);
        filter.update(box);
    }
    EXPECT_NEAR(filter.velocity().x, 4.0F, 0.5F);
    EXPECT_NEAR(filter.velocity().y, 0.0F, 0.5F);

    // Two steps without a measurement continue the motion.
    filter.predict(2.0);
    EXPECT_NEAR(filter.box().x, box.x + 8.0F, 1.5F);
}

TEST(KalmanBoxFilter, UncertaintyGrowsWithoutMeasurements)
{
    KalmanBoxFilter filter(cv::Rect2f(0.0F, 0.0F, 40.0F, 120.0F));
    const double before = filter.covariance()(0, 0);
    filter.predict(10.0);
    EXPECT_GT(filter.covariance()(0, 0), before);
}
