#include <gtest/gtest.h>

#include <algorithm>

#include "meldog_perception/tracking/linear_assignment.hpp"

using meldog_perception::tracking::AssignmentResult;
using meldog_perception::tracking::solveAssignment;

namespace {

    cv::Mat1d matrix(int rows, int cols, std::initializer_list<double> values)
    {
        cv::Mat1d m(rows, cols);
        std::copy(values.begin(), values.end(), m.begin());
        return m;
    }

    std::vector<std::pair<int, int>> sorted(std::vector<std::pair<int, int>> pairs)
    {
        std::sort(pairs.begin(), pairs.end());
        return pairs;
    }
}

TEST(LinearAssignment, MatchesDiagonal)
{
    const auto result = solveAssignment(matrix(2, 2, {0.1, 0.9, 0.8, 0.2}), 0.5);
    EXPECT_EQ(sorted(result.matches), (std::vector<std::pair<int, int>>{{0, 0}, {1, 1}}));
    EXPECT_TRUE(result.unmatchedRows.empty());
    EXPECT_TRUE(result.unmatchedCols.empty());
}

TEST(LinearAssignment, FindsGlobalOptimumNotGreedy)
{
    // Greedy would take (0,0)=0.1 and then (1,1)=0.9; the optimum is 0.2 + 0.15.
    const auto result = solveAssignment(matrix(2, 2, {0.1, 0.2, 0.15, 0.9}), 1.0);
    EXPECT_EQ(sorted(result.matches), (std::vector<std::pair<int, int>>{{0, 1}, {1, 0}}));
}

TEST(LinearAssignment, RectangularLeavesColumnsUnmatched)
{
    const auto result = solveAssignment(matrix(1, 3, {0.9, 0.3, 0.5}), 0.8);
    EXPECT_EQ(result.matches, (std::vector<std::pair<int, int>>{{0, 1}}));
    EXPECT_EQ(result.unmatchedCols, (std::vector<int>{0, 2}));
}

TEST(LinearAssignment, RejectsCostAboveLimit)
{
    const auto result = solveAssignment(matrix(1, 1, {0.95}), 0.8);
    EXPECT_TRUE(result.matches.empty());
    EXPECT_EQ(result.unmatchedRows, (std::vector<int>{0}));
    EXPECT_EQ(result.unmatchedCols, (std::vector<int>{0}));
}

TEST(LinearAssignment, PrefersOneGoodMatchOverTwoPoorOnes)
{
    // (0,0)+(1,1) = 0.75 + 0.75 = 1.5 is worse than (0,1) = 0.05 with both others
    // unmatched (0.8), so only one pair is kept.
    const auto result = solveAssignment(matrix(2, 2, {0.75, 0.05, 0.9, 0.75}), 0.8);
    EXPECT_EQ(result.matches, (std::vector<std::pair<int, int>>{{0, 1}}));
    EXPECT_EQ(result.unmatchedRows, (std::vector<int>{1}));
    EXPECT_EQ(result.unmatchedCols, (std::vector<int>{0}));
}

TEST(LinearAssignment, EmptyInput)
{
    const auto result = solveAssignment(cv::Mat1d(0, 2), 0.8);
    EXPECT_TRUE(result.matches.empty());
    EXPECT_EQ(result.unmatchedCols, (std::vector<int>{0, 1}));
}
