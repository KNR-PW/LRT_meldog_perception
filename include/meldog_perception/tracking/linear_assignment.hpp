#pragma once

#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace meldog_perception::tracking {

    struct AssignmentResult
    {
        std::vector<std::pair<int, int>> matches;   // (row, col)
        std::vector<int> unmatchedRows;
        std::vector<int> unmatchedCols;
    };

    // Minimum-cost matching of rows to columns where a pair is allowed only if its
    // cost is <= costLimit. Leaving a row and a column unmatched together costs
    // costLimit, as in lap.lapjv(extend_cost=True, cost_limit=...) used by ByteTrack.
    // cost: CV_64F, rows x cols (either may be 0).
    AssignmentResult solveAssignment(const cv::Mat1d& cost, double costLimit);
}
