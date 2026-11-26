#include "meldog_perception/tracking/linear_assignment.hpp"

#include <limits>

namespace meldog_perception::tracking {

    namespace {

        // Hungarian algorithm (shortest augmenting paths with potentials) for a
        // square matrix. Returns the column assigned to each row. O(n^3).
        std::vector<int> hungarian(const cv::Mat1d& a)
        {
            const int n = a.rows;
            const double inf = std::numeric_limits<double>::infinity();

            // 1-based arrays; column 0 is a virtual start node.
            std::vector<double> u(n + 1, 0.0);
            std::vector<double> v(n + 1, 0.0);
            std::vector<int> rowOfCol(n + 1, 0);
            std::vector<int> way(n + 1, 0);

            for (int row = 1; row <= n; ++row) {
                rowOfCol[0] = row;
                int col0 = 0;
                std::vector<double> minValue(n + 1, inf);
                std::vector<bool> used(n + 1, false);

                do {
                    used[col0] = true;
                    const int row0 = rowOfCol[col0];
                    double delta = inf;
                    int col1 = 0;

                    for (int col = 1; col <= n; ++col) {
                        if (used[col]) {
                            continue;
                        }
                        const double reduced = a(row0 - 1, col - 1) - u[row0] - v[col];
                        if (reduced < minValue[col]) {
                            minValue[col] = reduced;
                            way[col] = col0;
                        }
                        if (minValue[col] < delta) {
                            delta = minValue[col];
                            col1 = col;
                        }
                    }

                    for (int col = 0; col <= n; ++col) {
                        if (used[col]) {
                            u[rowOfCol[col]] += delta;
                            v[col] -= delta;
                        } else {
                            minValue[col] -= delta;
                        }
                    }
                    col0 = col1;
                } while (rowOfCol[col0] != 0);

                do {
                    const int col1 = way[col0];
                    rowOfCol[col0] = rowOfCol[col1];
                    col0 = col1;
                } while (col0 != 0);
            }

            std::vector<int> colOfRow(n, -1);
            for (int col = 1; col <= n; ++col) {
                if (rowOfCol[col] > 0) {
                    colOfRow[rowOfCol[col] - 1] = col - 1;
                }
            }
            return colOfRow;
        }
    }

    AssignmentResult solveAssignment(const cv::Mat1d& cost, double costLimit)
    {
        const int rows = cost.rows;
        const int cols = cost.cols;
        AssignmentResult result;

        if (rows == 0 || cols == 0) {
            for (int r = 0; r < rows; ++r) {
                result.unmatchedRows.push_back(r);
            }
            for (int c = 0; c < cols; ++c) {
                result.unmatchedCols.push_back(c);
            }
            return result;
        }

        // Extended square matrix: every row and column can also go to a dummy
        // partner for costLimit / 2, so a real pair costing more than costLimit
        // is never better than leaving both unmatched.
        const int n = rows + cols;
        cv::Mat1d extended(n, n, 0.0);
        cost.copyTo(extended(cv::Rect(0, 0, cols, rows)));
        extended(cv::Rect(cols, 0, rows, rows)).setTo(costLimit / 2.0);
        extended(cv::Rect(0, rows, cols, cols)).setTo(costLimit / 2.0);

        const std::vector<int> colOfRow = hungarian(extended);

        std::vector<bool> colMatched(cols, false);
        for (int r = 0; r < rows; ++r) {
            const int c = colOfRow[r];
            if (c >= 0 && c < cols && cost(r, c) <= costLimit) {
                result.matches.emplace_back(r, c);
                colMatched[c] = true;
            } else {
                result.unmatchedRows.push_back(r);
            }
        }
        for (int c = 0; c < cols; ++c) {
            if (!colMatched[c]) {
                result.unmatchedCols.push_back(c);
            }
        }
        return result;
    }
}
