///////////////////////////////////////////////////////////////////////////////
// Hungarian.h: Header file for Class HungarianAlgorithm.
// 
// This is a C++ wrapper with slight modification of a hungarian algorithm implementation by Markus Buehren.
// The original implementation is a few mex-functions for use in MATLAB, found here:
// http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem
// 
// Both this code and the orignal code are published under the BSD license.
// by Cong Ma, 2016
// 

#ifndef OPENCV_SORT_HUNGARIAN_ALGORITHM_H
#define OPENCV_SORT_HUNGARIAN_ALGORITHM_H 2

#include <iostream>
#include <vector>

using namespace std;

namespace SORT {
    class HungarianAlgorithm {
    public:
        HungarianAlgorithm() = delete;

        static double Solve(const vector<double> &DistMatrix, const size_t &nRows, const size_t &nCols,
                            vector<int> &Assignment);

    private:
        static void
        assignmentoptimal(int *assignment, double *cost, double *distMatrix, size_t nOfRows, size_t nOfColumns);

        static void buildassignmentvector(int *assignment, const bool *starMatrix, size_t nOfRows, size_t nOfColumns);

        static void
        computeassignmentcost(const int *assignment, double *cost, const double *distMatrix, size_t nOfRows);

        static void
        step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
               bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);

        static void
        step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
               bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);

        static void step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
                          bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);

        static void step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
                          bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim,
                          size_t row, size_t col);

        static void step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
                          bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
    };
}

#endif//OPENCV_SORT_HUNGARIAN_ALGORITHM_H