//
// Created by yuanlu on 2023/1/2.
//

#ifndef OPENCV_SORT_SORT_H
#define OPENCV_SORT_SORT_H

#include "KalmanTracker.h"
#include "Hungarian.h"
#include <set>

#ifndef SORT_TRACKER_DEBUG
#ifdef DEBUG
#define SORT_TRACKER_DEBUG (DEBUG && 1)
#else
#define SORT_TRACKER_DEBUG 1
#endif
#endif
namespace SORT {
    struct TrackingBox {
        int id{};
        Rect_<float> box;

        TrackingBox() = default;

        TrackingBox(int id, Rect_<float> box) : id(id), box(box) {}
    };

    class SORT {
    public:

        static const constexpr int min_hits = 3;
        static const constexpr int max_age = 1;
        static const constexpr double iouThreshold = 0.3;

        int frame_count = 0;
        vector<KalmanTracker> trackers;

        // variables used in the for-loop
        vector<Rect_<float>> predictedBoxes;
        vector<double> iouMatrix;
        vector<int> assignment;
        set<signed int> unmatchedDetections;
        set<signed int> unmatchedTrajectories;
        set<signed int> allItems;
        set<signed int> matchedItems;
        vector<cv::Point> matchedPairs;
        vector<TrackingBox> frameTrackingResult;
        unsigned int trkNum = 0;
        unsigned int detNum = 0;

        int kf_count = 0;

        SORT();

        void handle(vector<TrackingBox> tbs
#if SORT_TRACKER_DEBUG
                , const cv::Mat &mat = cv::Mat()
#endif
        );

    private:

        // 计算两个边界框之间的IOU
        static double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt) {
            float in = (bb_test & bb_gt).area();
            float un = bb_test.area() + bb_gt.area() - in;

            if (un < DBL_EPSILON)return 0;

            return (double) (in / un);
        }

#if SORT_TRACKER_DEBUG

        static auto getRandomColor(std::size_t num) {
            RNG rng(0xFFFFFFFF);
            vector<Scalar_<int>> randColor(num);
            for (auto &color: randColor) rng.fill(color, RNG::UNIFORM, 0, 256);
            return randColor;
        }

#endif

    };
}


#endif //OPENCV_SORT_SORT_H
