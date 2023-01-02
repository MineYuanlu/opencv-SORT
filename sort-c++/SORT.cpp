//
// Created by yuanlu on 2023/1/2.
//

#include "SORT.h"

namespace SORT {

    void SORT::handle(vector<TrackingBox> tbs
#if SORT_TRACKER_DEBUG
            , const cv::Mat &mat
#endif
    ) {
        frame_count++;

#if SORT_TRACKER_DEBUG
        auto start_time = cv::getTickCount();
#endif

        if (trackers.empty()) // the first frame met
        {
            // initialize kalman trackers using first detections.
            for (auto &tb: tbs) {
                KalmanTracker trk = KalmanTracker(tb.box, kf_count);
                trackers.push_back(trk);
            }
            // output the first frame detections
            for (size_t id = 0; id < tbs.size(); ++id) {
                TrackingBox &tb = tbs[id];
                frameTrackingResult.emplace_back(static_cast<int>(id + 1), tb.box);
            }
            return;
        } else {

            ///////////////////////////////////////
            // 3.1. get predicted locations from existing trackers.
            predictedBoxes.clear();

            for (auto it = trackers.begin(); it != trackers.end();) {
                Rect_<float> pBox = (*it).predict();
                if (pBox.x >= 0 && pBox.y >= 0) {
                    predictedBoxes.push_back(pBox);
                    it++;
                } else {
                    it = trackers.erase(it);
                    //cerr << "Box invalid at frame: " << frame_count << endl;
                }
            }

            ///////////////////////////////////////
            // 3.2. associate detections to tracked object (both represented as bounding boxes)
            // dets : detFrameData[fi]
            trkNum = predictedBoxes.size();
            detNum = tbs.size();

            iouMatrix.resize(trkNum * detNum, 0);

            for (unsigned int i = 0; i < trkNum; i++) { // compute iou matrix as a distance matrix
                for (unsigned int j = 0; j < detNum; j++) {
                    // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                    iouMatrix[i * detNum + j] = 1 - GetIOU(predictedBoxes[i], tbs[j].box);
                }
            }

            // solve the assignment problem using hungarian algorithm.
            // the resulting assignment is [track(prediction) : detection], with len=preNum
            assignment.clear();
            HungarianAlgorithm::Solve(iouMatrix, trkNum, detNum, assignment);

            // find matches, unmatched_detections and unmatched_predictions
            unmatchedTrajectories.clear();
            unmatchedDetections.clear();
            allItems.clear();
            matchedItems.clear();

            if (detNum > trkNum) { //	there are unmatched detections

                for (unsigned int n = 0; n < detNum; n++)
                    allItems.insert(static_cast<int>(n));

                for (unsigned int i = 0; i < trkNum; ++i)
                    matchedItems.insert(assignment[i]);

                set_difference(allItems.begin(), allItems.end(),
                               matchedItems.begin(), matchedItems.end(),
                               insert_iterator<decltype(unmatchedDetections)>(unmatchedDetections,
                                                                              unmatchedDetections.begin()));
            } else if (detNum < trkNum) {// there are unmatched trajectory/predictions

                for (unsigned int i = 0; i < trkNum; ++i)
                    if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                        unmatchedTrajectories.insert(static_cast<int>(i));
            }// else;

            // filter out matched with low IOU
            matchedPairs.clear();
            for (unsigned int i = 0; i < trkNum; ++i) {
                if (assignment[i] == -1) // pass over invalid values
                    continue;
                if (1 - iouMatrix[i * detNum + assignment[i]] < iouThreshold) {
                    unmatchedTrajectories.insert(static_cast<int>(i));
                    unmatchedDetections.insert(assignment[i]);
                } else
                    matchedPairs.emplace_back(i, assignment[i]);
            }

            ///////////////////////////////////////
            // 3.3. updating trackers

            // update matched trackers with assigned detections.
            // each prediction is corresponding to a tracker
            int detIdx, trkIdx;
            for (auto &matchedPair: matchedPairs) {
                trkIdx = matchedPair.x;
                detIdx = matchedPair.y;
                trackers[trkIdx].update(tbs[detIdx].box);
            }

            // create and initialise new trackers for unmatched detections
            for (auto umd: unmatchedDetections) {
                trackers.emplace_back(tbs[umd].box, kf_count);
            }

            // get trackers' output
            frameTrackingResult.clear();
            for (auto it = trackers.begin(); it != trackers.end();) {
                if (((*it).m_time_since_update < 1) &&
                    ((*it).m_hit_streak >= min_hits || frame_count <= min_hits)) {
                    TrackingBox res;
                    res.box = (*it).get_state();
                    res.id = (*it).m_id + 1;
                    frameTrackingResult.push_back(res);
                    it++;
                } else
                    it++;

                // remove dead tracklet
                if (it != trackers.end() && (*it).m_time_since_update > max_age)
                    it = trackers.erase(it);
            }
        }


#if SORT_TRACKER_DEBUG
        if (mat.empty()) return;
        auto fps = 1 / (static_cast<double>(cv::getTickCount() - start_time) / cv::getTickFrequency());
        static auto randColor = getRandomColor(20);

        for (auto tb: frameTrackingResult) {
            cv::rectangle(mat, tb.box, randColor[tb.id % randColor.size()], 2, 8, 0);
            static const auto color_id = cv::Scalar(255, 0, 255);//帧率
            cv::putText(mat, to_string(tb.id), cv::Point2f(tb.box.x, tb.box.y + 15),
                        cv::FONT_HERSHEY_COMPLEX, 0.4, color_id);
        }

        static const auto color_fps = cv::Scalar(255, 0, 255);//帧率
        cv::putText(mat,
                    "fps: " + std::to_string((int) fps),
                    cv::Size(0, 30), cv::FONT_HERSHEY_COMPLEX, 0.8, color_fps);
        imshow("sort tracker debug", mat);
        cv::waitKey(10);
#endif
    }
}
