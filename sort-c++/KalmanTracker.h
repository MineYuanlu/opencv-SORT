///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef OPENCV_SORT_KALMAN_H
#define OPENCV_SORT_KALMAN_H 2

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


namespace SORT {
// This class represents the internel state of individual tracked objects observed as bounding box.
//此类表示作为边界框观察到的单个跟踪对象的内部状态。
    class KalmanTracker {
    public:
        using StateType = Rect_<float>;

        explicit KalmanTracker(int &kf_count) {
            init_kf(StateType());
            m_id = kf_count;
            //kf_count++;
        }

        KalmanTracker(StateType initRect, int &kf_count) {
            init_kf(initRect);
            m_id = kf_count;
            kf_count++;
        }

        ~KalmanTracker() { m_history.clear(); }

        StateType predict();

        void update(StateType stateMat);

        StateType get_state() const;


        int m_time_since_update = 0;
        int m_hits = 0;
        int m_hit_streak = 0;
        int m_age = 0;
        int m_id;

    private:
        void init_kf(const StateType &stateMat);

        static StateType get_rect_xysr(float cx, float cy, float s, float r);

        cv::KalmanFilter kf;
        cv::Mat measurement;

        std::vector<StateType> m_history;
    };
}

#endif //OPENCV_SORT_KALMAN_H