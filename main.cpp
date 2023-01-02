///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//  
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <fstream>
#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.

#include "SORT.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


struct TrackingBox0 {
    int frame{};
    int id{};
    Rect_<float> box;
};

void TestSORT(const string &seqName, bool display);


int main() {
    vector<string> sequences = {"PETS09-S2L1", "TUD-Campus", "TUD-Stadtmitte", "ETH-Bahnhof", "ETH-Sunnyday",
                                "ETH-Pedcross2", "KITTI-13", "KITTI-17", "ADL-Rundle-6", "ADL-Rundle-8", "Venice-2"};
    for (const auto &seq: sequences)
        TestSORT(seq, true);
    //TestSORT("PETS09-S2L1", true);

    return 0;
}


void TestSORT(const string &seqName, bool display) {
    cout << "Processing " << seqName << "..." << endl;

    // 1. read detection file
    ifstream detectionFile;
    string detFileName = "../assert/data/" + seqName + "/det.txt";
    detectionFile.open(detFileName);

    if (!detectionFile.is_open()) {
        cerr << "Error: can not find file " << detFileName << endl;
        return;
    }

    string detLine;
    istringstream ss;
    vector<TrackingBox0> detData;
    char ch;
    float tpx, tpy, tpw, tph;

    while (getline(detectionFile, detLine)) {
        TrackingBox0 tb;

        ss.str(detLine);
        ss >> tb.frame >> ch >> tb.id >> ch;
        ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph;
        ss.str("");

        tb.box = Rect_<float>(Point_<float>(tpx, tpy), Point_<float>(tpx + tpw, tpy + tph));
        detData.push_back(tb);
    }
    detectionFile.close();

    // 2. group detData by frame
    int maxFrame = 0;
    for (auto tb: detData) // find max frame number
    {
        if (maxFrame < tb.frame)
            maxFrame = tb.frame;
    }

    vector<vector<SORT::TrackingBox>> detFrameData;
    vector<SORT::TrackingBox> tempVec;
    for (int fi = 0; fi < maxFrame; fi++) {
        for (auto tb: detData)
            if (tb.frame == fi + 1) // frame num starts from 1
                tempVec.emplace_back(tb.id, tb.box);
        detFrameData.push_back(tempVec);
        tempVec.clear();
    }

#if SORT_TRACKER_DEBUG
    Rect_<float> rect;
    for (const auto &tbs: detFrameData)
        for (const auto &tb: tbs)
            rect |= tb.box;
#endif

    SORT::SORT sort;
    for (const auto &tbs: detFrameData) {
        if (display) {
#if SORT_TRACKER_DEBUG
            cv::Mat mat(cv::Size(rect.br() + cv::Point2f(1, 1)), CV_8UC3, cv::Scalar());
#endif
            sort.handle(tbs
#if SORT_TRACKER_DEBUG
                    , mat
#endif
            );
        } else sort.handle(tbs);
    }
}
