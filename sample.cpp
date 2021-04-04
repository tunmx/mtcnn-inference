#include "FaceDetector.h"
#include "FaceRecognition.h"
#include <iostream>
#include "opencv2/opencv.hpp"



using namespace std;

void drawRect(cv::Mat &img, FaceInfo &faceInfo) {
    cv::rectangle(img, Point(faceInfo.bbox.xmin, faceInfo.bbox.ymin), Point(faceInfo.bbox.xmax, faceInfo.bbox.ymax), cv::Scalar(0, 0, 255));
}

void drawRectReg(cv::Mat &img, FaceInfo &faceInfo) {
    cv::rectangle(img, Point(faceInfo.bbox_reg[0], faceInfo.bbox_reg[1]), Point(faceInfo.bbox_reg[2], faceInfo.bbox_reg[3]), cv::Scalar(0, 0, 255));
}

int main(int argc, char **argv) {
    string model_filename = "/Users/yh-mac/fromgithub/Keras-YOLOv4/face_module/face_mobile.mlz";
    FaceRecognition faceRecognition(model_filename);
    cv::Mat img = cv::imread("/Users/yh-mac/Desktop/bangzi.jpeg");
    const int min_size = 50;
//    const float threshold[3] = {0.7f, 0.6f, 0.6f};
    const float threshold[3] = {0.7f, 0.6f, 0.6f};
    const float factor = 0.709;
    TickMeter tm;
    tm.start();
//    vector<FaceInfo> faceInfo = faceRecognition.faceDetector->ProposalNet(img, min_size, threshold[0], factor);
//    vector<FaceInfo> res = faceRecognition.faceDetector->NextStage(img, faceInfo, 24, 24, 2, threshold[1]);
    vector<FaceInfo> faceInfo = faceRecognition.faceDetector->Detect_mtcnn(img, 50, threshold, factor, 3);
    tm.stop();
    std::cout << tm.getTimeSec() << endl;//输出是s
    cout << "size: " << faceInfo.size() << endl;
    for (int i = 0; i < faceInfo.size(); ++i) {
//        cout << faceInfo[i].bbox.score << endl;
        drawRect(img, faceInfo[i]);
    }
    cv::imshow("s", img);
    cv::waitKey(0);
    return 0;
}